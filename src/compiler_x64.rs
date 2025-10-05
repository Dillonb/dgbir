use std::{collections::BTreeMap, marker::PhantomData};

use crate::{
    abi::{get_function_argument_registers, get_return_value_registers, get_scratch_registers, reg_constants},
    compiler::{Compiler, ConstOrReg, GenericAssembler, LiteralPool},
    ir::{BlockReference, CompareType, Constant, DataType, IRFunctionInternal},
    reg_pool::{register_type, RegPool},
    register_allocator::{alloc_for, Register, RegisterAllocations},
};
use dynasmrt::{dynasm, x64::X64Relocation, Assembler, AssemblyOffset, VecAssembler};
use log::{info, trace};

impl GenericAssembler<X64Relocation> for Assembler<X64Relocation> {
    type R = X64Relocation;
    fn new_dynamic_label(&mut self) -> dynasmrt::DynamicLabel {
        self.new_dynamic_label()
    }
}

impl GenericAssembler<X64Relocation> for VecAssembler<X64Relocation> {
    type R = X64Relocation;
    fn new_dynamic_label(&mut self) -> dynasmrt::DynamicLabel {
        self.new_dynamic_label()
    }
}

pub struct X64Compiler<'a, Ops> {
    scratch_regs: RegPool,
    func: &'a IRFunctionInternal,
    allocations: RegisterAllocations,
    entrypoint: AssemblyOffset,
    block_labels: Vec<dynasmrt::DynamicLabel>,
    phantom: PhantomData<Ops>,
}

impl<'a, Ops: GenericAssembler<X64Relocation>> Compiler<'a, X64Relocation, Ops> for X64Compiler<'a, Ops> {
    fn new_dynamic_label(ops: &mut Ops) -> dynasmrt::DynamicLabel {
        ops.new_dynamic_label()
    }

    fn offset(ops: &mut Ops) -> usize {
        let AssemblyOffset(offset) = ops.offset();
        return offset;
    }

    fn get_scratch_regs(&self) -> &RegPool {
        &self.scratch_regs
    }

    fn new(ops: &mut Ops, func: &'a mut IRFunctionInternal) -> Self {
        let allocations = alloc_for(func);

        info!("Function after allocation:\n{}", func);

        // Stack bytes used: aligned to 16 bytes
        // Note: x64 is stupid, and the CALL instruction leaves the stack pointer misaligned.
        // Take this into account.
        let misalignment = (func.stack_bytes_used + 8) % 16;
        let correction = if misalignment == 0 { 0 } else { 16 - misalignment };
        let stack_bytes_used = func.stack_bytes_used + correction;
        info!(
            "Function uses {} bytes of stack, misaligned by {}, corrected to {}",
            func.stack_bytes_used, misalignment, stack_bytes_used
        );
        func.stack_bytes_used = stack_bytes_used;

        let entrypoint = ops.offset();

        let block_labels = func.blocks.iter().map(|_| ops.new_dynamic_label()).collect();

        X64Compiler {
            entrypoint,
            scratch_regs: RegPool::new(get_scratch_registers()),
            func,
            allocations,
            block_labels,
            phantom: PhantomData,
        }
    }

    fn prologue(&self, ops: &mut Ops) {
        // Setup stack
        if self.func.stack_bytes_used > 0 {
            dynasm!(ops
                ; sub rsp, self.func.stack_bytes_used.try_into().unwrap()
            )
        }

        // Save callee-saved registers to stack
        for (reg, stack_location) in &self.allocations.callee_saved {
            match reg {
                Register::GPR(r) => {
                    assert_eq!(reg.size(), 8);
                    let ofs = self
                        .func
                        .get_stack_offset_for_location(*stack_location as u64, DataType::U64)
                        as i32;
                    dynasm!(ops
                        ; mov [rsp + ofs], Rq(*r as u8)
                    )
                }
                Register::SIMD(r) => {
                    assert_eq!(reg.size(), 16);
                    let ofs = self
                        .func
                        .get_stack_offset_for_location(*stack_location as u64, DataType::U128)
                        as i32;
                    dynasm!(ops
                        ; movdqu OWORD [rsp + ofs], Rx(*r as u8)
                    );
                }
            }
        }
    }

    fn epilogue(&self, _ops: &mut Ops) {
        trace!("Epilogue: emitting nothing");
    }

    fn jump_to_dynamic_label(&self, ops: &mut Ops, label: dynasmrt::DynamicLabel) {
        dynasm!(ops
            ; jmp =>label
        );
    }

    fn move_to_reg(&self, ops: &mut Ops, lp: &mut LiteralPool, from: ConstOrReg, to: Register) {
        trace!("move_to_reg(): Moving {:?} to {:?}", from, to);
        match (from, to) {
            (c, Register::GPR(r_to)) if c.is_const() => {
                let c = c.to_u64_const().unwrap();
                if let Ok(c) = c.try_into() {
                    dynasm!(ops
                        ; mov Rd(r_to as u8), c
                    );
                } else {
                    dynasm!(ops
                        ; mov Rq(r_to as u8), QWORD c as i64
                    )
                }
            }
            (ConstOrReg::U32(c), Register::GPR(r_to)) => {
                dynasm!(ops
                    ; mov Rd(r_to as u8), c as i32
                );
            }
            (ConstOrReg::S32(c), Register::GPR(r_to)) => {
                dynasm!(ops
                    ; mov Rd(r_to as u8), c as i32
                    ; movsx Rq(r_to as u8), Rd(r_to as u8)
                );
            }
            (ConstOrReg::U64(c), Register::GPR(r_to)) => {
                dynasm!(ops
                    ; mov Rq(r_to as u8), QWORD c as i64
                )
            }
            (ConstOrReg::GPR(r_from), Register::GPR(r_to)) => {
                dynasm!(ops
                    ; mov Rq(r_to as u8), Rq(r_from as u8)
                );
            }
            (ConstOrReg::SIMD(r_from), Register::SIMD(r_to)) => {
                dynasm!(ops
                    ; movdqa Rx(r_to as u8), Rx(r_from as u8)
                );
            }
            (ConstOrReg::F32(c), Register::SIMD(r_to)) => {
                let literal = X64Compiler::add_literal(ops, lp, Constant::F32(c));
                dynasm!(ops
                    ; movss Rx(r_to as u8), DWORD [=>literal]
                );
            }
            (ConstOrReg::SIMD(r_from), Register::GPR(r_to)) => {
                dynasm!(ops
                    ; movq Rq(r_to as u8), Rx(r_from as u8)
                );
            }
            (ConstOrReg::GPR(r_from), Register::SIMD(r_to)) => {
                dynasm!(ops
                    ; movq Rx(r_to as u8), Rq(r_from as u8)
                );
            }
            _ => todo!("Unimplemented move operation: {:?} to {:?}", from, to),
        }
    }

    // TODO: this is exactly the same in all compilers, figure out how to share this
    fn emit_literal_pool(&self, ops: &mut Ops, lp: LiteralPool) {
        for (literal, label) in lp.literals {
            ops.align(literal.size(), 0);
            match literal {
                Constant::U32(c) => {
                    dynasm!(ops
                        ; =>label
                        ; .u32 c
                    );
                }
                Constant::F32(c) => {
                    dynasm!(ops
                        ; =>label
                        ; .f32 *c
                    );
                }
                _ => todo!("Unsupported literal type: {:?}", literal),
            }
        }
    }

    fn on_new_block_begin(&self, ops: &mut Ops, block_index: usize) {
        // This "resolves" the block label so it can be jumped to from elsewhere in the program.
        // This should be done once per block.
        dynasm!(ops
            ; =>self.block_labels[block_index]
        );
    }

    fn get_func(&self) -> &IRFunctionInternal {
        &self.func
    }

    fn get_allocations(&self) -> &RegisterAllocations {
        &self.allocations
    }

    fn get_entrypoint(&self) -> AssemblyOffset {
        self.entrypoint
    }

    fn get_block_label(&self, block_index: usize) -> dynasmrt::DynamicLabel {
        self.block_labels[block_index]
    }

    fn branch(
        &self,
        ops: &mut Ops,
        lp: &mut LiteralPool,
        cond: &ConstOrReg,
        if_true: &BlockReference,
        if_false: &BlockReference,
    ) {
        match cond {
            ConstOrReg::GPR(c) => {
                dynasm!(ops
                    ; test Rq(*c as u8), Rq(*c as u8)
                    ; jz >if_false
                );
            }
            _ => todo!("Unsupported branch condition: {:?}", cond),
        }
        self.call_block(ops, lp, if_true);
        dynasm!(ops
            ; if_false:
        );
        self.call_block(ops, lp, if_false);
    }

    fn ret(&self, ops: &mut Ops, lp: &mut LiteralPool, value: &Option<ConstOrReg>) {
        if let Some(v) = value {
            let retval_reg = *get_return_value_registers()
                .iter()
                .find(|r| v.is_same_type_as(r))
                .unwrap();
            self.move_to_reg(ops, lp, *v, retval_reg);
        }

        // Pop callee-saved regs from stack
        // TODO: move this to the epilogue and emit a jmp to the end of the function here (to make
        // multiple returns more efficient)
        for (reg, stack_location) in &self.allocations.callee_saved {
            match *reg {
                Register::GPR(r) => {
                    assert_eq!(reg.size(), 8);
                    dynasm!(ops
                        ; mov Rq(r as u8), [rsp + self.func.get_stack_offset_for_location(*stack_location as u64, DataType::U64) as i32]
                    )
                }
                Register::SIMD(r) => {
                    assert_eq!(reg.size(), 16);
                    dynasm!(ops
                        ; movdqu Rx(r as u8), OWORD [rsp + self.func.get_stack_offset_for_location(*stack_location as u64, DataType::U128) as i32]
                    )
                }
            }
        }

        // Fix sp
        if self.func.stack_bytes_used > 0 {
            dynasm!(ops
                ; add rsp, self.func.stack_bytes_used.try_into().unwrap()
            );
        }
        dynasm!(ops
            ; ret
        );
    }

    fn add(&self, ops: &mut Ops, lp: &mut LiteralPool, tp: DataType, r_out: Register, a: ConstOrReg, b: ConstOrReg) {
        match (tp, r_out) {
            (DataType::U32 | DataType::S32, Register::GPR(r_out)) => {
                self.move_to_reg(ops, lp, a, Register::GPR(r_out));
                let b = self.materialize_as_gpr(ops, lp, b);
                dynasm!(ops
                    ; add Rd(r_out as u8), Rd(b.r() as u8)
                )
            }
            (DataType::U64 | DataType::S64, Register::GPR(r_out)) => {
                self.move_to_reg(ops, lp, a, Register::GPR(r_out));
                let b = self.materialize_as_gpr(ops, lp, b);
                dynasm!(ops
                    ; add Rq(r_out as u8), Rq(b.r() as u8)
                )
            }
            (DataType::F32, Register::SIMD(r_out)) => {
                self.move_to_reg(ops, lp, a, Register::SIMD(r_out));
                let b = self.materialize_as_simd(ops, lp, b);
                dynasm!(ops
                    ; addss Rx(r_out as u8), Rx(b.r() as u8)
                )
            }
            _ => todo!("Unsupported Add operation: {:?} + {:?} with type {:?}", a, b, tp),
        }
    }

    fn compare(
        &self,
        ops: &mut Ops,
        lp: &mut LiteralPool,
        r_out: usize,
        tp: DataType,
        a: ConstOrReg,
        cmp_type: CompareType,
        b: ConstOrReg,
    ) {
        // First, zero the output register
        dynasm!(ops
            ; xor Rd(r_out as u8), Rd(r_out as u8)
        );

        let signed = tp.is_signed();
        match tp {
            DataType::Bool => {
                assert!(matches!(cmp_type, CompareType::Equal | CompareType::NotEqual));
                let a = self.materialize_as_gpr(ops, lp, a);
                let b = self.materialize_as_gpr(ops, lp, b);
                dynasm!(ops
                    ; cmp Rq(a.r() as u8), Rq(b.r() as u8)
                );
            }
            DataType::U8
            | DataType::S8
            | DataType::U16
            | DataType::S16
            | DataType::U32
            | DataType::S32
            | DataType::U64
            | DataType::S64
            | DataType::Ptr => {
                let a = self.materialize_as_gpr(ops, lp, a);
                let b = self.materialize_as_gpr(ops, lp, b);
                dynasm!(ops
                    ; cmp Rq(a.r() as u8), Rq(b.r() as u8)
                );
            }
            DataType::U128 => todo!("Compare with 128-bit integer type"),
            DataType::F32 => {
                let a = self.materialize_as_simd(ops, lp, a);
                let b = self.materialize_as_simd(ops, lp, b);
                dynasm!(ops
                    ; comiss Rx(a.r() as u8), Rx(b.r() as u8)
                );
            }
            DataType::F64 => {
                let a = self.materialize_as_simd(ops, lp, a);
                let b = self.materialize_as_simd(ops, lp, b);
                dynasm!(ops
                    ; comisd Rx(a.r() as u8), Rx(b.r() as u8)
                );
            }
        }

        match (signed, cmp_type) {
            (false, CompareType::LessThan) => {
                dynasm!(ops
                    ; setb Rb(r_out as u8)
                );
            }
            (_, CompareType::Equal) => {
                dynasm!(ops
                    ; sete Rb(r_out as u8)
                );
            }
            (_, CompareType::NotEqual) => {
                dynasm!(ops
                    ; setne Rb(r_out as u8)
                );
            }
            (true, CompareType::LessThan) => {
                dynasm!(ops
                    ; setl Rb(r_out as u8)
                )
            }
            (true, CompareType::GreaterThan) => {
                dynasm!(ops
                    ; setg Rb(r_out as u8)
                );
            },
            (true, CompareType::LessThanOrEqual) => {
                dynasm!(ops
                    ; setle Rb(r_out as u8)
                );
            }
            (true, CompareType::GreaterThanOrEqual) => {
                dynasm!(ops
                    ; setge Rb(r_out as u8)
                );
            }

            (false, CompareType::GreaterThan) => todo!("Compare with type GreaterThanUnsigned"),
            (false, CompareType::LessThanOrEqual) => todo!("Compare with type LessThanOrEqualUnsigned"),
            (false, CompareType::GreaterThanOrEqual) => {
                dynasm!(ops
                    ; setae Rb(r_out as u8)
                );
            }
        }
    }

    fn load_ptr(
        &self,
        ops: &mut Ops,
        lp: &mut LiteralPool,
        r_out: Register,
        tp: DataType,
        ptr: ConstOrReg,
        offset: u64,
    ) {
        match (r_out, ptr, tp) {
            (Register::GPR(r_out), ptr, DataType::U32 | DataType::S32 | DataType::F32) => {
                let ptr = self.materialize_as_gpr(ops, lp, ptr);
                dynasm!(ops
                    ; mov Rd(r_out as u8), [Rq(ptr.r() as u8) + offset as i32]
                );
            }
            (Register::SIMD(r_out), ptr, DataType::U32 | DataType::S32 | DataType::F32) => {
                let ptr = self.materialize_as_gpr(ops, lp, ptr);
                dynasm!(ops
                    ; movd Rx(r_out as u8), [Rq(ptr.r() as u8) + offset as i32]
                );
            }
            (Register::GPR(r_out), ptr, DataType::U64 | DataType::S64 | DataType::F64) => {
                let ptr = self.materialize_as_gpr(ops, lp, ptr);
                dynasm!(ops
                    ; mov Rq(r_out as u8), [Rq(ptr.r() as u8) + offset as i32]
                );
            }
            (Register::SIMD(r_out), ptr, DataType::U64 | DataType::S64 | DataType::F64) => {
                let ptr = self.materialize_as_gpr(ops, lp, ptr);
                dynasm!(ops
                    ; movq Rx(r_out as u8), QWORD [Rq(ptr.r() as u8) + offset as i32]
                );
            }
            _ => todo!("Unsupported LoadPtr operation: Load {} from [{:?}] with type {}", r_out, ptr, tp),
        }
    }

    fn write_ptr(
        &self,
        ops: &mut Ops,
        lp: &mut LiteralPool,
        ptr: ConstOrReg,
        offset: u64,
        value: ConstOrReg,
        data_type: DataType,
    ) {
        match (ptr, value, data_type) {
            (ptr, value, DataType::U32 | DataType::S32 | DataType::F32) => {
                let address = self.materialize_as_gpr(ops, lp, ptr);
                let value = self.materialize_as_gpr(ops, lp, value);

                dynasm!(ops
                    ; mov [Rq(address.r() as u8) + offset as i32], Rd(value.r() as u8)
                );
            }
            (ptr, value, DataType::U64 | DataType::S64 | DataType::F64) => {
                let ptr = self.materialize_as_gpr(ops, lp, ptr);
                let value = self.materialize_as_gpr(ops, lp, value);
                dynasm!(ops
                    ; mov QWORD [Rq(ptr.r() as u8) + offset as i32], Rq(value.r() as u8)
                );
            }
            _ => todo!("Unsupported WritePtr operation: {:?} = {:?} with type {}", ptr, value, data_type),
        }
    }

    fn spill_to_stack(&self, ops: &mut Ops, to_spill: ConstOrReg, stack_location: ConstOrReg, tp: DataType) {
        match (&to_spill, &stack_location, tp) {
            (ConstOrReg::GPR(r), ConstOrReg::U64(location), DataType::U8 | DataType::S8) => {
                let offset = self.func.get_stack_offset_for_location(*location, tp) as i32;
                dynasm!(ops
                    ; mov [rsp + offset], Rb(*r as u8)
                )
            }
            (ConstOrReg::GPR(r), ConstOrReg::U64(location), DataType::U16 | DataType::S16) => {
                let offset = self.func.get_stack_offset_for_location(*location, tp) as i32;
                dynasm!(ops
                    ; mov [rsp + offset], Rw(*r as u8)
                )
            }
            (ConstOrReg::GPR(r), ConstOrReg::U64(location), DataType::U32 | DataType::S32) => {
                let offset = self.func.get_stack_offset_for_location(*location, tp) as i32;
                dynasm!(ops
                    ; mov [rsp + offset], Rd(*r as u8)
                )
            }
            (ConstOrReg::GPR(r), ConstOrReg::U64(location), DataType::Ptr | DataType::U64 | DataType::S64) => {
                let offset = self.func.get_stack_offset_for_location(*location, tp) as i32;
                dynasm!(ops
                    ; mov [rsp + offset], Rq(*r as u8)
                )
            }
            (ConstOrReg::SIMD(r), ConstOrReg::U64(location), DataType::F32) => {
                let offset = self.func.get_stack_offset_for_location(*location, tp) as i32;
                dynasm!(ops
                    ; movd DWORD [rsp + offset], Rx(*r as u8)
                )
            }
            (ConstOrReg::GPR(r), ConstOrReg::U64(location), DataType::Bool) => {
                let offset = self.func.get_stack_offset_for_location(*location, tp) as i32;
                dynasm!(ops
                    ; mov [rsp + offset], Rb(*r as u8)
                )
            }
            (ConstOrReg::SIMD(r), ConstOrReg::U64(location), DataType::F64) => {
                let offset = self.func.get_stack_offset_for_location(*location, tp) as i32;
                dynasm!(ops
                    ; movq QWORD [rsp + offset], Rx(*r as u8)
                )
            }
            _ => todo!(
                "Unsupported SpillToStack operation: {:?} to offset {:?} with datatype {}",
                to_spill,
                stack_location,
                tp
            ),
        }
    }

    fn load_from_stack(&self, ops: &mut Ops, r_out: Register, stack_location: ConstOrReg, tp: DataType) {
        match (r_out, &stack_location, tp) {
            (Register::GPR(r_out), ConstOrReg::U64(location), DataType::U8 | DataType::S8) => {
                let offset = self.func.get_stack_offset_for_location(*location, tp) as i32;
                dynasm!(ops
                    ; mov Rb(r_out as u8), [rsp + offset]
                    ; movzx Rd(r_out as u8), Rb(r_out as u8)
                )
            }
            (Register::GPR(r_out), ConstOrReg::U64(location), DataType::U16 | DataType::S16) => {
                let offset = self.func.get_stack_offset_for_location(*location, tp) as i32;
                dynasm!(ops
                    ; mov Rw(r_out as u8), [rsp + offset]
                    ; movzx Rd(r_out as u8), Rw(r_out as u8)
                )
            }
            (Register::GPR(r_out), ConstOrReg::U64(location), DataType::U32 | DataType::S32 | DataType::F32) => {
                let offset = self.func.get_stack_offset_for_location(*location, tp) as i32;
                dynasm!(ops
                    ; mov Rd(r_out as u8), [rsp + offset]
                )
            }
            (Register::SIMD(r_out), ConstOrReg::U64(location), DataType::U32 | DataType::S32 | DataType::F32) => {
                let offset = self.func.get_stack_offset_for_location(*location, tp) as i32;
                dynasm!(ops
                    ; movd Rx(r_out as u8), DWORD [rsp + offset]
                )
            }
            (
                Register::GPR(r_out),
                ConstOrReg::U64(location),
                DataType::Ptr | DataType::U64 | DataType::S64 | DataType::F64,
            ) => {
                let offset = self.func.get_stack_offset_for_location(*location, tp) as i32;
                dynasm!(ops
                    ; mov Rq(r_out as u8), [rsp + offset]
                )
            }
            (Register::GPR(r_out), ConstOrReg::U64(location), DataType::Bool) => {
                let offset = self.func.get_stack_offset_for_location(*location, tp) as i32;
                dynasm!(ops
                    ; mov Rb(r_out as u8), [rsp + offset]
                    ; movzx Rd(r_out as u8), Rb(r_out as u8)
                )
            }
            (Register::SIMD(r_out), ConstOrReg::U64(location), DataType::F64) => {
                let offset = self.func.get_stack_offset_for_location(*location, tp) as i32;
                dynasm!(ops
                    ; movq Rx(r_out as u8), QWORD [rsp + offset]
                )
            }
            _ => todo!(
                "Unsupported LoadFromStack operation: load {} from offset {:?} with datatype {}",
                r_out,
                stack_location,
                tp
            ),
        }
    }

    fn left_shift(
        &self,
        ops: &mut Ops,
        lp: &mut LiteralPool,
        r_out: usize,
        n: ConstOrReg,
        amount: ConstOrReg,
        tp: DataType,
    ) {
        if let Some(amount) = amount.to_u64_const() {
            let amount = amount as u32;
            match (tp, n) {
                (DataType::U8 | DataType::S8, ConstOrReg::GPR(r_n)) => {
                    dynasm!(ops
                        ; movzx Rd(r_out as u8), Rb(r_n as u8) // zero out all but the lower 8 bits
                        ; shl Rb(r_out as u8), amount as i8 & 0b111
                    );
                }
                (DataType::U16 | DataType::S16, ConstOrReg::GPR(r_n)) => {
                    dynasm!(ops
                        ; movzx Rd(r_out as u8), Rw(r_n as u8) // zero out all but the lower 16 bits
                        ; shl Rw(r_out as u8), amount as i8 & 0b1111
                    );
                }
                (DataType::U32 | DataType::S32, ConstOrReg::GPR(r_n)) => {
                    dynasm!(ops
                        ; mov Rd(r_out as u8), Rd(r_n as u8)
                        ; shl Rd(r_out as u8), amount as i8 & 0b11111
                    );
                }
                (DataType::U64 | DataType::S64, ConstOrReg::GPR(r_n)) => {
                    dynasm!(ops
                        ; mov Rq(r_out as u8), Rq(r_n as u8)
                        ; shl Rq(r_out as u8), amount as i8 & 0b111111
                    );
                }
                _ => todo!("Unsupported LeftShift operation: {:?} << {:?} with type {}", n, amount, tp),
            }
        } else if let Some(r_amount) = amount.to_reg() {
            self.move_to_reg(ops, lp, n, Register::GPR(r_out));
            let amount = self.materialize_as_gpr(ops, lp, amount);
            match tp {
                DataType::U32 | DataType::S32 => {
                    dynasm!(ops
                        ; shlx Rd(r_out as u8), Rd(r_out as u8), Rd(amount.r() as u8)
                    );
                }
                DataType::U64 | DataType::S64 => {
                    dynasm!(ops
                        ; shlx Rq(r_out as u8), Rq(r_out as u8), Rq(amount.r() as u8)
                    );
                }
                _ => todo!("LeftShift with register amount: {} {:?} >> {:?}", tp, n, r_amount),
            }
        }
    }

    fn right_shift(
        &self,
        ops: &mut Ops,
        lp: &mut LiteralPool,
        r_out: usize,
        n: ConstOrReg,
        amount: ConstOrReg,
        tp: DataType,
    ) {
        if let Some(amount) = amount.to_u64_const() {
            let amount = amount as u32;
            self.move_to_reg(ops, lp, n, Register::GPR(r_out));
            match tp {
                DataType::U8 => {
                    dynasm!(ops
                        ; movzx Rd(r_out as u8), Rb(r_out as u8)
                        ; shr Rb(r_out as u8), amount as i8 & 0b111
                    );
                }
                DataType::S8 => {
                    dynasm!(ops
                        ; movzx Rd(r_out as u8), Rb(r_out as u8)
                        ; sar Rb(r_out as u8), amount as i8 & 0b111
                    );
                }
                DataType::U16 => {
                    dynasm!(ops
                        ; movzx Rd(r_out as u8), Rw(r_out as u8)
                        ; shr Rw(r_out as u8), amount as i8 & 0b1111
                    );
                }
                DataType::S16 => {
                    dynasm!(ops
                        ; movzx Rd(r_out as u8), Rw(r_out as u8)
                        ; sar Rw(r_out as u8), amount as i8 & 0b1111
                    );
                }
                DataType::U32 => {
                    dynasm!(ops
                        ; shr Rd(r_out as u8), amount as i8 & 0b11111
                    );
                }
                DataType::S32 => {
                    dynasm!(ops
                        ; sar Rd(r_out as u8), amount as i8 & 0b11111
                    );
                }
                DataType::U64 => {
                    dynasm!(ops
                        ; shr Rq(r_out as u8), amount as i8 & 0b111111
                    );
                }
                DataType::S64 => {
                    dynasm!(ops
                        ; sar Rq(r_out as u8), amount as i8 & 0b111111
                    );
                }
                _ => todo!("Unsupported RightShift operation: {:?} >> {:?} with type {}", n, amount, tp),
            }
        } else if let Some(r_amount) = amount.to_reg() {
            self.move_to_reg(ops, lp, n, Register::GPR(r_out));
            let amount = self.materialize_as_gpr(ops, lp, amount);
            match tp {
                DataType::U32 => {
                    dynasm!(ops
                        ; shrx Rd(r_out as u8), Rd(r_out as u8), Rd(amount.r() as u8)
                    );
                }
                DataType::U64 => {
                    dynasm!(ops
                        ; shrx Rq(r_out as u8), Rq(r_out as u8), Rq(amount.r() as u8)
                    );
                }
                _ => todo!("RightShift with register amount: {} {:?} >> {:?}", tp, n, r_amount),
            }
        }
    }

    fn convert(
        &self,
        ops: &mut Ops,
        lp: &mut LiteralPool,
        r_out: Register,
        input: ConstOrReg,
        from_tp: DataType,
        to_tp: DataType,
    ) {
        match (r_out, to_tp, input, from_tp) {
            (Register::GPR(r_out), DataType::U32, input, DataType::U32) => {
                self.move_to_reg(ops, lp, input, Register::GPR(r_out));
            }
            (Register::GPR(r_out), DataType::U64, input, DataType::U64) => {
                self.move_to_reg(ops, lp, input, Register::GPR(r_out));
            }
            (Register::GPR(r_out), DataType::U64, ConstOrReg::GPR(r_in), DataType::U32) => {
                dynasm!(ops
                    // Mov r32 -> r32 zero-extends
                    ; mov Rd(r_out as u8), Rd(r_in as u8)
                );
            }
            (Register::GPR(r_out), DataType::S64, input, DataType::S8) => {
                let input = self.materialize_as_gpr(ops, lp, input);
                dynasm!(ops
                    ; movsx Rq(r_out as u8), Rb(input.r() as u8)
                )
            }
            (Register::GPR(r_out), DataType::S64, input, DataType::S16) => {
                let input = self.materialize_as_gpr(ops, lp, input);
                dynasm!(ops
                    ; movsx Rq(r_out as u8), Rw(input.r() as u8)
                )
            }
            (Register::GPR(r_out), DataType::S64, input, DataType::S32) => {
                let input = self.materialize_as_gpr(ops, lp, input);
                dynasm!(ops
                    ; movsx Rq(r_out as u8), Rd(input.r() as u8)
                );
            }
            (Register::SIMD(r_out), DataType::F32, input, DataType::S32) => {
                let input = self.materialize_as_gpr(ops, lp, input);
                dynasm!(ops
                    ; cvtsi2ss Rx(r_out as u8), Rd(input.r() as u8)
                );
            }
            (Register::GPR(r_out), DataType::S32, input, DataType::F32) => {
                let input = self.materialize_as_simd(ops, lp, input);
                dynasm!(ops
                    ; cvtss2si Rd(r_out as u8), Rx(input.r() as u8)
                );
            }
            (Register::SIMD(r_out), DataType::F64, input, DataType::S32) => {
                let input = self.materialize_as_gpr(ops, lp, input);
                dynasm!(ops
                    ; cvtsi2sd Rx(r_out as u8), Rd(input.r() as u8)
                );
            }
            (Register::SIMD(r_out), DataType::F32, input, DataType::F64) => {
                let input = self.materialize_as_simd(ops, lp, input);
                dynasm!(ops
                    ; cvtsd2ss Rx(r_out as u8), Rx(input.r() as u8)
                );
            }
            (Register::SIMD(r_out), DataType::F64, input, DataType::F32) => {
                let input = self.materialize_as_simd(ops, lp, input);
                dynasm!(ops
                    ; cvtss2sd Rx(r_out as u8), Rx(input.r() as u8)
                );
            }
            _ => todo!("Unsupported convert operation: {:?} -> {:?} types {} -> {}", input, r_out, from_tp, to_tp),
        }
    }

    fn and(&self, ops: &mut Ops, lp: &mut LiteralPool, tp: DataType, r_out: Register, a: ConstOrReg, b: ConstOrReg) {
        let a_const = a.to_u64_const();
        let b_const = b.to_u64_const();
        if a_const.is_some() && b_const.is_some() {
            let result = a_const.unwrap() & b_const.unwrap();
            dynasm!(ops
                ; mov Rq(r_out.expect_gpr() as u8), QWORD result as i64
            );
        } else {
            match (tp, r_out) {
                (DataType::U32, Register::GPR(r_out)) => {
                    let a = self.materialize_as_gpr(ops, lp, a);
                    let b = self.materialize_as_gpr(ops, lp, b);
                    dynasm!(ops
                        ; mov Rd(r_out as u8), Rd(a.r() as u8)
                        ; and Rd(r_out as u8), Rd(b.r() as u8)
                    );
                }
                (DataType::U64, Register::GPR(r_out)) => {
                    let a = self.materialize_as_gpr(ops, lp, a);
                    let b = self.materialize_as_gpr(ops, lp, b);
                    dynasm!(ops
                        ; mov Rq(r_out as u8), Rq(a.r() as u8)
                        ; and Rq(r_out as u8), Rq(b.r() as u8)
                    );
                }
                _ => todo!("Unsupported AND operation: {:?} & {:?} with type {:?}", a, b, tp),
            }
        }
    }

    fn or(&self, ops: &mut Ops, lp: &mut LiteralPool, tp: DataType, r_out: Register, a: ConstOrReg, b: ConstOrReg) {
        match (tp, r_out) {
            (DataType::U32, Register::GPR(r_out)) => {
                let a = self.materialize_as_gpr(ops, lp, a);
                let b = self.materialize_as_gpr(ops, lp, b);
                dynasm!(ops
                    ; mov Rd(r_out as u8), Rd(a.r() as u8)
                    ; or Rd(r_out as u8), Rd(b.r() as u8)
                );
            }
            (DataType::U64, Register::GPR(r_out)) => {
                let a = self.materialize_as_gpr(ops, lp, a);
                let b = self.materialize_as_gpr(ops, lp, b);
                dynasm!(ops
                    ; mov Rq(r_out as u8), Rq(a.r() as u8)
                    ; or Rq(r_out as u8), Rq(b.r() as u8)
                );
            }
            _ => todo!("Unsupported OR operation: {:?} | {:?} with type {:?}", a, b, tp),
        }
    }

    fn not(&self, ops: &mut Ops, lp: &mut LiteralPool, tp: DataType, r_out: Register, a: ConstOrReg) {
        match tp {
            DataType::U32 => {
                self.move_to_reg(ops, lp, a, r_out);
                dynasm!(ops
                    ; not Rd(r_out.expect_gpr() as u8)
                );
            }
            DataType::U64 => {
                self.move_to_reg(ops, lp, a, r_out);
                dynasm!(ops
                    ; not Rq(r_out.expect_gpr() as u8)
                );
            }
            _ => todo!("Unsupported NOT operation with type {:?}", tp),
        }
    }

    fn xor(&self, ops: &mut Ops, lp: &mut LiteralPool, tp: DataType, r_out: Register, a: ConstOrReg, b: ConstOrReg) {
        match (tp, r_out) {
            (DataType::U32 | DataType::S32, Register::GPR(r_out)) => {
                self.move_to_reg(ops, lp, a, Register::GPR(r_out));
                let b = self.materialize_as_gpr(ops, lp, b);
                dynasm!(ops
                    ; xor Rd(r_out as u8), Rd(b.r() as u8)
                );
            }
            (DataType::U64 | DataType::S64, Register::GPR(r_out)) => {
                self.move_to_reg(ops, lp, a, Register::GPR(r_out));
                let b = self.materialize_as_gpr(ops, lp, b);
                dynasm!(ops
                    ; xor Rq(r_out as u8), Rq(b.r() as u8)
                );
            }
            _ => todo!("Unsupported XOR operation: {:?} ^ {:?} with type {:?}", a, b, tp),
        }
    }

    fn subtract(
        &self,
        ops: &mut Ops,
        lp: &mut LiteralPool,
        tp: DataType,
        r_out: Register,
        minuend: ConstOrReg,
        subtrahend: ConstOrReg,
    ) {
        match (tp, r_out) {
            (DataType::U32 | DataType::S32, Register::GPR(r_out)) => {
                let minuend = self.materialize_as_gpr(ops, lp, minuend);
                let subtrahend = self.materialize_as_gpr(ops, lp, subtrahend);
                dynasm!(ops
                    ; mov Rd(r_out as u8), Rd(minuend.r() as u8)
                    ; sub Rd(r_out as u8), Rd(subtrahend.r() as u8)
                );
            }
            (DataType::U64 | DataType::S64, Register::GPR(r_out)) => {
                let minuend = self.materialize_as_gpr(ops, lp, minuend);
                let subtrahend = self.materialize_as_gpr(ops, lp, subtrahend);
                dynasm!(ops
                    ; mov Rq(r_out as u8), Rq(minuend.r() as u8)
                    ; sub Rq(r_out as u8), Rq(subtrahend.r() as u8)
                );
            }
            (DataType::F32, Register::SIMD(r_out)) => {
                let minuend = self.materialize_as_simd(ops, lp, minuend);
                let subtrahend = self.materialize_as_simd(ops, lp, subtrahend);
                dynasm!(ops
                    ; movss Rx(r_out as u8), Rx(minuend.r() as u8)
                    ; subss Rx(r_out as u8), Rx(subtrahend.r() as u8)
                );
            }
            (DataType::F64, Register::SIMD(r_out)) => {
                let minuend = self.materialize_as_simd(ops, lp, minuend);
                let subtrahend = self.materialize_as_simd(ops, lp, subtrahend);
                dynasm!(ops
                    ; movsd Rx(r_out as u8), Rx(minuend.r() as u8)
                    ; subsd Rx(r_out as u8), Rx(subtrahend.r() as u8)
                );
            }
            _ => todo!("Unsupported Subtract operation with type {:?}", tp),
        }
    }

    fn multiply(
        &self,
        ops: &mut Ops,
        lp: &mut LiteralPool,
        result_tp: DataType,
        arg_tp: DataType,
        output_regs: Vec<Option<Register>>,
        a: ConstOrReg,
        b: ConstOrReg,
    ) {
        match (result_tp, arg_tp, output_regs.len()) {
            (DataType::U32, DataType::U32, 2) => {
                let edx = self.scratch_regs.reserve::<register_type::GPR>(reg_constants::RDX);
                let eax = self.scratch_regs.reserve::<register_type::GPR>(reg_constants::RAX);

                self.move_to_reg(ops, lp, a, eax.reg());
                let b = self.materialize_as_gpr(ops, lp, b);

                let r_out_lo = output_regs[0].unwrap().expect_gpr();
                let r_out_hi = output_regs[1].unwrap().expect_gpr();

                dynasm!(ops
                    ; mul Rd(b.r() as u8)
                    ; mov Rd(r_out_lo as u8), Rd(eax.r() as u8)
                    ; mov Rd(r_out_hi as u8), Rd(edx.r() as u8)
                );
            }
            (DataType::S32, DataType::S32, 2) => {
                let edx = self.scratch_regs.reserve::<register_type::GPR>(reg_constants::RDX);
                let eax = self.scratch_regs.reserve::<register_type::GPR>(reg_constants::RAX);

                self.move_to_reg(ops, lp, a, eax.reg());
                let b = self.materialize_as_gpr(ops, lp, b);

                let r_out_lo = output_regs[0].unwrap().expect_gpr();
                let r_out_hi = output_regs[1].unwrap().expect_gpr();

                dynasm!(ops
                    ; imul Rd(b.r() as u8)
                    ; mov Rd(r_out_lo as u8), Rd(eax.r() as u8)
                    ; mov Rd(r_out_hi as u8), Rd(edx.r() as u8)
                );
            }
            (DataType::U64, DataType::U64, 2) => {
                let rdx = self.scratch_regs.reserve::<register_type::GPR>(reg_constants::RDX);
                let rax = self.scratch_regs.reserve::<register_type::GPR>(reg_constants::RAX);

                self.move_to_reg(ops, lp, a, rax.reg());
                let b = self.materialize_as_gpr(ops, lp, b);

                let r_out_lo = output_regs[0].unwrap().expect_gpr();
                let r_out_hi = output_regs[1].unwrap().expect_gpr();

                dynasm!(ops
                    ; mul Rq(b.r() as u8)
                    ; mov Rq(r_out_lo as u8), Rq(rax.r() as u8)
                    ; mov Rq(r_out_hi as u8), Rq(rdx.r() as u8)
                );
            }
            (DataType::S64, DataType::S64, 2) => {
                let rdx = self.scratch_regs.reserve::<register_type::GPR>(reg_constants::RDX);
                let rax = self.scratch_regs.reserve::<register_type::GPR>(reg_constants::RAX);

                self.move_to_reg(ops, lp, a, rax.reg());
                let b = self.materialize_as_gpr(ops, lp, b);

                let r_out_lo = output_regs[0].unwrap().expect_gpr();
                let r_out_hi = output_regs[1].unwrap().expect_gpr();

                dynasm!(ops
                    ; imul Rq(b.r() as u8)
                    ; mov Rq(r_out_lo as u8), Rq(rax.r() as u8)
                    ; mov Rq(r_out_hi as u8), Rq(rdx.r() as u8)
                );
            }
            (DataType::F32, DataType::F32, 1) => {
                self.move_to_reg(ops, lp, a, output_regs[0].unwrap());
                let b = self.materialize_as_simd(ops, lp, b);
                let r_out = output_regs[0].unwrap().expect_simd();
                dynasm!(ops
                    ; mulss Rx(r_out as u8), Rx(b.r() as u8)
                );
            }
            (DataType::F64, DataType::F64, 1) => {
                self.move_to_reg(ops, lp, a, output_regs[0].unwrap());
                let b = self.materialize_as_simd(ops, lp, b);
                let r_out = output_regs[0].unwrap().expect_simd();
                dynasm!(ops
                    ; mulsd Rx(r_out as u8), Rx(b.r() as u8)
                );
            }
            _ => todo!(
                "Unsupported Multiply operation: {:?} * {:?} with result type {} ({} regs) and arg type {}",
                a,
                b,
                result_tp,
                output_regs.len(),
                arg_tp
            ),
        }
    }

    fn divide(
        &self,
        ops: &mut Ops,
        lp: &mut LiteralPool,
        tp: DataType,
        r_quotient: Option<Register>,
        r_remainder: Option<Register>,
        dividend: ConstOrReg,
        divisor: ConstOrReg,
    ) {
        match tp {
            DataType::U32 => {
                let edx = self.scratch_regs.reserve::<register_type::GPR>(reg_constants::RDX);
                let eax = self.scratch_regs.reserve::<register_type::GPR>(reg_constants::RAX);

                self.move_to_reg(ops, lp, dividend, eax.reg());
                let divisor = self.materialize_as_gpr(ops, lp, divisor);

                let r_quotient = r_quotient.unwrap().expect_gpr();
                let r_remainder = r_remainder.unwrap().expect_gpr();

                dynasm!(ops
                    ; xor edx, edx
                    ; div Rd(divisor.r() as u8)
                    ; mov Rd(r_quotient as u8), eax
                    // Use the value here so it's obvious the `edx` value continuing to live is important
                    ; mov Rd(r_remainder as u8), Rd(edx.r() as u8)
                );
            }
            DataType::S32 => {
                let edx = self.scratch_regs.reserve::<register_type::GPR>(reg_constants::RDX);
                let eax = self.scratch_regs.reserve::<register_type::GPR>(reg_constants::RAX);

                self.move_to_reg(ops, lp, dividend, eax.reg());
                let divisor = self.materialize_as_gpr(ops, lp, divisor);

                let r_quotient = r_quotient.unwrap().expect_gpr();
                let r_remainder = r_remainder.unwrap().expect_gpr();

                dynasm!(ops
                    ; cqo
                    ; idiv Rd(divisor.r() as u8)
                    ; mov Rd(r_quotient as u8), eax
                    // Use the value here so it's obvious the `edx` value continuing to live is important
                    ; mov Rd(r_remainder as u8), Rd(edx.r() as u8)
                );
            }
            DataType::U64 => {
                let rdx = self.scratch_regs.reserve::<register_type::GPR>(reg_constants::RDX);
                let rax = self.scratch_regs.reserve::<register_type::GPR>(reg_constants::RAX);

                self.move_to_reg(ops, lp, dividend, rax.reg());
                let divisor = self.materialize_as_gpr(ops, lp, divisor);

                let r_quotient = r_quotient.unwrap().expect_gpr();
                let r_remainder = r_remainder.unwrap().expect_gpr();

                dynasm!(ops
                    ; xor edx, edx
                    ; div Rq(divisor.r() as u8)
                    ; mov Rq(r_quotient as u8), rax
                    // Use the value here so it's obvious the `rdx` value continuing to live is important
                    ; mov Rq(r_remainder as u8), Rq(rdx.r() as u8)
                );
            }
            DataType::S64 => {
                let rdx = self.scratch_regs.reserve::<register_type::GPR>(reg_constants::RDX);
                let rax = self.scratch_regs.reserve::<register_type::GPR>(reg_constants::RAX);

                self.move_to_reg(ops, lp, dividend, rax.reg());
                let divisor = self.materialize_as_gpr(ops, lp, divisor);

                let r_quotient = r_quotient.unwrap().expect_gpr();
                let r_remainder = r_remainder.unwrap().expect_gpr();

                dynasm!(ops
                    ; cqo
                    ; idiv Rq(divisor.r() as u8)
                    ; mov Rq(r_quotient as u8), rax
                    // Use the value here so it's obvious the `rdx` value continuing to live is important
                    ; mov Rq(r_remainder as u8), Rq(rdx.r() as u8)
                );
            }
            DataType::F32 => {
                let r_out = r_quotient.unwrap().expect_simd();

                if r_remainder.is_some() {
                    panic!("Remainder is not supported for F32 division");
                }

                self.move_to_reg(ops, lp, dividend, Register::SIMD(r_out));
                let divisor = self.materialize_as_simd(ops, lp, divisor);
                dynasm!(ops
                    ; divss Rx(r_out as u8), Rx(divisor.r() as u8)
                );
            }
            _ => panic!("Divide with unknown type: {}", tp),
        }
    }

    fn square_root(&self, _ops: &mut Ops, _lp: &mut LiteralPool, _tp: DataType, _r_out: Register, _value: ConstOrReg) {
        todo!()
    }

    fn absolute_value(
        &self,
        _ops: &mut Ops,
        _lp: &mut LiteralPool,
        _tp: DataType,
        _r_out: Register,
        _value: ConstOrReg,
    ) {
        todo!()
    }

    fn negate(&self, _ops: &mut Ops, _lp: &mut LiteralPool, _tp: DataType, _r_out: Register, _value: ConstOrReg) {
        todo!()
    }

    fn call_function(
        &self,
        ops: &mut Ops,
        lp: &mut LiteralPool,
        address: ConstOrReg,
        active_volatile_regs: Vec<Register>,
        r_out: Option<Register>,
        args: Vec<ConstOrReg>,
    ) {
        let active_regs = self
            .scratch_regs
            .active_regs()
            .into_iter()
            .chain(active_volatile_regs.into_iter())
            .collect::<Vec<_>>();

        let stack_bytes_needed = active_regs.iter().map(|r| r.size()).sum::<usize>();
        let misalignment = stack_bytes_needed % 16;
        let stack_bytes_needed = stack_bytes_needed + misalignment;

        dynasm!(ops
            ; sub rsp, stack_bytes_needed as i32 // Allocate stack space for the call
        );

        let mut stack_offsets = BTreeMap::new();
        let mut stack_offset = 0;
        for reg in active_regs.iter() {
            stack_offsets.insert(reg, stack_offset);
            match reg {
                Register::GPR(r) => {
                    dynasm!(ops
                        ; mov [rsp + stack_offset], Rq(*r as u8)
                    );
                }
                Register::SIMD(r) => {
                    dynasm!(ops
                        ; movdqu [rsp + stack_offset], Rx(*r as u8)
                    )
                }
            }
            stack_offset += reg.size() as i32;
        }

        // Move the arguments into place
        let moves = args
            .into_iter()
            .zip(get_function_argument_registers().into_iter())
            .collect::<BTreeMap<ConstOrReg, Register>>();

        // TODO: rework this so that the output of this is used as the input to `move_regs_multi`
        // so I don't have to mark this as unused with the underscore.
        let _reservations = moves
            .values()
            .map(|r| match r {
                Register::GPR(_) => self.scratch_regs.reserve::<register_type::GPR>(*r),
                Register::SIMD(_) => todo!("Reserving SIMD registers for arguments"),
            })
            .collect::<Vec<_>>();

        self.move_regs_multi(ops, lp, moves);

        let address = self.materialize_as_gpr(ops, lp, address);
        dynasm!(ops
            ; call Rq(address.r() as u8)
        );

        if let Some(to) = r_out {
            trace!("Moving return value from {} to {}", get_return_value_registers()[0], to);
            self.move_to_reg(ops, lp, get_return_value_registers()[0].to_const_or_reg(), to);
        }

        for reg in active_regs.iter() {
            match reg {
                Register::GPR(r) => {
                    dynasm!(ops
                        ; mov Rq(*r as u8), [rsp + stack_offsets[reg]]
                    );
                }
                Register::SIMD(r) => {
                    dynasm!(ops
                        ; movdqu Rx(*r as u8), [rsp + stack_offsets[reg]]
                    )
                }
            }
        }
        dynasm!(ops
            ; add rsp, stack_bytes_needed as i32 // Deallocate stack space for the call
        );
    }
}
