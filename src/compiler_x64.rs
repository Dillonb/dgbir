use crate::{
    abi::{get_return_value_registers, get_scratch_registers},
    compiler::{Compiler, ConstOrReg, LiteralPool},
    ir::{BlockReference, CompareType, Constant, DataType, IRFunctionInternal},
    reg_pool::{register_type, RegPool},
    register_allocator::{alloc_for, Register, RegisterAllocations},
};
use dynasmrt::{dynasm, x64::X64Relocation, DynasmApi, DynasmLabelApi};

type Ops = dynasmrt::Assembler<X64Relocation>;

pub struct X64Compiler<'a> {
    scratch_regs: RegPool,
    func: &'a IRFunctionInternal,
    allocations: RegisterAllocations,
    entrypoint: dynasmrt::AssemblyOffset,
    block_labels: Vec<dynasmrt::DynamicLabel>,
}

impl<'a> Compiler<'a, Ops> for X64Compiler<'a> {
    fn new_dynamic_label(ops: &mut Ops) -> dynasmrt::DynamicLabel {
        ops.new_dynamic_label()
    }

    fn new(ops: &mut Ops, func: &'a mut IRFunctionInternal) -> Self {
        let allocations = alloc_for(func);

        println!("Function after allocation:\n{}", func);

        // Stack bytes used: aligned to 16 bytes
        // Note: x64 is stupid, and the CALL instruction leaves the stack pointer misaligned.
        // Take this into account.
        let misalignment = (func.stack_bytes_used + 8) % 16;
        let correction = if misalignment == 0 { 0 } else { 16 - misalignment };
        let stack_bytes_used = func.stack_bytes_used + correction;
        println!(
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
                    let ofs = self.func.get_stack_offset_for_location(*stack_location as u64, DataType::U64) as i32;
                    dynasm!(ops
                        ; mov [rsp + ofs], Rq(*r as u8)
                    )
                }
                Register::SIMD(_) => todo!("Saving a preserved SIMD register"),
            }
        }
    }

    fn epilogue(&self, _ops: &mut Ops) {
        println!("Epilogue: emitting nothing");
    }

    fn jump_to_dynamic_label(&self, ops: &mut Ops, label: dynasmrt::DynamicLabel) {
        dynasm!(ops
            ; jmp =>label
        );
    }

    fn move_to_reg(&self, ops: &mut Ops, _lp: &mut LiteralPool, from: ConstOrReg, to: Register) {
        match (from, to) {
            (ConstOrReg::U32(c), Register::GPR(r_to)) => {
                dynasm!(ops
                    ; mov Rd(r_to as u8), c as i32
                );
                // It was a constant, so no need to remove the source
            }
            (ConstOrReg::U64(_), Register::GPR(_)) => todo!("Moving {:?} to {}", from, to),
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

    fn get_entrypoint(&self) -> dynasmrt::AssemblyOffset {
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
                Register::SIMD(_) => todo!("Restoring saved SIMD register from stack"),
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
        match (tp, r_out, a, b) {
            (DataType::U32, Register::GPR(r_out), ConstOrReg::U32(c1), ConstOrReg::U32(c2)) => {
                dynasm!(ops
                    ; mov Rd(r_out as u8), ((c1 + c2) as i32)
                )
            }
            (DataType::U32, Register::GPR(r_out), ConstOrReg::GPR(r), ConstOrReg::U32(c)) => {
                dynasm!(ops
                    ; mov Rd(r_out as u8), Rd(r as u8)
                    ; add Rd(r_out as u8), c as i32
                )
            }
            (DataType::U32, Register::GPR(r_out), ConstOrReg::GPR(r1), ConstOrReg::GPR(r2)) => {
                dynasm!(ops
                    ; mov Rd(r_out as u8), Rd(r1 as u8)
                    ; add Rd(r_out as u8), Rd(r2 as u8)
                )
            }
            (DataType::F32, Register::SIMD(r_out), ConstOrReg::SIMD(r), ConstOrReg::F32(c)) => {
                let literal = Self::add_literal(ops, lp, Constant::F32(c));
                dynasm!(ops
                    ; movss Rx(r_out as u8), Rx(r as u8)
                    ; addss Rx(r_out as u8), DWORD [=>literal]
                )
            }
            (DataType::F32, Register::SIMD(r_out), ConstOrReg::SIMD(r1), ConstOrReg::SIMD(r2)) => {
                dynasm!(ops
                    ; movss Rx(r_out as u8), Rx(r1 as u8)
                    ; addss Rx(r_out as u8), Rx(r2 as u8)
                )
            }
            _ => todo!("Unsupported Add operation: {:?} + {:?} with type {:?}", a, b, tp),
        }
    }

    fn compare(&self, ops: &mut Ops, r_out: usize, a: ConstOrReg, cmp_type: CompareType, b: ConstOrReg) {
        // First, zero the output register
        dynasm!(ops
            ; xor Rd(r_out as u8), Rd(r_out as u8)
        );

        match (a, b) {
            (ConstOrReg::GPR(r1), ConstOrReg::GPR(r2)) => {
                dynasm!(ops
                    ; cmp Rq(r1 as u8), Rq(r2 as u8)
                );
            }
            (ConstOrReg::GPR(r1), ConstOrReg::U32(c2)) => {
                if c2 < 0x7FFF_FFFF {
                    dynasm!(ops
                        ; cmp Rq(r1 as u8), c2 as i32
                    );
                } else {
                    todo!("Too big for i32, move to temp reg and compare")
                }
            }
            _ => todo!("Unsupported Compare operation: {:?} = {:?} {:?} {:?}", r_out, a, cmp_type, b),
        }

        match cmp_type {
            CompareType::LessThanUnsigned => {
                dynasm!(ops
                    ; setb Rb(r_out as u8)
                );
            }
            CompareType::Equal => todo!("Compare with type Equal"),
            CompareType::NotEqual => todo!("Compare with type NotEqual"),
            CompareType::LessThanSigned => todo!("Compare with type LessThanSigned"),
            CompareType::GreaterThanSigned => todo!("Compare with type GreaterThanSigned"),
            CompareType::LessThanOrEqualSigned => todo!("Compare with type LessThanOrEqualSigned"),
            CompareType::GreaterThanOrEqualSigned => todo!("Compare with type GreaterThanOrEqualSigned"),
            CompareType::GreaterThanUnsigned => todo!("Compare with type GreaterThanUnsigned"),
            CompareType::LessThanOrEqualUnsigned => todo!("Compare with type LessThanOrEqualUnsigned"),
            CompareType::GreaterThanOrEqualUnsigned => todo!("Compare with type GreaterThanOrEqualUnsigned"),
        }
    }

    fn load_ptr(
        &self,
        _ops: &mut Ops,
        _lp: &mut LiteralPool,
        _r_out: Register,
        _tp: DataType,
        _ptr: ConstOrReg,
        _offset: u64,
    ) {
        todo!("load_ptr")
    }

    fn write_ptr(
        &self,
        ops: &mut Ops,
        _lp: &mut LiteralPool,
        ptr: ConstOrReg,
        offset: u64,
        value: ConstOrReg,
        data_type: DataType,
    ) {
        match (ptr, value, data_type) {
            (ConstOrReg::U64(ptr), ConstOrReg::U32(value), DataType::U32) => {
                let r_address = self.scratch_regs.borrow::<register_type::GPR>();
                let r_value = self.scratch_regs.borrow::<register_type::GPR>();

                dynasm!(ops
                    ; mov Rq(r_address.r() as u8), QWORD (ptr + offset) as i64
                    ; mov Rd(r_value.r() as u8), value as i32
                    ; mov [Rq(r_address.r() as u8)], Rd(r_value.r() as u8)
                );
            }
            (ConstOrReg::U64(ptr), ConstOrReg::GPR(r_value), DataType::U32) => {
                let r_address = self.scratch_regs.borrow::<register_type::GPR>();

                dynasm!(ops
                    ; mov Rq(r_address.r() as u8), QWORD (ptr + offset) as i64
                    ; mov DWORD [Rq(r_address.r() as u8)], Rd(r_value as u8)
                );
            }
            (ConstOrReg::GPR(_), ConstOrReg::U32(_), DataType::U32) => todo!(),
            (ConstOrReg::GPR(r_ptr), ConstOrReg::GPR(r_value), DataType::U32) => {
                dynasm!(ops
                    ; mov DWORD [Rq(r_ptr as u8) + offset as i32], Rd(r_value as u8)
                );
            }
            _ => todo!("Unsupported WritePtr operation: {:?} = {:?} with type {}", ptr, value, data_type),
        }
    }

    fn spill_to_stack(&self, ops: &mut Ops, to_spill: ConstOrReg, stack_location: ConstOrReg, tp: DataType) {
        match (&to_spill, &stack_location, tp) {
            (ConstOrReg::GPR(r), ConstOrReg::U64(location), DataType::U32) => {
                let offset = self.func.get_stack_offset_for_location(*location, DataType::U32) as i32;
                dynasm!(ops
                    ; mov [rsp + offset], Rd(*r as u8)
                )
            }
            (ConstOrReg::GPR(r), ConstOrReg::U64(location), DataType::Ptr) => {
                let offset = self.func.get_stack_offset_for_location(*location, DataType::Ptr) as i32;
                dynasm!(ops
                    ; mov [rsp + offset], Rq(*r as u8)
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
            (Register::GPR(r_out), ConstOrReg::U64(location), DataType::U32) => {
                let offset = self.func.get_stack_offset_for_location(*location, tp) as i32;
                dynasm!(ops
                    ; mov Rd(r_out as u8), [rsp + offset]
                )
            }
            (Register::GPR(r_out), ConstOrReg::U64(location), DataType::Ptr) => {
                let offset = self.func.get_stack_offset_for_location(*location, tp) as i32;
                dynasm!(ops
                    ; mov Rq(r_out as u8), [rsp + offset]
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

    fn left_shift(&self, _ops: &mut Ops, _r_out: usize, _n: ConstOrReg, _amount: ConstOrReg, _tp: DataType) {
        todo!("Implement left shift operation")
    }

    fn right_shift(&self, _ops: &mut Ops, _r_out: usize, _n: ConstOrReg, _amount: ConstOrReg, _tp: DataType) {
        todo!("Implement right shift operation")
    }

    fn convert(&self, ops: &mut Ops, r_out: Register, input: ConstOrReg, from_tp: DataType, to_tp: DataType) {
        match (r_out, to_tp, input, from_tp) {
            (Register::GPR(r_out), DataType::U64, ConstOrReg::GPR(r_in), DataType::U32) => {
                dynasm!(ops
                    // Mov r32 -> r32 zero-extends
                    ; mov Rd(r_out as u8), Rd(r_in as u8)
                );
            }
            (Register::GPR(r_out), DataType::S64, ConstOrReg::GPR(r_in), DataType::S32) => {
                dynasm!(ops
                    ; movsx Rq(r_out as u8), Rd(r_in as u8)
                );
            }
            _ => todo!("Unsupported convert operation: {:?} -> {:?} types {} -> {}", input, r_out, from_tp, to_tp),
        }
    }

    fn and(
        &self,
        _ops: &mut Ops,
        _lp: &mut LiteralPool,
        _tp: DataType,
        _r_out: Register,
        _a: ConstOrReg,
        _b: ConstOrReg,
    ) {
        todo!()
    }

    fn or(
        &self,
        _ops: &mut Ops,
        _lp: &mut LiteralPool,
        _tp: DataType,
        _r_out: Register,
        _a: ConstOrReg,
        _b: ConstOrReg,
    ) {
        todo!()
    }

    fn not(&self, _ops: &mut Ops, _lp: &mut LiteralPool, _tp: DataType, _r_out: Register, _a: ConstOrReg) {
        todo!()
    }

    fn xor(
        &self,
        _ops: &mut Ops,
        _lp: &mut LiteralPool,
        _tp: DataType,
        _r_out: Register,
        _a: ConstOrReg,
        _b: ConstOrReg,
    ) {
        todo!()
    }

    fn subtract(
        &self,
        _ops: &mut Ops,
        _lp: &mut LiteralPool,
        _tp: DataType,
        _r_out: Register,
        _minuend: ConstOrReg,
        _subtrahend: ConstOrReg,
    ) {
        todo!()
    }

    fn multiply(
        &self,
        _ops: &mut Ops,
        _lp: &mut LiteralPool,
        _tp: DataType,
        _r_out: Register,
        _a: ConstOrReg,
        _b: ConstOrReg,
    ) {
        todo!()
    }

    fn divide(
        &self,
        _ops: &mut Ops,
        _lp: &mut LiteralPool,
        _tp: DataType,
        _r_quotient: Option<Register>,
        _r_remainder: Option<Register>,
        _dividend: ConstOrReg,
        _divisor: ConstOrReg,
    ) {
        todo!()
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
        _ops: &mut Ops,
        _lp: &mut LiteralPool,
        _address: ConstOrReg,
        _active_volatile_regs: Vec<Register>,
        _r_out: Option<Register>,
        _args: Vec<ConstOrReg>,
    ) {
        todo!()
    }
}
