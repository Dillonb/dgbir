use std::{collections::BTreeMap, iter, marker::PhantomData};

use crate::{
    abi::{get_function_argument_registers, get_return_value_registers, get_scratch_registers, reg_constants},
    compiler::{Compiler, ConstOrReg, GenericAssembler, LiteralPool},
    ir::{BlockReference, CompareType, Constant, DataType, IRFunctionInternal},
    reg_pool::{register_type, RegPool},
    register_allocator::{alloc_for, Register, RegisterAllocations, RegisterIndex},
};
use dynasmrt::{aarch64::Aarch64Relocation, dynasm, Assembler, AssemblyOffset, VecAssembler};
use log::{debug, info, trace, warn};

impl GenericAssembler<Aarch64Relocation> for Assembler<Aarch64Relocation> {
    type R = Aarch64Relocation;
    fn new_dynamic_label(&mut self) -> dynasmrt::DynamicLabel {
        self.new_dynamic_label()
    }
}

impl GenericAssembler<Aarch64Relocation> for VecAssembler<Aarch64Relocation> {
    type R = Aarch64Relocation;
    fn new_dynamic_label(&mut self) -> dynasmrt::DynamicLabel {
        self.new_dynamic_label()
    }
}

fn load_64_bit_constant<Ops: GenericAssembler<Aarch64Relocation>>(
    ops: &mut Ops,
    lp: &mut LiteralPool,
    reg: RegisterIndex,
    value: u64,
) {
    trace!("Loading 64-bit constant: 0x{:X}", value);
    if value <= 0xFFFF {
        dynasm!(ops
            ; movz X(reg), value as u32
        );
    } else {
        // TODO: check if the value can fit in a U16, U32, etc and zero extend when loading
        let literal = Aarch64Compiler::add_literal(ops, lp, Constant::U64(value));
        trace!("Loading using literal pool");
        dynasm!(ops
            ; ldr X(reg), =>literal
        );
    }
}

fn load_64_bit_signed_constant<Ops: GenericAssembler<Aarch64Relocation>>(
    ops: &mut Ops,
    lp: &mut LiteralPool,
    reg: RegisterIndex,
    value: i64,
) {
    if value >= 0 {
        load_64_bit_constant(ops, lp, reg, value as u64);
    } else if value > i32::MIN.into() {
        let literal = Aarch64Compiler::add_literal(ops, lp, Constant::S32(value as i32));
        dynasm!(ops
            ; ldrsw X(reg), =>literal
        );
    } else {
        // We need to load the full 64 bits anyway, so just use the 64-bit load
        load_64_bit_constant(ops, lp, reg, value as u64);
    }
}

fn load_32_bit_constant<Ops: GenericAssembler<Aarch64Relocation>>(
    ops: &mut Ops,
    lp: &mut LiteralPool,
    reg: RegisterIndex,
    value: u32,
) {
    if value <= 0xFFFF {
        dynasm!(ops
            ; movz W(reg), value
        );
    } else {
        let literal = Aarch64Compiler::add_literal(ops, lp, Constant::U32(value));
        dynasm!(ops
            ; ldr W(reg), =>literal
        );
    }
}

pub struct Aarch64Compiler<'a, Ops> {
    scratch_regs: RegPool,
    func: &'a IRFunctionInternal,
    allocations: RegisterAllocations,
    entrypoint: dynasmrt::AssemblyOffset,
    block_labels: Vec<dynasmrt::DynamicLabel>,
    phantom: PhantomData<Ops>,
}

impl<'a, Ops: GenericAssembler<Aarch64Relocation>> Compiler<'a, Aarch64Relocation, Ops> for Aarch64Compiler<'a, Ops> {
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

        // Stack bytes used: aligned to 16 bytes
        let misalignment = func.stack_bytes_used % 16;
        let correction = if misalignment == 0 { 0 } else { 16 - misalignment };
        let stack_bytes_used = func.stack_bytes_used + correction;
        info!(
            "Function uses {} bytes of stack, misaligned by {}, corrected to {}",
            func.stack_bytes_used, misalignment, stack_bytes_used
        );
        func.stack_bytes_used = stack_bytes_used;

        let entrypoint = ops.offset();

        let block_labels = func.blocks.iter().map(|_| ops.new_dynamic_label()).collect::<Vec<_>>();

        Aarch64Compiler {
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
                ; sub sp, sp, self.func.stack_bytes_used.try_into().unwrap()
            )
        }

        // Save callee-saved registers to stack
        for (reg, stack_location) in &self.allocations.callee_saved {
            match reg {
                Register::GPR(r) => {
                    assert_eq!(reg.size(), 8);
                    dynasm!(ops
                        ; str X(*r), [sp, self.func.get_stack_offset_for_location(*stack_location as u64, DataType::U64)]
                    )
                }
                Register::SIMD(r) => {
                    assert_eq!(reg.size(), 16);
                    dynasm!(ops
                        ; str Q(*r), [sp, self.func.get_stack_offset_for_location(*stack_location as u64, DataType::U128)]
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
            ; b =>label
        )
    }

    fn move_to_reg(&self, ops: &mut Ops, lp: &mut LiteralPool, from: ConstOrReg, to: Register) {
        match (from, to) {
            (ConstOrReg::U16(c), Register::GPR(r_to)) => {
                load_32_bit_constant(ops, lp, r_to, c as u32);
            }
            (ConstOrReg::S16(c), Register::GPR(r_to)) => {
                load_64_bit_signed_constant(ops, lp, r_to, c.into());
            }
            (ConstOrReg::U32(c), Register::GPR(r_to)) => {
                load_32_bit_constant(ops, lp, r_to, c);
            }
            (ConstOrReg::S32(c), Register::GPR(r_to)) => {
                load_64_bit_signed_constant(ops, lp, r_to, c.into());
            }
            (ConstOrReg::U64(c), Register::GPR(r)) => {
                load_64_bit_constant(ops, lp, r, c);
            }
            (ConstOrReg::S64(c), Register::GPR(r)) => {
                load_64_bit_signed_constant(ops, lp, r, c);
            }
            (ConstOrReg::GPR(r_from), Register::GPR(r_to)) => {
                dynasm!(ops
                    ; mov X(r_to), X(r_from)
                );
            }
            (ConstOrReg::GPR(r_from), Register::SIMD(r_to)) => {
                dynasm!(ops
                    ; fmov D(r_to), X(r_from)
                );
            }
            (ConstOrReg::SIMD(r_from), Register::GPR(r_to)) => {
                dynasm!(ops
                    ; fmov X(r_to), D(r_from)
                );
            }
            (ConstOrReg::SIMD(r_from), Register::SIMD(r_to)) => {
                dynasm!(ops
                    ; mov V(r_to).B16, V(r_from).B16
                )
            }
            (ConstOrReg::F32(value), Register::SIMD(r_to)) => {
                dynasm!(ops
                    ; fmov S(r_to), *value
                )
            }
            _ => todo!("Unimplemented move operation: {:?} to {:?}", from, to),
        }
    }

    // TODO: this is exactly the same in all compilers, figure out how to share this
    fn emit_literal_pool(&self, ops: &mut Ops, lp: LiteralPool) {
        for (literal, label) in lp.literals {
            trace!("Aligning to {} bytes for literal {:?}", literal.size(), literal);
            ops.align(literal.size(), 0);
            match literal {
                Constant::U32(c) => {
                    dynasm!(ops
                        ; =>label
                        ; .u32 c
                    );
                }
                Constant::S32(c) => {
                    dynasm!(ops
                        ; =>label
                        ; .i32 c
                    );
                }
                Constant::U64(c) => {
                    dynasm!(ops
                        ; =>label
                        ; .u64 c
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
        self.func
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
                    ; cbz W(*c), >if_false
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
            self.move_to_reg(
                ops,
                lp,
                *v,
                *get_return_value_registers()
                    .iter()
                    .find(|r| v.is_same_type_as(*r))
                    .unwrap(),
            );
        }

        // Pop callee-saved regs from stack
        // TODO: move this to the epilogue and emit a jmp to the end of the function here (to make
        // multiple returns more efficient)
        for (reg, stack_location) in &self.allocations.callee_saved {
            match *reg {
                Register::GPR(r) => {
                    assert_eq!(reg.size(), 8);
                    dynasm!(ops
                        ; ldr X(r), [sp, self.func.get_stack_offset_for_location(*stack_location as u64, DataType::U64)]
                    )
                }
                Register::SIMD(r) => {
                    assert_eq!(reg.size(), 16);
                    dynasm!(ops
                        ; ldr Q(r), [sp, self.func.get_stack_offset_for_location(*stack_location as u64, DataType::U128)]
                    )
                }
            }
        }

        // Fix sp
        if self.func.stack_bytes_used > 0 {
            dynasm!(ops
                ; add sp, sp, self.func.stack_bytes_used.try_into().unwrap()
            );
        }
        dynasm!(ops
            ; ret
        );
    }

    fn add(&self, ops: &mut Ops, lp: &mut LiteralPool, tp: DataType, r_out: Register, a: ConstOrReg, b: ConstOrReg) {
        match (tp, r_out) {
            (DataType::U32 | DataType::S32, Register::GPR(r_out)) => {
                let a = self.materialize_as_gpr(ops, lp, a);
                let b = self.materialize_as_gpr(ops, lp, b);
                dynasm!(ops
                    ; add W(r_out), W(a.r()), W(b.r())
                );
            }
            (DataType::U64 | DataType::S64, Register::GPR(r_out)) => {
                let a = self.materialize_as_gpr(ops, lp, a);
                let b = self.materialize_as_gpr(ops, lp, b);
                dynasm!(ops
                    ; add X(r_out), X(a.r()), X(b.r())
                );
            }
            (DataType::F32, Register::SIMD(r_out)) => {
                let a = self.materialize_as_simd(ops, lp, a);
                let b = self.materialize_as_simd(ops, lp, b);
                dynasm!(ops
                    ; fadd S(r_out), S(a.r()), S(b.r())
                );
            }
            (DataType::F64, Register::SIMD(r_out)) => {
                let a = self.materialize_as_simd(ops, lp, a);
                let b = self.materialize_as_simd(ops, lp, b);
                dynasm!(ops
                    ; fadd D(r_out), D(a.r()), D(b.r())
                );
            }
            _ => todo!("Unsupported Add operation: ({:?}, {:?})", tp, r_out),
        }
    }

    fn compare(
        &self,
        ops: &mut Ops,
        lp: &mut LiteralPool,
        r_out: RegisterIndex,
        data_type: DataType,
        a: ConstOrReg,
        cmp_type: CompareType,
        b: ConstOrReg,
    ) {
        fn set_reg_by_flags<Ops: GenericAssembler<Aarch64Relocation>>(
            ops: &mut Ops,
            signed: bool,
            cmp_type: CompareType,
            r_out: RegisterIndex,
        ) {
            // https://developer.arm.com/documentation/100076/0100/A64-Instruction-Set-Reference/A64-General-Instructions/CSET
            // https://developer.arm.com/documentation/100076/0100/A64-Instruction-Set-Reference/Condition-Codes/Condition-code-suffixes-and-related-flags?lang=en
            match (signed, cmp_type) {
                (false, CompareType::LessThan) => {
                    dynasm!(ops
                        ; cset W(r_out), lo // unsigned "lower"
                    )
                }
                (_, CompareType::Equal) => {
                    dynasm!(ops
                        ; cset W(r_out), eq // "equal"
                    )
                }
                (_, CompareType::NotEqual) => {
                    dynasm!(ops
                        ; cset W(r_out), ne // "not equal"
                    )
                }
                (true, CompareType::LessThan) => {
                    dynasm!(ops
                        ; cset W(r_out), lt // signed "less than"
                    )
                }
                (true, CompareType::GreaterThan) => {
                    dynasm!(ops
                        ; cset W(r_out), gt // signed "greater than"
                    )
                }
                (true, CompareType::LessThanOrEqual) => {
                    dynasm!(ops
                        ; cset W(r_out), le // signed "less than or equal"
                    )
                }
                (true, CompareType::GreaterThanOrEqual) => {
                    dynasm!(ops
                        ; cset W(r_out), ge // signed "greater than or equal"
                    )
                }
                (false, CompareType::GreaterThan) => todo!("Compare with type GreaterThanUnsigned"),
                (false, CompareType::LessThanOrEqual) => todo!("Compare with type LessThanOrEqualUnsigned"),
                (false, CompareType::GreaterThanOrEqual) => todo!("Compare with type GreaterThanOrEqualUnsigned"),
            }
        }

        let signed = data_type.is_signed();

        if data_type.is_integer() {
            match (a, b) {
                (ConstOrReg::GPR(r1), ConstOrReg::GPR(r2)) => {
                    dynasm!(ops
                        ; cmp X(r1), X(r2)
                    );
                    set_reg_by_flags(ops, signed, cmp_type, r_out);
                }
                (ConstOrReg::GPR(r), c) if c.is_const() => {
                    let c = c.to_u64_const().unwrap();
                    if c < 4096 {
                        dynasm!(ops
                            ; cmp XSP(r), c as u32
                        );
                    } else {
                        let r_temp = self.scratch_regs.borrow::<register_type::GPR>();
                        load_64_bit_constant(ops, lp, r_temp.r(), c);
                        dynasm!(ops
                            ; cmp XSP(r), X(r_temp.r())
                        );
                    }
                    set_reg_by_flags(ops, signed, cmp_type, r_out);
                }
                (c, ConstOrReg::GPR(r)) if c.is_const() => {
                    let c = c.to_u64_const().unwrap();
                    let r_temp = self.scratch_regs.borrow::<register_type::GPR>();
                    load_64_bit_constant(ops, lp, r_temp.r(), c);
                    dynasm!(ops
                        ; cmp XSP(r_temp.r()), X(r)
                    );
                    set_reg_by_flags(ops, signed, cmp_type, r_out);
                }
                (c1, c2) if c1.is_const() && c2.is_const() => match (signed, cmp_type) {
                    (_, CompareType::Equal) => {
                        dynasm!(ops
                            ; mov W(r_out), (c1.to_u64_const().unwrap() == c2.to_u64_const().unwrap()) as u32
                        )
                    }
                    (_, CompareType::NotEqual) => {
                        dynasm!(ops
                            ; mov W(r_out), (c1.to_u64_const().unwrap() != c2.to_u64_const().unwrap()) as u32
                        )
                    }
                    (true, CompareType::LessThan) => todo!("Compare constants with type LessThanSigned"),
                    (true, CompareType::GreaterThan) => todo!("Compare constants with type GreaterThanSigned"),
                    (true, CompareType::LessThanOrEqual) => todo!("Compare constants with type LessThanOrEqualSigned"),
                    (true, CompareType::GreaterThanOrEqual) => {
                        dynasm!(ops
                            ; mov W(r_out), (c1.to_s64_const().unwrap() >= c2.to_s64_const().unwrap()) as u32
                        )
                    }
                    (false, CompareType::LessThan) => todo!("Compare constants with type LessThanUnsigned"),
                    (false, CompareType::GreaterThan) => todo!("Compare constants with type GreaterThanUnsigned"),
                    (false, CompareType::LessThanOrEqual) => {
                        todo!("Compare constants with type LessThanOrEqualUnsigned")
                    }
                    (false, CompareType::GreaterThanOrEqual) => {
                        todo!("Compare constants with type GreaterThanOrEqualUnsigned")
                    }
                },
                _ => todo!(
                    "Unsupported integer Compare operation: {:?} = {:?} {:?} {:?} with data type {:?}",
                    r_out,
                    a,
                    cmp_type,
                    b,
                    data_type
                ),
            }
        } else if data_type.is_float() {
            let signed = true; // Floats are always signed
            let a = self.materialize_as_simd(ops, lp, a);
            let b = self.materialize_as_simd(ops, lp, b);
            match data_type {
                DataType::F32 => {
                    dynasm!(ops
                        ; fcmp S(a.r()), S(b.r())
                    );
                }
                DataType::F64 => {
                    dynasm!(ops
                        ; fcmp D(a.r()), D(b.r())
                    );
                }
                _ => todo!("Unsupported float Compare operation with data type {:?}", data_type),
            }
            set_reg_by_flags(ops, signed, cmp_type, r_out);
        } else {
            todo!("Unsupported Compare operation with data type: {:?}", data_type);
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
            (Register::GPR(r_out), ConstOrReg::U64(ptr), DataType::U32 | DataType::S32) => {
                let r_ptr = self.scratch_regs.borrow::<register_type::GPR>();
                load_64_bit_constant(ops, lp, r_ptr.r(), ptr);
                dynasm!(ops
                    ; ldr W(r_out), [X(r_ptr.r()), offset as u32]
                );
            }
            (Register::GPR(r_out), ConstOrReg::GPR(r_ptr), DataType::U64) => {
                dynasm!(ops
                    ; ldr X(r_out), [X(r_ptr), offset as u32]
                );
            }
            (Register::GPR(r_out), ConstOrReg::GPR(r_ptr), DataType::U32 | DataType::S32) => {
                dynasm!(ops
                    ; ldr W(r_out), [X(r_ptr), offset as u32]
                );
            }
            (Register::SIMD(r_out), ConstOrReg::GPR(r_ptr), DataType::F64) => {
                dynasm!(ops
                    ; ldr D(r_out), [X(r_ptr), offset as u32]
                );
            }
            (Register::SIMD(r_out), ConstOrReg::GPR(r_ptr), DataType::F32) => {
                dynasm!(ops
                    ; ldr S(r_out), [X(r_ptr), offset as u32]
                );
            }
            _ => todo!("Unsupported LoadPtr operation: Load {:?} with address [{:?}] and type {}", r_out, ptr, tp),
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
            (ConstOrReg::U64(ptr), ConstOrReg::U32(value), DataType::U32) => {
                let r_address = self.scratch_regs.borrow::<register_type::GPR>();
                let r_value = self.scratch_regs.borrow::<register_type::GPR>();

                load_64_bit_constant(ops, lp, r_address.r(), ptr);
                load_32_bit_constant(ops, lp, r_value.r(), value);
                dynasm!(ops
                    ; str W(r_value.r()), [X(r_address.r()), offset as u32]
                );
            }
            (ConstOrReg::U64(ptr), ConstOrReg::GPR(r_value), DataType::U32) => {
                let r_ptr = self.scratch_regs.borrow::<register_type::GPR>();
                load_64_bit_constant(ops, lp, r_ptr.r(), ptr);
                dynasm!(ops
                    ; str W(r_value), [X(r_ptr.r()), offset as u32]
                );
            }
            (ConstOrReg::GPR(r_ptr), ConstOrReg::U32(c_value), DataType::U32) => {
                let r_value = self.scratch_regs.borrow::<register_type::GPR>();
                load_64_bit_constant(ops, lp, r_value.r(), c_value.into());
                dynasm!(ops
                    ; str W(r_value.r()), [X(r_ptr), offset as u32]
                );
            }
            (ConstOrReg::GPR(r_ptr), ConstOrReg::GPR(r_value), DataType::U32 | DataType::S32 | DataType::F32) => {
                dynasm!(ops
                    ; str W(r_value), [X(r_ptr), offset as u32]
                );
            }
            (ConstOrReg::GPR(r_ptr), ConstOrReg::SIMD(r_value), DataType::U32 | DataType::S32 | DataType::F32) => {
                dynasm!(ops
                    ; str S(r_value), [X(r_ptr), offset as u32]
                );
            }
            (ConstOrReg::GPR(r_ptr), ConstOrReg::GPR(r_value), DataType::U64 | DataType::S64 | DataType::F64) => {
                dynasm!(ops
                    ; str X(r_value), [X(r_ptr), offset as u32]
                );
            }
            (ConstOrReg::GPR(r_ptr), ConstOrReg::U64(value), DataType::U64) => {
                let r_value = self.scratch_regs.borrow::<register_type::GPR>();
                load_64_bit_constant(ops, lp, r_value.r(), value);
                dynasm!(ops
                    ; str X(r_value.r()), [X(r_ptr), offset as u32]
                );
            }
            (ConstOrReg::GPR(r_ptr), c, DataType::U64 | DataType::S64) if c.is_const() => {
                let r_value = self.scratch_regs.borrow::<register_type::GPR>();
                load_64_bit_constant(ops, lp, r_value.r(), c.to_u64_const().unwrap());
                dynasm!(ops
                    ; str X(r_value.r()), [X(r_ptr), offset as u32]
                );
            }
            (ConstOrReg::GPR(r_ptr), ConstOrReg::SIMD(r_value), DataType::U64 | DataType::S64 | DataType::F64) => {
                dynasm!(ops
                    ; str D(r_value), [X(r_ptr), offset as u32]
                );
            }
            _ => todo!("Unsupported WritePtr operation: {:?} = {:?} with type {}", ptr, value, data_type),
        }
    }

    fn spill_to_stack(&self, ops: &mut Ops, to_spill: ConstOrReg, stack_offset: ConstOrReg, tp: DataType) {
        match (&to_spill, &stack_offset, tp) {
            (ConstOrReg::GPR(r), ConstOrReg::U64(offset), DataType::U8 | DataType::S8 | DataType::Bool) => {
                dynasm!(ops
                    ; strb W(*r), [sp, self.func.get_stack_offset_for_location(*offset, DataType::U8)]
                )
            }
            (ConstOrReg::GPR(r), ConstOrReg::U64(offset), DataType::U16 | DataType::S16) => {
                dynasm!(ops
                    ; strh W(*r), [sp, self.func.get_stack_offset_for_location(*offset, DataType::U16)]
                )
            }
            (ConstOrReg::GPR(r), ConstOrReg::U64(offset), DataType::U32 | DataType::S32) => {
                dynasm!(ops
                    ; str W(*r), [sp, self.func.get_stack_offset_for_location(*offset, DataType::U32)]
                )
            }
            (ConstOrReg::GPR(r), ConstOrReg::U64(offset), DataType::U64 | DataType::S64 | DataType::Ptr) => {
                dynasm!(ops
                    ; str X(*r), [sp, self.func.get_stack_offset_for_location(*offset, DataType::U64)]
                )
            }
            (ConstOrReg::SIMD(r), ConstOrReg::U64(offset), DataType::F32) => {
                dynasm!(ops
                    ; str S(*r), [sp, self.func.get_stack_offset_for_location(*offset, DataType::F32)]
                )
            }
            (ConstOrReg::SIMD(r), ConstOrReg::U64(offset), DataType::F64) => {
                dynasm!(ops
                    ; str D(*r), [sp, self.func.get_stack_offset_for_location(*offset, DataType::F64)]
                )
            }
            _ => todo!(
                "Unsupported SpillToStack operation: {:?} to offset {:?} with datatype {}",
                to_spill,
                stack_offset,
                tp
            ),
        }
    }

    fn load_from_stack(&self, ops: &mut Ops, r_out: Register, stack_offset: ConstOrReg, tp: DataType) {
        match (r_out, &stack_offset, tp) {
            (Register::GPR(r_out), ConstOrReg::U64(offset), DataType::U8 | DataType::S8 | DataType::Bool) => {
                dynasm!(ops
                    ; ldrb W(r_out), [sp, self.func.get_stack_offset_for_location(*offset, DataType::U8)]
                )
            }
            (Register::GPR(r_out), ConstOrReg::U64(offset), DataType::U16 | DataType::S16) => {
                dynasm!(ops
                    ; ldrh W(r_out), [sp, self.func.get_stack_offset_for_location(*offset, DataType::U16)]
                )
            }
            (Register::GPR(r_out), ConstOrReg::U64(offset), DataType::U32 | DataType::S32) => {
                dynasm!(ops
                    ; ldr W(r_out), [sp, self.func.get_stack_offset_for_location(*offset, DataType::U32)]
                )
            }
            (Register::GPR(r_out), ConstOrReg::U64(offset), DataType::U64 | DataType::S64 | DataType::Ptr) => {
                dynasm!(ops
                    ; ldr X(r_out), [sp, self.func.get_stack_offset_for_location(*offset, DataType::U64)]
                )
            }
            (Register::SIMD(r_out), ConstOrReg::U64(offset), DataType::F32) => {
                dynasm!(ops
                    ; ldr S(r_out), [sp, self.func.get_stack_offset_for_location(*offset, DataType::F32)]
                )
            }
            (Register::SIMD(r_out), ConstOrReg::U64(offset), DataType::F64) => {
                dynasm!(ops
                    ; ldr D(r_out), [sp, self.func.get_stack_offset_for_location(*offset, DataType::F64)]
                )
            }
            _ => todo!(
                "Unsupported LoadFromStack operation: load {} from offset {:?} with datatype {}",
                r_out,
                stack_offset,
                tp
            ),
        }
    }

    fn left_shift(
        &self,
        ops: &mut Ops,
        lp: &mut LiteralPool,
        r_out: RegisterIndex,
        n: ConstOrReg,
        amount: ConstOrReg,
        tp: DataType,
    ) -> () {
        if let Some(amount) = amount.to_u64_const() {
            let amount = amount as u32;
            if let Some(base) = n.to_u64_const() {
                match tp {
                    DataType::U64 | DataType::S64 => {
                        let result = base.wrapping_shl(amount);
                        load_64_bit_constant(ops, lp, r_out, result);
                    }
                    DataType::U32 | DataType::S32 => {
                        let result = (base as i32).wrapping_shl(amount) as i64;
                        load_64_bit_constant(ops, lp, r_out, result as u64);
                    }
                    _ => todo!("LeftShift with constant base with tp {}", tp),
                }
            } else {
                match (tp, n) {
                    (DataType::U8 | DataType::S8, ConstOrReg::GPR(r_n)) => {
                        dynasm!(ops
                            ; lsl W(r_out), W(r_n), amount & 0b111
                            ; and WSP(r_out), W(r_out), 0xFF // Mask to 8 bits
                        );
                    }
                    (DataType::U16 | DataType::S16, ConstOrReg::GPR(r_n)) => {
                        dynasm!(ops
                            ; lsl W(r_out), W(r_n), amount & 0b1111
                            ; and WSP(r_out), W(r_out), 0xFFFF // Mask to 16
                        );
                    }
                    (DataType::U32 | DataType::S32, ConstOrReg::GPR(r_n)) => {
                        dynasm!(ops
                            ; lsl W(r_out), W(r_n), amount & 0b11111
                        );
                    }
                    (DataType::U64 | DataType::S64, ConstOrReg::GPR(r_n)) => {
                        dynasm!(ops
                            ; lsl X(r_out), X(r_n), amount & 0b111111
                        );
                    }
                    _ => todo!("Unsupported LeftShift operation: {:?} << {:?} with type {}", n, amount, tp),
                }
            }
        } else if let Some(Register::GPR(r_amount)) = amount.to_reg() {
            match (tp, n) {
                (DataType::U8, ConstOrReg::GPR(_r_n)) => todo!("LeftShift with GPR amount for U8"),
                (DataType::S8, ConstOrReg::GPR(_r_n)) => todo!("LeftShift with GPR amount for S8"),
                (DataType::U16, ConstOrReg::GPR(_r_n)) => todo!("LeftShift with GPR amount for U16"),
                (DataType::S16, ConstOrReg::GPR(_r_n)) => todo!("LeftShift with GPR amount for S16"),
                (DataType::U32, ConstOrReg::GPR(r_n)) => {
                    dynasm!(ops
                        ; lslv W(r_out), W(r_n), W(r_amount)
                    );
                }
                (DataType::U32, c) if c.is_const() => {
                    let c = c.to_u64_const().unwrap() as u32;
                    let r_temp = self.scratch_regs.borrow::<register_type::GPR>();
                    load_32_bit_constant(ops, lp, r_temp.r(), c);
                    dynasm!(ops
                        ; lslv W(r_out), W(r_temp.r()), W(r_amount)
                    );
                }
                (DataType::S32, ConstOrReg::GPR(_r_n)) => todo!("LeftShift with GPR amount for S32"),
                (DataType::U64, ConstOrReg::GPR(r_n)) => {
                    dynasm!(ops
                        ; lslv X(r_out), X(r_n), X(r_amount)
                    );
                }
                (DataType::U64, c) if c.is_const() => {
                    let c = c.to_u64_const().unwrap();
                    let r_temp = self.scratch_regs.borrow::<register_type::GPR>();
                    load_64_bit_constant(ops, lp, r_temp.r(), c);
                    dynasm!(ops
                        ; lslv X(r_out), X(r_temp.r()), X(r_amount)
                    );
                }
                (DataType::S64, ConstOrReg::GPR(_r_n)) => todo!("LeftShift with GPR amount for S64"),

                _ => todo!("Unsupported DataType {} or unsupported register type for LeftShift operation with GPR amount: {:?} << GPR({:?})", tp, n, r_amount),
            }
        } else {
            panic!("RightShift amount must be a constant or a GPR, got: {:?}", amount);
        }
    }

    fn right_shift(
        &self,
        ops: &mut Ops,
        lp: &mut LiteralPool,
        r_out: RegisterIndex,
        n: ConstOrReg,
        amount: ConstOrReg,
        tp: DataType,
    ) {
        if let Some(amount) = amount.to_u64_const() {
            let amount = amount as u32;
            match (tp, n) {
                (DataType::U8, ConstOrReg::GPR(r_n)) => {
                    dynasm!(ops
                        ; and WSP(r_out), W(r_n), 0xFF
                        ; lsr W(r_out), W(r_out), amount & 0b111
                    );
                }
                (DataType::S8, ConstOrReg::GPR(r_n)) => {
                    // Shift left to put the sign bit in the 32 bit sign bit position, then shift
                    // right.
                    dynasm!(ops
                        ; lsl W(r_out), W(r_n), 24
                        ; asr W(r_out), W(r_out), (amount & 0b111) + 24
                        ; and WSP(r_out), W(r_out), 0xFF // Mask to 8 bits
                    );
                }
                (DataType::U16, ConstOrReg::GPR(r_n)) => {
                    dynasm!(ops
                        ; and WSP(r_out), W(r_n), 0xFFFF
                        ; lsr W(r_out), W(r_out), amount & 0b1111
                    );
                }
                (DataType::S16, ConstOrReg::GPR(r_n)) => {
                    // Shift left to put the sign bit in the 32 bit sign bit position, then shift
                    // right.
                    dynasm!(ops
                        ; lsl W(r_out), W(r_n), 16
                        ; asr W(r_out), W(r_out), (amount & 0b1111) + 16
                        ; and WSP(r_out), W(r_out), 0xFFFF // Mask to 16 bits
                    );
                }
                (DataType::U32, ConstOrReg::GPR(r_n)) => {
                    dynasm!(ops
                        ; lsr W(r_out), W(r_n), amount & 0b11111
                    );
                }
                (DataType::U32, c) if c.is_const() => {
                    let c = c.to_u64_const().unwrap() as u32;
                    let c = c >> (amount & 0b11111);
                    load_32_bit_constant(ops, lp, r_out, c);
                }
                (DataType::S32, ConstOrReg::GPR(r_n)) => {
                    dynasm!(ops
                        ; asr W(r_out), W(r_n), amount & 0b11111
                    );
                }
                (DataType::U64, ConstOrReg::GPR(r_n)) => {
                    dynasm!(ops
                        ; lsr X(r_out), X(r_n), amount & 0b111111
                    );
                }
                (DataType::S64, ConstOrReg::GPR(r_n)) => {
                    dynasm!(ops
                        ; asr X(r_out), X(r_n), amount & 0b111111
                    );
                }
                _ => todo!("Unsupported RightShift operation: {:?} >> {:?} with type {}", n, amount, tp),
            }
        } else if let Some(Register::GPR(r_amount)) = amount.to_reg() {
            match (tp, n) {
                (DataType::U8, ConstOrReg::GPR(_r_n)) => todo!("RightShift with GPR amount for U8"),
                (DataType::S8, ConstOrReg::GPR(_r_n)) => todo!("RightShift with GPR amount for S8"),
                (DataType::U16, ConstOrReg::GPR(_r_n)) => todo!("RightShift with GPR amount for U16"),
                (DataType::S16, ConstOrReg::GPR(_r_n)) => todo!("RightShift with GPR amount for S16"),
                (DataType::U32, ConstOrReg::GPR(r_n)) => {
                    dynasm!(ops
                        ; lsrv W(r_out), W(r_n), W(r_amount)
                    );
                }
                (DataType::S32, ConstOrReg::GPR(_r_n)) => todo!("RightShift with GPR amount for S32"),
                (DataType::U64, ConstOrReg::GPR(r_n)) => {
                    dynasm!(ops
                        ; lsrv X(r_out), X(r_n), X(r_amount)
                    );
                }
                (DataType::S64, ConstOrReg::GPR(_r_n)) => todo!("RightShift with GPR amount for S64"),

                (DataType::U8,  c) if c.is_const() => todo!("RightShift const with GPR amount for U8"),
                (DataType::S8,  c) if c.is_const() => todo!("RightShift const with GPR amount for S8"),
                (DataType::U16, c) if c.is_const() => todo!("RightShift const with GPR amount for U16"),
                (DataType::S16, c) if c.is_const() => todo!("RightShift const with GPR amount for S16"),
                (DataType::U32, c) if c.is_const() => {
                    let c = c.to_u64_const().unwrap() as u32;
                    load_32_bit_constant(ops, lp, r_out, c);
                    dynasm!(ops
                        ; lsrv W(r_out), W(r_out), W(r_amount)
                    );
                },
                (DataType::S32, c) if c.is_const() => todo!("RightShift const with GPR amount for S32"),
                (DataType::U64, c) if c.is_const() => {
                    let c = c.to_u64_const().unwrap();
                    load_64_bit_constant(ops, lp, r_out, c);
                    dynasm!(ops
                        ; lsrv X(r_out), X(r_out), X(r_amount)
                    );
                },
                (DataType::S64, c) if c.is_const() => todo!("RightShift const with GPR amount for S64"),

                _ => todo!("Unsupported DataType {} or unsupported register type for RightShift operation with GPR amount: {:?} >> {:?}", tp, n, amount),
            }
        } else {
            panic!("RightShift amount must be a constant or a GPR, got: {:?}", amount);
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
        match (r_out, to_tp, from_tp) {
            (Register::GPR(r_out), DataType::U64, DataType::U32) => {
                let input = self.materialize_as_gpr(ops, lp, input);
                dynasm!(ops
                    // mov to a 32 bit register zero extends it by default
                    ; mov W(r_out), W(input.r())
                );
            }
            (Register::GPR(r_out), DataType::S64, DataType::S32) => {
                let input = self.materialize_as_gpr(ops, lp, input);
                dynasm!(ops
                    ; sxtw X(r_out), W(input.r())
                );
            }
            (Register::GPR(r_out), DataType::S64, DataType::S8) => {
                let input = self.materialize_as_gpr(ops, lp, input);
                dynasm!(ops
                    // Shift the sign bit into the 32 bit sign position
                    ; lsl W(r_out), W(input.r()), 24
                    // Sign extend to 64 bits
                    ; sxtw X(r_out), W(r_out)
                    // Then shift arithmetic back to the original position
                    ; asr X(r_out), X(r_out), 24
                );
            }
            (Register::GPR(r_out), DataType::S64, DataType::S16) => {
                let input = self.materialize_as_gpr(ops, lp, input);
                dynasm!(ops
                    // Shift the sign bit into the 32 bit sign position
                    ; lsl W(r_out), W(input.r()), 16
                    // Sign extend to 64 bits
                    ; sxtw X(r_out), W(r_out)
                    // Then shift arithmetic back to the original position
                    ; asr X(r_out), X(r_out), 16
                );
            }
            (Register::SIMD(r_out), DataType::F64, DataType::S32) => {
                let input = self.materialize_as_gpr(ops, lp, input);
                dynasm!(ops
                    ; scvtf D(r_out), W(input.r())
                );
            }
            (Register::SIMD(r_out), DataType::F32, DataType::S32) => {
                let input = self.materialize_as_gpr(ops, lp, input);
                dynasm!(ops
                    ; scvtf S(r_out), W(input.r())
                );
            }
            (Register::SIMD(r_out), DataType::F32, DataType::F64) => {
                let input = self.materialize_as_simd(ops, lp, input);
                dynasm!(ops
                    ; fcvt S(r_out), D(input.r())
                )
            }
            (Register::GPR(r_out), DataType::S32, DataType::F32) => {
                warn!("TODO: this is assuming round towards zero in all cases, which is not always true");
                let input = self.materialize_as_simd(ops, lp, input);
                dynasm!(ops
                    ; fcvtzs W(r_out), S(input.r())
                )
            }
            (Register::GPR(r_out), DataType::S32, DataType::F64) => {
                warn!("TODO: this is assuming round towards zero in all cases, which is not always true");
                let value = self.materialize_as_simd(ops, lp, input);
                dynasm!(ops
                    ; fcvtzs W(r_out), D(value.r())
                )
            }
            (r_out, DataType::U64, DataType::U64) => {
                self.move_to_reg(ops, lp, input, r_out);
            }
            (r_out, DataType::U32, DataType::U32) => {
                self.move_to_reg(ops, lp, input, r_out);
            }
            (Register::SIMD(r_out), DataType::F64, DataType::F32) => {
                let value = self.materialize_as_simd(ops, lp, input);
                dynasm!(ops
                    ; fcvt D(r_out), S(value.r())
                );
            }
            _ => todo!("Unsupported convert operation: {:?} -> {:?} types {} -> {}", input, r_out, from_tp, to_tp),
        }
    }

    fn and(&self, ops: &mut Ops, lp: &mut LiteralPool, tp: DataType, r_out: Register, a: ConstOrReg, b: ConstOrReg) {
        match (tp, r_out) {
            (DataType::U32, Register::GPR(r_out)) => {
                let a = self.materialize_as_gpr(ops, lp, a);
                let b = self.materialize_as_gpr(ops, lp, b);
                dynasm!(ops
                    ; and W(r_out), W(a.r()), W(b.r())
                );
            }
            (DataType::U64, Register::GPR(r_out)) => {
                let a = self.materialize_as_gpr(ops, lp, a);
                let b = self.materialize_as_gpr(ops, lp, b);
                dynasm!(ops
                    ; and X(r_out), X(a.r()), X(b.r())
                );
            }
            (DataType::Bool, Register::GPR(r_out)) => {
                let a = self.materialize_as_gpr(ops, lp, a);
                let b = self.materialize_as_gpr(ops, lp, b);
                let r_temp = self.scratch_regs.borrow::<register_type::GPR>();
                dynasm!(ops
                    ; cmp XSP(a.r()), 0
                    ; cset X(r_temp.r()), ne
                    ; cmp XSP(b.r()), 0
                    ; cset X(r_out), ne
                    ; and X(r_out), X(r_temp.r()), X(r_out)
                );
            }
            _ => todo!("Unsupported AND operation: {:?} = {:?} & {:?} with type {:?}", r_out, a, b, tp),
        }
    }

    fn or(&self, ops: &mut Ops, lp: &mut LiteralPool, tp: DataType, r_out: Register, a: ConstOrReg, b: ConstOrReg) {
        match (tp, r_out) {
            (DataType::U32, Register::GPR(r_out)) => {
                let a = self.materialize_as_gpr(ops, lp, a);
                let b = self.materialize_as_gpr(ops, lp, b);
                dynasm!(ops
                    ; orr W(r_out), W(a.r()), W(b.r())
                );
            }
            (DataType::U64, Register::GPR(r_out)) => {
                let a = self.materialize_as_gpr(ops, lp, a);
                let b = self.materialize_as_gpr(ops, lp, b);
                dynasm!(ops
                    ; orr X(r_out), X(a.r()), X(b.r())
                );
            }
            _ => todo!("Unsupported OR operation: {:?} | {:?} with type {:?}", a, b, tp),
        }
    }

    fn not(&self, ops: &mut Ops, lp: &mut LiteralPool, tp: DataType, r_out: Register, a: ConstOrReg) {
        let r_out = r_out.expect_gpr();
        match tp {
            DataType::U32 => {
                let a = self.materialize_as_gpr(ops, lp, a);
                dynasm!(ops
                    ; mvn W(r_out), W(a.r())
                );
            }
            DataType::U64 => {
                let a = self.materialize_as_gpr(ops, lp, a);
                dynasm!(ops
                    ; mvn X(r_out), X(a.r())
                );
            }
            DataType::Bool => {
                let a = self.materialize_as_gpr(ops, lp, a);
                dynasm!(ops
                    ; cmp XSP(a.r()), 0
                    ; cset X(r_out), eq
                )
            }
            _ => todo!("Unsupported (non-const) NOT operation: GPR({}) : {} = !{:?}", r_out, tp, a),
        }
    }

    fn xor(&self, ops: &mut Ops, lp: &mut LiteralPool, tp: DataType, r_out: Register, a: ConstOrReg, b: ConstOrReg) {
        match (tp, r_out) {
            (DataType::U32, Register::GPR(r_out)) => {
                let a = self.materialize_as_gpr(ops, lp, a);
                let b = self.materialize_as_gpr(ops, lp, b);
                dynasm!(ops
                    ; eor W(r_out), W(a.r()), W(b.r())
                );
            }
            (DataType::U64, Register::GPR(r_out)) => {
                let a = self.materialize_as_gpr(ops, lp, a);
                let b = self.materialize_as_gpr(ops, lp, b);
                dynasm!(ops
                    ; eor X(r_out), X(a.r()), X(b.r())
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
        if minuend.is_const() && subtrahend.is_const() {
            match tp {
                DataType::U32 => load_32_bit_constant(
                    ops,
                    lp,
                    r_out.expect_gpr(),
                    minuend.to_u64_const().unwrap() as u32 - subtrahend.to_u64_const().unwrap() as u32,
                ),
                DataType::S32 => {
                    let result = (minuend.to_s64_const().unwrap() as i32)
                        .wrapping_sub(subtrahend.to_s64_const().unwrap() as i32)
                        as i64;
                    load_64_bit_signed_constant(ops, lp, r_out.expect_gpr(), result);
                }
                DataType::U64 => load_64_bit_constant(
                    ops,
                    lp,
                    r_out.expect_gpr(),
                    minuend.to_u64_const().unwrap() - subtrahend.to_u64_const().unwrap(),
                ),
                _ => todo!(
                    "Unsupported Sub operation with result type {} and constants: {:?} + {:?}",
                    tp,
                    minuend,
                    subtrahend
                ),
            }
            return;
        } else {
            match (tp, r_out) {
                (DataType::U32 | DataType::S32, Register::GPR(r_out)) => {
                    let minuend = self.materialize_as_gpr(ops, lp, minuend);
                    let subtrahend = self.materialize_as_gpr(ops, lp, subtrahend);
                    dynasm!(ops
                        ; sub W(r_out), W(minuend.r()), W(subtrahend.r())
                    )
                }
                (DataType::U64 | DataType::S64, Register::GPR(r_out)) => {
                    let minuend = self.materialize_as_gpr(ops, lp, minuend);
                    let subtrahend = self.materialize_as_gpr(ops, lp, subtrahend);
                    dynasm!(ops
                        ; sub X(r_out), X(minuend.r()), X(subtrahend.r())
                    )
                }
                (DataType::F32, Register::SIMD(r_out)) => {
                    let minuend = self.materialize_as_simd(ops, lp, minuend);
                    let subtrahend = self.materialize_as_simd(ops, lp, subtrahend);
                    dynasm!(ops
                        ; fsub S(r_out), S(minuend.r()), S(subtrahend.r())
                    )
                }
                (DataType::F64, Register::SIMD(r_out)) => {
                    let minuend = self.materialize_as_simd(ops, lp, minuend);
                    let subtrahend = self.materialize_as_simd(ops, lp, subtrahend);
                    dynasm!(ops
                        ; fsub D(r_out), D(minuend.r()), D(subtrahend.r())
                    )
                }
                _ => todo!("Unsupported Sub operation: {:?} - {:?} with type {:?}", minuend, subtrahend, tp),
            }
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
                let a = self.materialize_as_gpr(ops, lp, a);
                let b = self.materialize_as_gpr(ops, lp, b);
                let r_out_lo = output_regs[0].unwrap().expect_gpr();
                let r_out_hi = output_regs[1].unwrap().expect_gpr();
                dynasm!(ops
                    ; umull X(r_out_hi), W(a.r()), W(b.r())
                    ; mov W(r_out_lo), W(r_out_hi)
                    ; lsr X(r_out_hi), X(r_out_hi), 32
                );
            }
            (DataType::S32, DataType::S32, 2) => {
                let a = self.materialize_as_gpr(ops, lp, a);
                let b = self.materialize_as_gpr(ops, lp, b);
                let r_out_lo = output_regs[0].unwrap().expect_gpr();
                let r_out_hi = output_regs[1].unwrap().expect_gpr();
                dynasm!(ops
                    ; smull X(r_out_hi), W(a.r()), W(b.r())
                    ; mov W(r_out_lo), W(r_out_hi)
                    ; lsr X(r_out_hi), X(r_out_hi), 32
                );
            }
            (DataType::U64, DataType::U64, 2) => {
                let a = self.materialize_as_gpr(ops, lp, a);
                let b = self.materialize_as_gpr(ops, lp, b);
                let r_out_lo = output_regs[0].unwrap().expect_gpr();
                let r_out_hi = output_regs[1].unwrap().expect_gpr();
                dynasm!(ops
                    ; umulh X(r_out_hi), X(a.r()), X(b.r())
                    ; mul X(r_out_lo), X(a.r()), X(b.r())
                );
            }
            (DataType::S64, DataType::S64, 2) => {
                let a = self.materialize_as_gpr(ops, lp, a);
                let b = self.materialize_as_gpr(ops, lp, b);
                let r_out_lo = output_regs[0].unwrap().expect_gpr();
                let r_out_hi = output_regs[1].unwrap().expect_gpr();
                dynasm!(ops
                    ; smulh X(r_out_hi), X(a.r()), X(b.r())
                    ; mul X(r_out_lo), X(a.r()), X(b.r())
                );
            }
            (DataType::F32, DataType::F32, 1) => {
                let a = self.materialize_as_simd(ops, lp, a);
                let b = self.materialize_as_simd(ops, lp, b);
                let r_out = output_regs[0].unwrap().expect_simd();
                dynasm!(ops
                    ; fmul S(r_out), S(a.r()), S(b.r())
                );
            }
            (DataType::F64, DataType::F64, 1) => {
                let a = self.materialize_as_simd(ops, lp, a);
                let b = self.materialize_as_simd(ops, lp, b);
                let r_out = output_regs[0].unwrap().expect_simd();
                dynasm!(ops
                    ; fmul D(r_out), D(a.r()), D(b.r())
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
            DataType::S32 => {
                let dividend = self.materialize_as_gpr(ops, lp, dividend);
                let divisor = self.materialize_as_gpr(ops, lp, divisor);
                let r_quotient = r_quotient.unwrap().expect_gpr();
                let r_remainder = r_remainder.unwrap().expect_gpr();
                dynasm!(ops
                    ; sdiv W(r_quotient), W(dividend.r()), W(divisor.r())
                    ; msub W(r_remainder), W(r_quotient), W(divisor.r()), W(dividend.r())
                );
            }
            DataType::S64 => {
                let dividend = self.materialize_as_gpr(ops, lp, dividend);
                let divisor = self.materialize_as_gpr(ops, lp, divisor);
                let r_quotient = r_quotient.unwrap().expect_gpr();
                let r_remainder = r_remainder.unwrap().expect_gpr();
                dynasm!(ops
                    ; sdiv X(r_quotient), X(dividend.r()), X(divisor.r())
                    ; msub X(r_remainder), X(r_quotient), X(divisor.r()), X(dividend.r())
                );
            }
            DataType::U32 => {
                let dividend = self.materialize_as_gpr(ops, lp, dividend);
                let divisor = self.materialize_as_gpr(ops, lp, divisor);
                let r_quotient = r_quotient.unwrap().expect_gpr();
                let r_remainder = r_remainder.unwrap().expect_gpr();
                dynasm!(ops
                    ; udiv W(r_quotient), W(dividend.r()), W(divisor.r())
                    ; msub W(r_remainder), W(r_quotient), W(divisor.r()), W(dividend.r())
                );
            }
            DataType::U64 => {
                let dividend = self.materialize_as_gpr(ops, lp, dividend);
                let divisor = self.materialize_as_gpr(ops, lp, divisor);
                let r_quotient = r_quotient.unwrap().expect_gpr();
                let r_remainder = r_remainder.unwrap().expect_gpr();
                dynasm!(ops
                    ; udiv X(r_quotient), X(dividend.r()), X(divisor.r())
                    ; msub X(r_remainder), X(r_quotient), X(divisor.r()), X(dividend.r())
                );
            }
            DataType::F32 => {
                let dividend = self.materialize_as_simd(ops, lp, dividend);
                let divisor = self.materialize_as_simd(ops, lp, divisor);

                if r_remainder.is_some() {
                    panic!("Remainder is not supported for F32 division");
                }
                let r_quotient = r_quotient.unwrap().expect_simd();

                dynasm!(ops
                    ; fdiv S(r_quotient), S(dividend.r()), S(divisor.r())
                );
            }
            DataType::F64 => {
                let dividend = self.materialize_as_simd(ops, lp, dividend);
                let divisor = self.materialize_as_simd(ops, lp, divisor);
                if r_remainder.is_some() {
                    panic!("Remainder is not supported for F64 division");
                }
                let r_quotient = r_quotient.unwrap().expect_simd();

                dynasm!(ops
                    ; fdiv D(r_quotient), D(dividend.r()), D(divisor.r())
                );
            }
            _ => todo!("Unsupported Divide operation: {:?} / {:?} with type {:?}", dividend, divisor, tp),
        }
    }

    fn square_root(&self, ops: &mut Ops, lp: &mut LiteralPool, tp: DataType, r_out: Register, value: ConstOrReg) {
        match (r_out, tp) {
            (Register::SIMD(r_out), DataType::F32) => {
                let value = self.materialize_as_simd(ops, lp, value);
                dynasm!(ops
                    ; fsqrt S(r_out), S(value.r())
                );
            }
            (Register::SIMD(r_out), DataType::F64) => {
                let value = self.materialize_as_simd(ops, lp, value);
                dynasm!(ops
                    ; fsqrt D(r_out), D(value.r())
                );
            }
            _ => todo!("Unsupported SquareRoot operation: ({:?}, {:?}, {:?})", r_out, tp, value),
        }
    }

    fn absolute_value(&self, ops: &mut Ops, lp: &mut LiteralPool, tp: DataType, r_out: Register, value: ConstOrReg) {
        match (tp, r_out) {
            (DataType::F32, Register::SIMD(r_out)) => {
                let value = self.materialize_as_simd(ops, lp, value);
                dynasm!(ops
                    ; fabs S(r_out), S(value.r())
                );
            }
            (DataType::F64, Register::SIMD(r_out)) => {
                let value = self.materialize_as_simd(ops, lp, value);
                dynasm!(ops
                    ; fabs D(r_out), D(value.r())
                );
            }
            _ => todo!("Unsupported AbsoluteValue operation: ({:?}, {:?})", tp, r_out),
        }
    }

    fn negate(&self, ops: &mut Ops, lp: &mut LiteralPool, tp: DataType, r_out: Register, value: ConstOrReg) {
        match (tp, r_out) {
            (DataType::F32, Register::SIMD(r_out)) => {
                let value = self.materialize_as_simd(ops, lp, value);
                dynasm!(ops
                    ; fneg S(r_out), S(value.r())
                );
            }
            (DataType::F64, Register::SIMD(r_out)) => {
                let value = self.materialize_as_simd(ops, lp, value);
                dynasm!(ops
                    ; fneg D(r_out), D(value.r())
                );
            }
            _ => todo!("Unsupported Negate operation: ({:?}, {:?}) with value {:?}", tp, r_out, value),
        }
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
            .chain(iter::once(reg_constants::LR))
            .chain(active_volatile_regs.into_iter())
            .collect::<Vec<_>>();

        let stack_bytes_needed = active_regs.iter().map(|r| r.size()).sum::<usize>();
        let misalignment = stack_bytes_needed % 16;
        let stack_bytes_needed = stack_bytes_needed + misalignment;

        dynasm!(ops
            ; sub sp, sp, stack_bytes_needed as u32 // Allocate stack space for the call
        );

        let mut stack_offsets = BTreeMap::new();
        let mut stack_offset = 0;
        for reg in active_regs.iter() {
            stack_offsets.insert(reg, stack_offset);
            match reg {
                Register::GPR(r) => {
                    dynasm!(ops
                        ; str X(*r), [sp, stack_offset]
                    );
                }
                Register::SIMD(_r) => todo!(),
            }
            stack_offset += reg.size() as u32;
        }

        // Move the arguments into place
        let moves = args
            .into_iter()
            .zip(get_function_argument_registers().into_iter())
            .collect::<BTreeMap<ConstOrReg, Register>>();
        self.move_regs_multi(ops, lp, moves);

        match address {
            ConstOrReg::U64(ptr) => {
                let temp_reg = self.scratch_regs.borrow::<register_type::GPR>();
                load_64_bit_constant(ops, lp, temp_reg.r(), ptr);
                dynasm!(ops
                    ; blr X(temp_reg.r())
                );
            }
            _ => todo!("Unsupported call to: {:?}", address),
        }

        if let Some(to) = r_out {
            debug!("Moving return value from {} to {}", get_return_value_registers()[0], to);
            self.move_to_reg(ops, lp, get_return_value_registers()[0].to_const_or_reg(), to);
        }

        for reg in active_regs.iter() {
            match reg {
                Register::GPR(r) => {
                    dynasm!(ops
                        ; ldr X(*r), [sp, stack_offsets[reg]]
                    );
                }
                Register::SIMD(_r) => todo!(),
            }
        }
        dynasm!(ops
            ; add sp, sp, stack_bytes_needed as u32 // Deallocate stack space for the call
        );
    }
}
