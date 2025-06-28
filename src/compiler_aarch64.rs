use std::{collections::BTreeMap, iter, marker::PhantomData};

use crate::{
    abi::{get_function_argument_registers, get_return_value_registers, get_scratch_registers, reg_constants},
    compiler::{Compiler, ConstOrReg, GenericAssembler, LiteralPool},
    ir::{BlockReference, CompareType, Constant, DataType, IRFunctionInternal},
    reg_pool::{register_type, RegPool},
    register_allocator::{alloc_for, Register, RegisterAllocations},
};
use dynasmrt::{aarch64::Aarch64Relocation, dynasm, Assembler, AssemblyOffset, VecAssembler};

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
    reg: u32,
    value: u64,
) {
    println!("Loading 64-bit constant: 0x{:X}", value);
    if value <= 0xFFFF {
        dynasm!(ops
            ; movz X(reg), value as u32
        );
    } else {
        // TODO: check if the value can fit in a U16, U32, etc and zero extend when loading
        let literal = Aarch64Compiler::add_literal(ops, lp, Constant::U64(value));
        println!("Loading using literal pool");
        dynasm!(ops
            ; ldr X(reg), =>literal
        );
    }
}

fn load_64_bit_signed_constant<Ops: GenericAssembler<Aarch64Relocation>>(
    ops: &mut Ops,
    lp: &mut LiteralPool,
    reg: u32,
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
    reg: u32,
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

    fn new(ops: &mut Ops, func: &'a mut IRFunctionInternal) -> Self {
        let allocations = alloc_for(func);

        // Stack bytes used: aligned to 16 bytes
        let misalignment = func.stack_bytes_used % 16;
        let correction = if misalignment == 0 { 0 } else { 16 - misalignment };
        let stack_bytes_used = func.stack_bytes_used + correction;
        println!(
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
                        ; str X(*r as u32), [sp, self.func.get_stack_offset_for_location(*stack_location as u64, DataType::U64) as u32]
                    )
                }
                Register::SIMD(r) => {
                    assert_eq!(reg.size(), 16);
                    dynasm!(ops
                        ; str Q(*r as u32), [sp, self.func.get_stack_offset_for_location(*stack_location as u64, DataType::U128)]
                    );
                }
            }
        }
    }

    fn epilogue(&self, _ops: &mut Ops) {
        println!("Epilogue: emitting nothing");
    }

    fn jump_to_dynamic_label(&self, ops: &mut Ops, label: dynasmrt::DynamicLabel) {
        dynasm!(ops
            ; b =>label
        )
    }

    fn move_to_reg(&self, ops: &mut Ops, lp: &mut LiteralPool, from: ConstOrReg, to: Register) {
        match (from, to) {
            (ConstOrReg::U32(c), Register::GPR(r_to)) => {
                load_32_bit_constant(ops, lp, r_to as u32, c);
            }
            (ConstOrReg::S32(c), Register::GPR(r_to)) => {
                load_64_bit_signed_constant(ops, lp, r_to as u32, c.into());
            }
            (ConstOrReg::U64(c), Register::GPR(r)) => {
                load_64_bit_constant(ops, lp, r as u32, c);
            }
            (ConstOrReg::S64(c), Register::GPR(r)) => {
                load_64_bit_signed_constant(ops, lp, r as u32, c);
            }
            (ConstOrReg::GPR(r_from), Register::GPR(r_to)) => {
                dynasm!(ops
                    ; mov X(r_to as u32), X(r_from as u32)
                );
            }
            (ConstOrReg::SIMD(r_from), Register::SIMD(r_to)) => {
                dynasm!(ops
                    ; mov V(r_to as u32).B16, V(r_from as u32).B16
                )
            }
            _ => todo!("Unimplemented move operation: {:?} to {:?}", from, to),
        }
    }

    // TODO: this is exactly the same in all compilers, figure out how to share this
    fn emit_literal_pool(&self, ops: &mut Ops, lp: LiteralPool) {
        for (literal, label) in lp.literals {
            println!("Aligning to {} bytes for literal {:?}", literal.size(), literal);
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
                    ; cbz W(c), >if_false
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
                        ; ldr X(r as u32), [sp, self.func.get_stack_offset_for_location(*stack_location as u64, DataType::U64) as u32]
                    )
                }
                Register::SIMD(r) => {
                    assert_eq!(reg.size(), 16);
                    dynasm!(ops
                        ; ldr Q(r as u32), [sp, self.func.get_stack_offset_for_location(*stack_location as u64, DataType::U128)]
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
        if a.is_const() && b.is_const() {
            match tp {
                DataType::U32 => load_32_bit_constant(
                    ops,
                    lp,
                    r_out.expect_gpr() as u32,
                    a.to_u64_const().unwrap() as u32 + b.to_u64_const().unwrap() as u32,
                ),
                DataType::S32 => {
                    let result =
                        (a.to_s64_const().unwrap() as i32).wrapping_add(b.to_s64_const().unwrap() as i32) as i64;
                    load_64_bit_signed_constant(ops, lp, r_out.expect_gpr() as u32, result);
                }
                DataType::U64 => load_64_bit_constant(
                    ops,
                    lp,
                    r_out.expect_gpr() as u32,
                    a.to_u64_const().unwrap() + b.to_u64_const().unwrap(),
                ),
                _ => todo!("Unsupported Add operation with result type {} and constants: {:?} + {:?}", tp, a, b),
            }
            return;
        } else {
            match (tp, r_out, a, b) {
                (DataType::U32 | DataType::S32, Register::GPR(r_out), ConstOrReg::GPR(r), c) if c.is_const() => {
                    let c = b.to_s64_const().unwrap();
                    if c > 0 && c < 4096 {
                        dynasm!(ops
                            ; add WSP(r_out as u32), WSP(r), c as u32
                        )
                    } else if c < 0 && c > -4096 {
                        let c = c.abs() as u32;
                        dynasm!(ops
                            ; sub WSP(r_out as u32), WSP(r), c
                        );
                    } else {
                        let r_temp = self.scratch_regs.borrow::<register_type::GPR>();
                        load_64_bit_signed_constant(ops, lp, r_temp.r(), c);
                        dynasm!(ops
                            ; add W(r_out as u32), W(r), W(r_temp.r())
                        )
                    }
                }
                (DataType::U64 | DataType::S64, Register::GPR(r_out), ConstOrReg::GPR(r), c) if c.is_const() => {
                    let c = c.to_s64_const().unwrap();
                    if c > 0 && c < 4096 {
                        dynasm!(ops
                            ; add XSP(r_out as u32), XSP(r), c as u32
                        )
                    } else if c < 0 && c > -4096 {
                        let c = c.abs() as u32;
                        dynasm!(ops
                            ; sub XSP(r_out as u32), XSP(r), c
                        );
                    } else {
                        let r_temp = self.scratch_regs.borrow::<register_type::GPR>();
                        load_64_bit_constant(ops, lp, r_temp.r(), c as u64);
                        dynasm!(ops
                            ; add X(r_out as u32), X(r), X(r_temp.r())
                        )
                    }
                }
                (DataType::U32 | DataType::S32, Register::GPR(r_out), ConstOrReg::GPR(r1), ConstOrReg::GPR(r2)) => {
                    dynasm!(ops
                        ; add W(r_out as u32), W(r1), W(r2)
                    )
                }
                (DataType::F32, Register::SIMD(r_out), ConstOrReg::SIMD(r1), ConstOrReg::SIMD(r2)) => {
                    dynasm!(ops
                        ; fadd S(r_out as u32), S(r1), S(r2)
                    )
                }
                (DataType::F32, Register::SIMD(r_out), ConstOrReg::SIMD(r), ConstOrReg::F32(c)) => {
                    dynasm!(ops
                        ; fmov S(r_out as u32), *c
                        ; fadd S(r_out as u32), S(r_out as u32), S(r)
                    )
                }
                _ => todo!("Unsupported Add operation: {:?} + {:?} with type {:?}", a, b, tp),
            }
        }
    }

    fn compare(
        &self,
        ops: &mut Ops,
        lp: &mut LiteralPool,
        r_out: usize,
        a: ConstOrReg,
        cmp_type: CompareType,
        b: ConstOrReg,
    ) {
        fn set_reg_by_flags<Ops: GenericAssembler<Aarch64Relocation>>(
            ops: &mut Ops,
            cmp_type: CompareType,
            r_out: usize,
        ) {
            // https://developer.arm.com/documentation/100076/0100/A64-Instruction-Set-Reference/A64-General-Instructions/CSET
            // https://developer.arm.com/documentation/100076/0100/A64-Instruction-Set-Reference/Condition-Codes/Condition-code-suffixes-and-related-flags?lang=en
            match cmp_type {
                CompareType::LessThanUnsigned => {
                    dynasm!(ops
                        ; cset W(r_out as u32), lo // unsigned "lower"
                    )
                }
                CompareType::Equal => {
                    dynasm!(ops
                        ; cset W(r_out as u32), eq // "equal"
                    )
                }
                CompareType::NotEqual => {
                    dynasm!(ops
                        ; cset W(r_out as u32), ne // "not equal"
                    )
                }
                CompareType::LessThanSigned => {
                    dynasm!(ops
                        ; cset W(r_out as u32), lt // signed "less than"
                    )
                }
                CompareType::GreaterThanSigned => todo!("Compare with type GreaterThanSigned"),
                CompareType::LessThanOrEqualSigned => {
                    dynasm!(ops
                        ; cset W(r_out as u32), le // signed "less than or equal"
                    )
                }
                CompareType::GreaterThanOrEqualSigned => {
                    dynasm!(ops
                        ; cset W(r_out as u32), ge // signed "greater than or equal"
                    )
                }
                CompareType::GreaterThanUnsigned => todo!("Compare with type GreaterThanUnsigned"),
                CompareType::LessThanOrEqualUnsigned => todo!("Compare with type LessThanOrEqualUnsigned"),
                CompareType::GreaterThanOrEqualUnsigned => todo!("Compare with type GreaterThanOrEqualUnsigned"),
            }
        }

        match (a, b) {
            (ConstOrReg::GPR(r1), ConstOrReg::GPR(r2)) => {
                dynasm!(ops
                    ; cmp X(r1 as u32), X(r2 as u32)
                );
                set_reg_by_flags(ops, cmp_type, r_out);
            }
            (ConstOrReg::GPR(r), c) if c.is_const() => {
                let c = c.to_u64_const().unwrap();
                if c < 4096 {
                    dynasm!(ops
                        ; cmp XSP(r as u32), c as u32
                    );
                    set_reg_by_flags(ops, cmp_type, r_out);
                } else {
                    todo!("Too big a constant here, load it to a temp and compare")
                }
            }
            (c, ConstOrReg::GPR(r)) if c.is_const() => {
                let c = c.to_u64_const().unwrap();
                let r_temp = self.scratch_regs.borrow::<register_type::GPR>();
                load_64_bit_constant(ops, lp, r_temp.r(), c);
                dynasm!(ops
                    ; cmp XSP(r_temp.r()), X(r as u32)
                );
                set_reg_by_flags(ops, cmp_type, r_out);
            }
            (c1, c2) if c1.is_const() && c2.is_const() => match cmp_type {
                CompareType::Equal => {
                    dynasm!(ops
                        ; mov W(r_out as u32), (c1.to_u64_const().unwrap() == c2.to_u64_const().unwrap()) as u32
                    )
                }
                CompareType::NotEqual => {
                    dynasm!(ops
                        ; mov W(r_out as u32), (c1.to_u64_const().unwrap() != c2.to_u64_const().unwrap()) as u32
                    )
                }
                CompareType::LessThanSigned => todo!("Compare constants with type LessThanSigned"),
                CompareType::GreaterThanSigned => todo!("Compare constants with type GreaterThanSigned"),
                CompareType::LessThanOrEqualSigned => todo!("Compare constants with type LessThanOrEqualSigned"),
                CompareType::GreaterThanOrEqualSigned => {
                    dynasm!(ops
                        ; mov W(r_out as u32), (c1.to_s64_const().unwrap() >= c2.to_s64_const().unwrap()) as u32
                    )
                }
                CompareType::LessThanUnsigned => todo!("Compare constants with type LessThanUnsigned"),
                CompareType::GreaterThanUnsigned => todo!("Compare constants with type GreaterThanUnsigned"),
                CompareType::LessThanOrEqualUnsigned => todo!("Compare constants with type LessThanOrEqualUnsigned"),
                CompareType::GreaterThanOrEqualUnsigned => {
                    todo!("Compare constants with type GreaterThanOrEqualUnsigned")
                }
            },
            _ => todo!("Unsupported Compare operation: {:?} = {:?} {:?} {:?}", r_out, a, cmp_type, b),
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
                    ; ldr W(r_out as u32), [X(r_ptr.r()), offset as u32]
                );
            }
            (Register::GPR(r_out), ConstOrReg::GPR(r_ptr), DataType::U64) => {
                dynasm!(ops
                    ; ldr X(r_out as u32), [X(r_ptr), offset as u32]
                );
            }
            (Register::GPR(r_out), ConstOrReg::GPR(r_ptr), DataType::U32 | DataType::S32) => {
                dynasm!(ops
                    ; ldr W(r_out as u32), [X(r_ptr), offset as u32]
                );
            }
            _ => todo!("Unsupported LoadPtr operation: Load [{:?}] with type {}", ptr, tp),
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
                    ; str W(r_value as u32), [X(r_ptr.r()), offset as u32]
                );
            }
            (ConstOrReg::GPR(r_ptr), ConstOrReg::U32(c_value), DataType::U32) => {
                let r_value = self.scratch_regs.borrow::<register_type::GPR>();
                load_64_bit_constant(ops, lp, r_value.r(), c_value.into());
                dynasm!(ops
                    ; str W(r_value.r() as u32), [X(r_ptr), offset as u32]
                );
            }
            (ConstOrReg::GPR(r_ptr), ConstOrReg::GPR(r_value), DataType::U32 | DataType::S32) => {
                dynasm!(ops
                    ; str W(r_value as u32), [X(r_ptr as u32), offset as u32]
                );
            }
            (ConstOrReg::GPR(r_ptr), ConstOrReg::GPR(r_value), DataType::U64 | DataType::S64) => {
                dynasm!(ops
                    ; str X(r_value as u32), [X(r_ptr as u32), offset as u32]
                );
            }
            (ConstOrReg::GPR(r_ptr), ConstOrReg::U64(value), DataType::U64) => {
                let r_value = self.scratch_regs.borrow::<register_type::GPR>();
                load_64_bit_constant(ops, lp, r_value.r(), value);
                dynasm!(ops
                    ; str X(r_value.r() as u32), [X(r_ptr as u32), offset as u32]
                );
            }
            (ConstOrReg::GPR(r_ptr), c, DataType::U64 | DataType::S64) if c.is_const() => {
                let r_value = self.scratch_regs.borrow::<register_type::GPR>();
                load_64_bit_constant(ops, lp, r_value.r(), c.to_u64_const().unwrap());
                dynasm!(ops
                    ; str X(r_value.r() as u32), [X(r_ptr as u32), offset as u32]
                );
            }
            _ => todo!("Unsupported WritePtr operation: {:?} = {:?} with type {}", ptr, value, data_type),
        }
    }

    fn spill_to_stack(&self, ops: &mut Ops, to_spill: ConstOrReg, stack_offset: ConstOrReg, tp: DataType) {
        match (&to_spill, &stack_offset, tp) {
            (ConstOrReg::GPR(r), ConstOrReg::U64(offset), DataType::U8 | DataType::S8) => {
                dynasm!(ops
                    ; strb W(*r), [sp, self.func.get_stack_offset_for_location(*offset, DataType::U8) as u32]
                )
            }
            (ConstOrReg::GPR(r), ConstOrReg::U64(offset), DataType::U32 | DataType::S32) => {
                dynasm!(ops
                    ; str W(*r), [sp, self.func.get_stack_offset_for_location(*offset, DataType::U32) as u32]
                )
            }
            (ConstOrReg::GPR(r), ConstOrReg::U64(offset), DataType::U64 | DataType::S64 | DataType::Ptr) => {
                dynasm!(ops
                    ; str X(*r), [sp, self.func.get_stack_offset_for_location(*offset, DataType::U64) as u32]
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
            (Register::GPR(r_out), ConstOrReg::U64(offset), DataType::U8) => {
                dynasm!(ops
                    ; ldrb W(r_out as u32), [sp, self.func.get_stack_offset_for_location(*offset, DataType::U8) as u32]
                )
            }
            (Register::GPR(r_out), ConstOrReg::U64(offset), DataType::U32) => {
                dynasm!(ops
                    ; ldr W(r_out as u32), [sp, self.func.get_stack_offset_for_location(*offset, DataType::U32) as u32]
                )
            }
            (Register::GPR(r_out), ConstOrReg::U64(offset), DataType::U64 | DataType::S64 | DataType::Ptr) => {
                dynasm!(ops
                    ; ldr X(r_out as u32), [sp, self.func.get_stack_offset_for_location(*offset, DataType::U64) as u32]
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

    fn left_shift(&self, ops: &mut Ops, r_out: usize, n: ConstOrReg, amount: ConstOrReg, tp: DataType) -> () {
        if let Some(amount) = amount.to_u64_const() {
            let amount = amount as u32;
            match (tp, n) {
                (DataType::U8 | DataType::S8, ConstOrReg::GPR(r_n)) => {
                    dynasm!(ops
                        ; lsl W(r_out as u32), W(r_n as u32), amount & 0b111
                        ; and WSP(r_out as u32), W(r_out as u32), 0xFF // Mask to 8 bits
                    );
                }
                (DataType::U16 | DataType::S16, ConstOrReg::GPR(r_n)) => {
                    dynasm!(ops
                        ; lsl W(r_out as u32), W(r_n as u32), amount & 0b1111
                        ; and WSP(r_out as u32), W(r_out as u32), 0xFFFF // Mask to 16
                    );
                }
                (DataType::U32 | DataType::S32, ConstOrReg::GPR(r_n)) => {
                    dynasm!(ops
                        ; lsl W(r_out as u32), W(r_n as u32), amount & 0b11111
                    );
                }
                (DataType::U64 | DataType::S64, ConstOrReg::GPR(r_n)) => {
                    dynasm!(ops
                        ; lsl X(r_out as u32), X(r_n as u32), amount & 0b111111
                    );
                }
                _ => todo!("Unsupported LeftShift operation: {:?} << {:?} with type {}", n, amount, tp),
            }
        } else if let Some(Register::GPR(r_amount)) = amount.to_reg() {
            match (tp, n) {
                (DataType::U8, ConstOrReg::GPR(_r_n)) => todo!("LeftShift with GPR amount for U8"),
                (DataType::S8, ConstOrReg::GPR(_r_n)) => todo!("LeftShift with GPR amount for S8"),
                (DataType::U16, ConstOrReg::GPR(_r_n)) => todo!("LeftShift with GPR amount for U16"),
                (DataType::S16, ConstOrReg::GPR(_r_n)) => todo!("LeftShift with GPR amount for S16"),
                (DataType::U32, ConstOrReg::GPR(r_n)) => {
                    dynasm!(ops
                        ; lslv W(r_out as u32), W(r_n as u32), W(r_amount as u32)
                    );
                }
                (DataType::S32, ConstOrReg::GPR(_r_n)) => todo!("LeftShift with GPR amount for S32"),
                (DataType::U64, ConstOrReg::GPR(_r_n)) => todo!("LeftShift with GPR amount for U64"),
                (DataType::S64, ConstOrReg::GPR(_r_n)) => todo!("LeftShift with GPR amount for S64"),

                _ => todo!("Unsupported DataType {} or unsupported register type for LeftShift operation with GPR amount: {:?} << {:?}", tp, n, r_amount),
            }
        } else {
            panic!("RightShift amount must be a constant or a GPR, got: {:?}", amount);
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
            match (tp, n) {
                (DataType::U8, ConstOrReg::GPR(r_n)) => {
                    dynasm!(ops
                        ; and WSP(r_out as u32), W(r_n as u32), 0xFF
                        ; lsr W(r_out as u32), W(r_out as u32), amount & 0b111
                    );
                }
                (DataType::S8, ConstOrReg::GPR(r_n)) => {
                    // Shift left to put the sign bit in the 32 bit sign bit position, then shift
                    // right.
                    dynasm!(ops
                        ; lsl W(r_out as u32), W(r_n as u32), 24
                        ; asr W(r_out as u32), W(r_out as u32), (amount & 0b111) + 24
                        ; and WSP(r_out as u32), W(r_out as u32), 0xFF // Mask to 8 bits
                    );
                }
                (DataType::U16, ConstOrReg::GPR(r_n)) => {
                    dynasm!(ops
                        ; and WSP(r_out as u32), W(r_n as u32), 0xFFFF
                        ; lsr W(r_out as u32), W(r_out as u32), amount & 0b1111
                    );
                }
                (DataType::S16, ConstOrReg::GPR(r_n)) => {
                    // Shift left to put the sign bit in the 32 bit sign bit position, then shift
                    // right.
                    dynasm!(ops
                        ; lsl W(r_out as u32), W(r_n as u32), 16
                        ; asr W(r_out as u32), W(r_out as u32), (amount & 0b1111) + 16
                        ; and WSP(r_out as u32), W(r_out as u32), 0xFFFF // Mask to 16 bits
                    );
                }
                (DataType::U32, ConstOrReg::GPR(r_n)) => {
                    dynasm!(ops
                        ; lsr W(r_out as u32), W(r_n as u32), amount & 0b11111
                    );
                }
                (DataType::U32, c) if c.is_const() => {
                    let c = c.to_u64_const().unwrap() as u32;
                    let c = c >> (amount & 0b11111);
                    load_32_bit_constant(ops, lp, r_out as u32, c);
                }
                (DataType::S32, ConstOrReg::GPR(r_n)) => {
                    dynasm!(ops
                        ; asr W(r_out as u32), W(r_n as u32), amount & 0b11111
                    );
                }
                (DataType::U64, ConstOrReg::GPR(r_n)) => {
                    dynasm!(ops
                        ; lsr X(r_out as u32), X(r_n as u32), amount & 0b111111
                    );
                }
                (DataType::S64, ConstOrReg::GPR(r_n)) => {
                    dynasm!(ops
                        ; asr X(r_out as u32), X(r_n as u32), amount & 0b111111
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
                        ; lsrv W(r_out as u32), W(r_n as u32), W(r_amount as u32)
                    );
                }
                (DataType::S32, ConstOrReg::GPR(_r_n)) => todo!("RightShift with GPR amount for S32"),
                (DataType::U64, ConstOrReg::GPR(_r_n)) => todo!("RightShift with GPR amount for U64"),
                (DataType::S64, ConstOrReg::GPR(_r_n)) => todo!("RightShift with GPR amount for S64"),

                _ => todo!("Unsupported DataType {} or unsupported register type for RightShift operation with GPR amount: {:?} >> {:?}", tp, n, r_amount),
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
        match (r_out, to_tp, input, from_tp) {
            (Register::GPR(r_out), DataType::U64, ConstOrReg::GPR(r_in), DataType::U32) => {
                dynasm!(ops
                    // mov to a 32 bit register zero extends it by default
                    ; mov W(r_out as u32), W(r_in as u32)
                );
            }
            (Register::GPR(r_out), DataType::S64, ConstOrReg::GPR(r_in), DataType::S32) => {
                dynasm!(ops
                    ; sxtw X(r_out as u32), W(r_in as u32)
                );
            }
            (Register::GPR(r_out), DataType::U32, ConstOrReg::U32(c), DataType::U32) => {
                load_32_bit_constant(ops, lp, r_out as u32, c);
            }
            (Register::GPR(r_out), DataType::S64, c, DataType::S32) if c.is_const() => {
                let c = (c.to_u64_const().unwrap() & 0xFFFFFFFF) as i32 as i64;
                load_64_bit_signed_constant(ops, lp, r_out as u32, c);
            }
            _ => todo!("Unsupported convert operation: {:?} -> {:?} types {} -> {}", input, r_out, from_tp, to_tp),
        }
    }

    fn and(&self, ops: &mut Ops, lp: &mut LiteralPool, tp: DataType, r_out: Register, a: ConstOrReg, b: ConstOrReg) {
        if a.is_const() && b.is_const() {
            let result = a.to_u64_const().unwrap() & b.to_u64_const().unwrap();
            match tp {
                DataType::U32 => load_32_bit_constant(ops, lp, r_out.expect_gpr() as u32, result as u32),
                _ => todo!("Unsupported AND operation with constants: {:?} & {:?} with type {:?}", a, b, tp),
            }
        } else {
            match (tp, r_out, a, b) {
                (DataType::U32, Register::GPR(r_out), ConstOrReg::GPR(r), ConstOrReg::GPR(r2)) => {
                    dynasm!(ops
                        ; and W(r_out as u32), W(r), W(r2)
                    );
                }
                (DataType::U64, Register::GPR(r_out), ConstOrReg::GPR(r), ConstOrReg::GPR(r2)) => {
                    dynasm!(ops
                        ; and X(r_out as u32), X(r), X(r2)
                    );
                }
                (DataType::U32, Register::GPR(r_out), ConstOrReg::GPR(r), c) if c.is_const() => {
                    let c = c.to_u64_const().unwrap() & 0xFFFFFFFF;
                    let r_temp = self.scratch_regs.borrow::<register_type::GPR>();
                    load_32_bit_constant(ops, lp, r_temp.r(), c as u32);
                    dynasm!(ops
                        ; and W(r_out as u32), W(r), W(r_temp.r())
                    );
                }
                (DataType::U64, Register::GPR(r_out), ConstOrReg::U32(c1), ConstOrReg::U16(c2)) => {
                    load_64_bit_constant(ops, lp, r_out as u32, c1 as u64 & c2 as u64);
                }
                (DataType::U64, Register::GPR(r_out), ConstOrReg::GPR(r), c) if c.is_const() => {
                    let c = c.to_u64_const().unwrap();
                    // TODO: if the const is small enough, use an and immediate
                    load_64_bit_constant(ops, lp, r_out as u32, c);
                    dynasm!(ops
                        ; and X(r_out as u32), X(r), X(r_out as u32)
                    );
                }
                _ => todo!("Unsupported AND operation: {:?} & {:?} with type {:?}", a, b, tp),
            }
        }
    }

    fn or(&self, ops: &mut Ops, lp: &mut LiteralPool, tp: DataType, r_out: Register, a: ConstOrReg, b: ConstOrReg) {
        let a_const = a.to_u64_const();
        let b_const = b.to_u64_const();
        if a_const.is_some() && b_const.is_some() {
            let result = a_const.unwrap() | b_const.unwrap();
            load_64_bit_constant(ops, lp, r_out.expect_gpr() as u32, result);
        } else {
            match (tp, r_out, a, b) {
                (DataType::U32, Register::GPR(r_out), ConstOrReg::GPR(r), ConstOrReg::GPR(r2)) => {
                    dynasm!(ops
                        ; orr W(r_out as u32), W(r), W(r2)
                    );
                }
                (DataType::U64, Register::GPR(r_out), ConstOrReg::GPR(r), ConstOrReg::GPR(r2)) => {
                    dynasm!(ops
                        ; orr X(r_out as u32), X(r), X(r2)
                    );
                }
                (DataType::U64, Register::GPR(r_out), ConstOrReg::GPR(r), c) if c.is_const() => {
                    load_64_bit_constant(ops, lp, r_out as u32, c.to_u64_const().unwrap());
                    dynasm!(ops
                        ; orr X(r_out as u32), X(r), X(r_out as u32)
                    );
                }
                (DataType::U64, Register::GPR(r_out), c, ConstOrReg::GPR(r)) if c.is_const() => {
                    load_64_bit_constant(ops, lp, r_out as u32, c.to_u64_const().unwrap());
                    dynasm!(ops
                        ; orr X(r_out as u32), X(r), X(r_out as u32)
                    );
                }
                _ => todo!("Unsupported OR operation: {:?} | {:?} with type {:?}", a, b, tp),
            }
        }
    }

    fn not(&self, ops: &mut Ops, lp: &mut LiteralPool, tp: DataType, r_out: Register, a: ConstOrReg) {
        let r_out = r_out.expect_gpr();
        if a.is_const() {
            let result = !a.to_u64_const().unwrap();
            match tp {
                DataType::U32 => load_32_bit_constant(ops, lp, r_out as u32, result as u32),
                DataType::S32 => load_64_bit_signed_constant(ops, lp, r_out as u32, result as i32 as i64),
                DataType::U64 => load_64_bit_constant(ops, lp, r_out as u32, result),
                _ => todo!("Unsupported NOT operation with constant: {:?} with type {:?}", a, tp),
            }
        } else {
            todo!("Unsupported (non-const) NOT operation: {:?}", a);
        }
    }

    fn xor(&self, ops: &mut Ops, lp: &mut LiteralPool, tp: DataType, r_out: Register, a: ConstOrReg, b: ConstOrReg) {
        let a_const = a.to_u64_const();
        let b_const = b.to_u64_const();
        if a_const.is_some() && b_const.is_some() {
            let result = a_const.unwrap() ^ b_const.unwrap();
            load_64_bit_constant(ops, lp, r_out.expect_gpr() as u32, result);
        } else {
            match (tp, r_out, a, b) {
                (DataType::U32, Register::GPR(r_out), ConstOrReg::GPR(r), ConstOrReg::GPR(r2)) => {
                    dynasm!(ops
                        ; eor W(r_out as u32), W(r), W(r2)
                    );
                }
                (DataType::U64, Register::GPR(r_out), ConstOrReg::GPR(r), ConstOrReg::GPR(r2)) => {
                    dynasm!(ops
                        ; eor X(r_out as u32), X(r), X(r2)
                    );
                }
                (DataType::U64, Register::GPR(r_out), ConstOrReg::GPR(r), c) if c.is_const() => {
                    load_64_bit_constant(ops, lp, r_out as u32, c.to_u64_const().unwrap());
                    dynasm!(ops
                        ; eor X(r_out as u32), X(r), X(r_out as u32)
                    );
                }
                _ => todo!("Unsupported XOR operation: {:?} ^ {:?} with type {:?}", a, b, tp),
            }
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
                    r_out.expect_gpr() as u32,
                    minuend.to_u64_const().unwrap() as u32 - subtrahend.to_u64_const().unwrap() as u32,
                ),
                DataType::S32 => {
                    let result = (minuend.to_s64_const().unwrap() as i32)
                        .wrapping_sub(subtrahend.to_s64_const().unwrap() as i32)
                        as i64;
                    load_64_bit_signed_constant(ops, lp, r_out.expect_gpr() as u32, result);
                }
                DataType::U64 => load_64_bit_constant(
                    ops,
                    lp,
                    r_out.expect_gpr() as u32,
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
            match (tp, r_out, minuend, subtrahend) {
                (DataType::U32 | DataType::S32, Register::GPR(r_out), ConstOrReg::GPR(r1), ConstOrReg::GPR(r2)) => {
                    dynasm!(ops
                        ; sub W(r_out as u32), W(r1), W(r2)
                    )
                }
                _ => todo!("Unsupported Sub operation: {:?} - {:?} with type {:?}", minuend, subtrahend, tp),
            }
        }
    }

    fn multiply(
        &self,
        ops: &mut Ops,
        _lp: &mut LiteralPool,
        result_tp: DataType,
        arg_tp: DataType,
        output_regs: Vec<Option<Register>>,
        a: ConstOrReg,
        b: ConstOrReg,
    ) {
        match (result_tp, arg_tp, output_regs.len(), a, b) {
            (DataType::U64, DataType::U32, 2, ConstOrReg::GPR(r_a), ConstOrReg::GPR(r_b)) => {
                let r_out_lo = output_regs[0].unwrap().expect_gpr();
                let r_out_hi = output_regs[1].unwrap().expect_gpr();
                dynasm!(ops
                    ; umull X(r_out_hi as u32), W(r_a as u32), W(r_b as u32)
                    ; mov W(r_out_lo as u32), W(r_out_hi as u32)
                    ; lsr X(r_out_hi as u32), X(r_out_hi as u32), 32
                );
            }
            (DataType::S64, DataType::S32, 2, ConstOrReg::GPR(r_a), ConstOrReg::GPR(r_b)) => {
                let r_out_lo = output_regs[0].unwrap().expect_gpr();
                let r_out_hi = output_regs[1].unwrap().expect_gpr();
                dynasm!(ops
                    ; smull X(r_out_hi as u32), W(r_a as u32), W(r_b as u32)
                    ; mov W(r_out_lo as u32), W(r_out_hi as u32)
                    ; lsr X(r_out_hi as u32), X(r_out_hi as u32), 32
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
        _lp: &mut LiteralPool,
        tp: DataType,
        r_quotient: Option<Register>,
        r_remainder: Option<Register>,
        dividend: ConstOrReg,
        divisor: ConstOrReg,
    ) {
        match (tp, dividend, divisor) {
            (DataType::S32, ConstOrReg::GPR(r_dividend), ConstOrReg::GPR(r_divisor)) => {
                let r_quotient = r_quotient.unwrap().expect_gpr();
                let r_remainder = r_remainder.unwrap().expect_gpr();
                dynasm!(ops
                    ; sdiv W(r_quotient as u32), W(r_dividend as u32), W(r_divisor as u32)
                    ; msub W(r_remainder as u32), W(r_quotient as u32), W(r_divisor as u32), W(r_dividend as u32)
                );
            }
            (DataType::U32, ConstOrReg::GPR(r_dividend), ConstOrReg::GPR(r_divisor)) => {
                let r_quotient = r_quotient.unwrap().expect_gpr();
                let r_remainder = r_remainder.unwrap().expect_gpr();
                dynasm!(ops
                    ; udiv W(r_quotient as u32), W(r_dividend as u32), W(r_divisor as u32)
                    ; msub W(r_remainder as u32), W(r_quotient as u32), W(r_divisor as u32), W(r_dividend as u32)
                );
            }
            _ => todo!("Unsupported Divide operation: {:?} / {:?} with type {:?}", dividend, divisor, tp),
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
                        ; str X(*r as u32), [sp, stack_offset]
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
            println!("Moving return value from {} to {}", get_return_value_registers()[0], to);
            self.move_to_reg(ops, lp, get_return_value_registers()[0].to_const_or_reg(), to);
        }

        for reg in active_regs.iter() {
            match reg {
                Register::GPR(r) => {
                    dynasm!(ops
                        ; ldr X(*r as u32), [sp, stack_offsets[reg]]
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
