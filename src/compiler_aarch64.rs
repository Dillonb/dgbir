use crate::{
    abi::{get_return_value_registers, get_scratch_registers},
    compiler::{Compiler, ConstOrReg, LiteralPool},
    ir::{BlockReference, CompareType, Constant, DataType, IRFunction},
    reg_pool::{register_type, RegPool},
    register_allocator::{alloc_for, Register, RegisterAllocations},
};
use dynasmrt::{aarch64::Aarch64Relocation, dynasm, AssemblyOffset, DynasmApi, DynasmLabelApi};

type Ops = dynasmrt::Assembler<Aarch64Relocation>;

fn load_64_bit_constant(ops: &mut Ops, lp: &mut LiteralPool, reg: u32, value: u64) {
    if value <= 0xFFFF {
        dynasm!(ops
            ; movz X(reg), value as u32
        );
    } else {
        let literal = Aarch64Compiler::add_literal(ops, lp, Constant::U64(value));
        dynasm!(ops
            ; ldr X(reg), =>literal
        );
    }
}

fn load_32_bit_constant(ops: &mut Ops, lp: &mut LiteralPool, reg: u32, value: u32) {
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

pub struct Aarch64Compiler<'a> {
    scratch_regs: RegPool,
    func: &'a IRFunction,
    allocations: RegisterAllocations,
    entrypoint: dynasmrt::AssemblyOffset,
    block_labels: Vec<dynasmrt::DynamicLabel>,
}

impl<'a> Compiler<'a, Ops> for Aarch64Compiler<'a> {
    fn new_dynamic_label(ops: &mut Ops) -> dynasmrt::DynamicLabel {
        ops.new_dynamic_label()
    }

    fn new(ops: &mut Ops, func: &'a mut IRFunction) -> Self {
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
        }
    }

    fn prologue(&self, ops: &mut Ops) {
        // Setup stack
        if self.get_func().stack_bytes_used > 0 {
            dynasm!(ops
                ; sub sp, sp, self.get_func().stack_bytes_used.try_into().unwrap()
            )
        }

        // Save callee-saved registers to stack
        for (reg, stack_location) in &self.allocations.callee_saved {
            match reg {
                Register::GPR(r) => {
                    assert_eq!(reg.size(), 8);
                    dynasm!(ops
                        ; str X(*r as u32), [sp, self.get_func().get_stack_offset_for_location(*stack_location as u64, DataType::U64) as u32]
                    )
                }
                Register::SIMD(r) => {
                    assert_eq!(reg.size(), 16);
                    dynasm!(ops
                        ; str Q(*r as u32), [sp, self.get_func().get_stack_offset_for_location(*stack_location as u64, DataType::U128)]
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
            (ConstOrReg::U64(_), Register::GPR(_)) => todo!("Moving {:?} to {}", from, to),
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

    fn get_func(&self) -> &IRFunction {
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
                        ; ldr Q(r as u32), [sp, self.get_func().get_stack_offset_for_location(*stack_location as u64, DataType::U128)]
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
        match (tp, r_out, a, b) {
            (DataType::U32, Register::GPR(r_out), ConstOrReg::U32(c1), ConstOrReg::U32(c2)) => {
                load_32_bit_constant(ops, lp, r_out as u32, c1 + c2);
            }
            (DataType::U32, Register::GPR(r_out), ConstOrReg::GPR(r), ConstOrReg::U32(c)) => {
                if c < 4096 {
                    dynasm!(ops
                        ; add WSP(r_out as u32), WSP(r), c
                    )
                } else {
                    let r_temp = self.scratch_regs.borrow::<register_type::GPR>();
                    let literal = Self::add_literal(ops, lp, Constant::U32(c));
                    dynasm!(ops
                        ; ldr W(r_temp.r()), =>literal
                        ; add W(r_out as u32), W(r), W(r_temp.r())
                    )
                }
            }
            (DataType::U32, Register::GPR(r_out), ConstOrReg::GPR(r1), ConstOrReg::GPR(r2)) => {
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

    fn compare(&self, ops: &mut Ops, r_out: usize, a: ConstOrReg, cmp_type: CompareType, b: ConstOrReg) {
        match (a, b) {
            (ConstOrReg::GPR(r1), ConstOrReg::GPR(r2)) => {
                dynasm!(ops
                    ; cmp X(r1 as u32), X(r2 as u32)
                )
            }
            (ConstOrReg::GPR(r1), ConstOrReg::U32(c2)) => {
                if c2 < 4096 {
                    dynasm!(ops
                        ; cmp XSP(r1 as u32), c2
                    )
                } else {
                    todo!("Too big a constant here, load it to a temp and compare")
                }
            }
            _ => todo!("Unsupported Compare operation: {:?} = {:?} {:?} {:?}", r_out, a, cmp_type, b),
        }

        match cmp_type {
            CompareType::LessThanUnsigned => {
                dynasm!(ops
                    ; cset W(r_out as u32), lo // unsigned "lower"
                )
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

                load_64_bit_constant(ops, lp, r_address.r(), ptr + offset);
                load_32_bit_constant(ops, lp, r_value.r(), value);
                dynasm!(ops
                    ; str W(r_value.r()), [X(r_address.r())]
                );
            }
            (ConstOrReg::U64(ptr), ConstOrReg::GPR(r_value), DataType::U32) => {
                let r_address = self.scratch_regs.borrow::<register_type::GPR>();
                load_64_bit_constant(ops, lp, r_address.r(), ptr + offset);
                dynasm!(ops
                    ; str W(r_value as u32), [X(r_address.r())]
                );
            }
            (ConstOrReg::GPR(_), ConstOrReg::U32(_), DataType::U32) => todo!(),
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
            _ => todo!("Unsupported WritePtr operation: {:?} = {:?} with type {}", ptr, value, data_type),
        }
    }

    fn spill_to_stack(&self, ops: &mut Ops, to_spill: ConstOrReg, stack_offset: ConstOrReg, tp: DataType) {
        match (&to_spill, &stack_offset, tp) {
            (ConstOrReg::GPR(r), ConstOrReg::U64(offset), DataType::U32) => {
                dynasm!(ops
                    ; str W(*r), [sp, self.func.get_stack_offset_for_location(*offset, DataType::U32) as u32]
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
            (Register::GPR(r_out), ConstOrReg::U64(offset), DataType::U32) => {
                dynasm!(ops
                    ; ldr W(r_out as u32), [sp, self.func.get_stack_offset_for_location(*offset, DataType::U32) as u32]
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
        } else if let Some(r_amount) = amount.to_reg() {
            todo!("LeftShift with register amount: {:?} << {:?}", n, r_amount);
        }
    }

    fn right_shift(&self, ops: &mut Ops, r_out: usize, n: ConstOrReg, amount: ConstOrReg, tp: DataType) {
        if let Some(amount) = amount.to_u64_const() {
            let amount = amount as u32;
            match (tp, n) {
                (DataType::U8, ConstOrReg::GPR(r_n)) => {
                    dynasm!(ops
                        ; lsr W(r_out as u32), W(r_n as u32), amount & 0b111
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
                        ; lsr W(r_out as u32), W(r_n as u32), amount & 0b1111
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
        } else if let Some(r_amount) = amount.to_reg() {
            todo!("RightShift with register amount: {:?} >> {:?}", n, r_amount);
        }
    }

    fn convert(&self, ops: &mut Ops, r_out: Register, input: ConstOrReg, from_tp: DataType, to_tp: DataType) {
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

    fn call(
        &self,
        _ops: &mut Ops,
        _lp: &mut LiteralPool,
        _address: ConstOrReg,
        _return_tp: Option<DataType>,
        _args: Vec<ConstOrReg>,
    ) {
        todo!()
    }
}
