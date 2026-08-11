use std::{collections::BTreeMap, iter, marker::PhantomData};

use crate::{
    abi::{assign_argument_registers, get_return_value_registers, get_scratch_registers, reg_constants},
    compiler::{lane_swizzle_byte_mask, Compiler, ConstOrReg, GenericAssembler, LiteralPool, MaterializedGpr},
    ir::{BlockReference, CompareType, Constant, DataType, IRFunctionInternal},
    reg_pool::{register_type, BorrowedReg, RegPool},
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

/// Detect whether an immediate can be encoded in `fmov`
fn f32_fits_fmov_immediate(value: f32) -> bool {
    let bits = value.to_bits();
    let b = (bits >> 29) & 1;
    let expected_exponent_rest = if b == 1 { 0b11111 } else { 0 };
    bits & 0x7FFFF == 0 && (bits >> 30) & 1 == b ^ 1 && (bits >> 25) & 0b11111 == expected_exponent_rest
}

/// Shuffle indices for [`variable_byte_shift_128`]. `tbl` zeroes any lane whose index byte is out
/// of range, so the 0x80 padding either side of 0x00..0x0F supplies the zero fill.
#[rustfmt::skip]
static BYTE_SHIFT_SHUFFLES: [u8; 48] = [
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
    0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
];

/// Shifts the 128 bit value in `r_out` by a byte granular amount only known at runtime.
///
/// The 16 byte window of [`BYTE_SHIFT_SHUFFLES`] at `16 - d` is the shuffle mask for a left shift
/// of `d` bytes, and the window at `16 + d` is the mask for a right shift of `d` bytes.
///
/// `r_amount` is in bits to match the constant amount path; sub-byte amounts are not supported
/// for 128 bit shifts.
fn variable_byte_shift_128<Ops: GenericAssembler<Aarch64Relocation>>(
    ops: &mut Ops,
    lp: &mut LiteralPool,
    scratch_regs: &RegPool,
    r_out: RegisterIndex,
    r_bytes: RegisterIndex,
    left: bool,
) {
    let r_shift = scratch_regs.borrow::<register_type::GPR>();
    let r_window = scratch_regs.borrow::<register_type::GPR>();
    let r_table = scratch_regs.borrow::<register_type::GPR>();
    let r_index = scratch_regs.borrow::<register_type::SIMD>();

    dynasm!(ops
        ; mov W(r_shift.r()), W(r_bytes)
        // Clamp to 16, which already shifts everything out, and keeps the window in the table.
        ; movz W(r_window.r()), 16
        ; cmp W(r_shift.r()), W(r_window.r())
        ; csel W(r_shift.r()), W(r_shift.r()), W(r_window.r()), lo
    );
    // r_window holds 16, so it becomes the window index with a single add or subtract.
    if left {
        dynasm!(ops
            ; sub W(r_window.r()), W(r_window.r()), W(r_shift.r())
        );
    } else {
        dynasm!(ops
            ; add W(r_window.r()), W(r_window.r()), W(r_shift.r())
        );
    }
    load_64_bit_constant(ops, lp, r_table.r(), BYTE_SHIFT_SHUFFLES.as_ptr() as u64);
    dynasm!(ops
        ; add X(r_table.r()), X(r_table.r()), X(r_window.r())
        ; ldr Q(r_index.r()), [X(r_table.r())]
        ; tbl V(r_out).B16, {V(r_out).B16 * 1}, V(r_index.r()).B16
    );
}

/// Shifts the 128 bit value in `r_out` by a byte granular amount. Both directions are a shuffle of
/// the 16 lanes, with zeroes shifted in.
fn byte_shift_128<Ops: GenericAssembler<Aarch64Relocation>>(
    ops: &mut Ops,
    lp: &mut LiteralPool,
    scratch_regs: &RegPool,
    r_out: RegisterIndex,
    bytes: ConstOrReg,
    left: bool,
) {
    if let Some(bytes) = bytes.to_u64_const() {
        let bytes = bytes as u32;
        if bytes == 0 {
            return;
        }
        if bytes >= 16 {
            dynasm!(ops
                ; eor V(r_out).B16, V(r_out).B16, V(r_out).B16
            );
            return;
        }
        let r_zero = scratch_regs.borrow::<register_type::SIMD>();
        dynasm!(ops
            ; eor V(r_zero.r()).B16, V(r_zero.r()).B16, V(r_zero.r()).B16
        );
        if left {
            // ext takes a 16 byte window of r_zero:r_out starting at 16 - bytes, which is the value
            // with `bytes` zero bytes shifted into the bottom.
            let index = 16 - bytes;
            dynasm!(ops
                ; ext V(r_out).B16, V(r_zero.r()).B16, V(r_out).B16, index
            );
        } else {
            dynasm!(ops
                ; ext V(r_out).B16, V(r_out).B16, V(r_zero.r()).B16, bytes
            );
        }
    } else if let Some(Register::GPR(r_bytes)) = bytes.to_reg() {
        variable_byte_shift_128(ops, lp, scratch_regs, r_out, r_bytes, left);
    } else {
        panic!("128 bit shift amount must be a constant or a GPR, got: {:?}", bytes);
    }
}

/// `ldr`/`str` of a Q register can only encode a 16 byte aligned immediate offset, so any other
/// offset has to be folded into the address register instead.
fn q_offset_is_encodable(offset: u64) -> bool {
    offset % 16 == 0 && offset <= 0xFFF * 16
}

/// Computes `r_ptr + offset` into a scratch register, for the 128 bit loads and stores that can't
/// always encode the offset themselves.
fn address_in_gpr<Ops: GenericAssembler<Aarch64Relocation>>(
    ops: &mut Ops,
    lp: &mut LiteralPool,
    scratch_regs: &RegPool,
    r_ptr: RegisterIndex,
    offset: u64,
) -> BorrowedReg<register_type::GPR> {
    let r_addr = scratch_regs.borrow::<register_type::GPR>();
    load_64_bit_constant(ops, lp, r_addr.r(), offset);
    dynasm!(ops
        ; add X(r_addr.r()), X(r_ptr), X(r_addr.r())
    );
    r_addr
}

/// Widens an already materialized value into a full 64 bit register, ready to be compared.
fn extend_as_comparable_gpr<Ops: GenericAssembler<Aarch64Relocation>>(
    ops: &mut Ops,
    scratch_regs: &RegPool,
    r: RegisterIndex,
    tp: DataType,
) -> MaterializedGpr {
    if matches!(tp, DataType::U64 | DataType::S64 | DataType::Ptr) {
        return MaterializedGpr::AlreadyGPR(r);
    }

    let out = scratch_regs.borrow::<register_type::GPR>();
    // Writing to a W register zero extends into the full X register, so the unsigned cases don't
    // need to name the 64 bit form.
    match tp {
        DataType::Bool | DataType::U8 => dynasm!(ops
            ; uxtb W(out.r()), W(r)
        ),
        DataType::S8 => dynasm!(ops
            ; sxtb X(out.r()), W(r)
        ),
        DataType::U16 => dynasm!(ops
            ; uxth W(out.r()), W(r)
        ),
        DataType::S16 => dynasm!(ops
            ; sxth X(out.r()), W(r)
        ),
        DataType::U32 => dynasm!(ops
            ; mov W(out.r()), W(r)
        ),
        DataType::S32 => dynasm!(ops
            ; sxtw X(out.r()), W(r)
        ),
        _ => todo!("Cannot widen data type {} for comparison", tp),
    }
    MaterializedGpr::TemporaryGPR(out)
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
                if f32_fits_fmov_immediate(*value) {
                    dynasm!(ops
                        ; fmov S(r_to), *value
                    )
                } else {
                    let literal = Self::add_literal(ops, lp, Constant::F32(value));
                    dynasm!(ops
                        ; ldr S(r_to), =>literal
                    )
                }
            }
            (c, Register::SIMD(r_to)) if c.is_const() => {
                // Zero extended into the low 64 bits, matching what a move from a GPR does.
                let r_temp = self.scratch_regs.borrow::<register_type::GPR>();
                load_64_bit_constant(ops, lp, r_temp.r(), c.to_u64_const().unwrap());
                dynasm!(ops
                    ; fmov D(r_to), X(r_temp.r())
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
                Constant::U128(c) => {
                    dynasm!(ops
                        ; =>label
                        ; .u64 c as u64
                        ; .u64 (c >> 64) as u64
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
            (DataType::U16 | DataType::S16, Register::GPR(r_out)) => {
                let a = self.materialize_as_gpr(ops, lp, a);
                let b = self.materialize_as_gpr(ops, lp, b);
                dynasm!(ops
                    ; add W(r_out), W(a.r()), W(b.r())
                );
            }
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
                (false, CompareType::GreaterThan) => {
                    dynasm!(ops
                        ; cset W(r_out), hi // unsigned "higher"
                    )
                }
                (false, CompareType::LessThanOrEqual) => {
                    dynasm!(ops
                        ; cset W(r_out), ls // unsigned "lower or same"
                    )
                }
                (false, CompareType::GreaterThanOrEqual) => {
                    dynasm!(ops
                        ; cset W(r_out), hs // unsigned "higher or same"
                    )
                }
            }
        }

        let signed = data_type.is_signed();

        if data_type.is_integer() {
            let a = self.materialize_as_gpr(ops, lp, a);
            let a = extend_as_comparable_gpr(ops, &self.scratch_regs, a.r(), data_type);
            let b = self.materialize_as_gpr(ops, lp, b);
            let b = extend_as_comparable_gpr(ops, &self.scratch_regs, b.r(), data_type);
            dynasm!(ops
                ; cmp X(a.r()), X(b.r())
            );
            set_reg_by_flags(ops, signed, cmp_type, r_out);
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
            (Register::SIMD(r_out), ptr, tp) if tp.size() == 16 => {
                let r_ptr = self.materialize_as_gpr(ops, lp, ptr);
                if q_offset_is_encodable(offset) {
                    dynasm!(ops
                        ; ldr Q(r_out), [X(r_ptr.r()), offset as u32]
                    );
                } else {
                    let r_addr = address_in_gpr(ops, lp, &self.scratch_regs, r_ptr.r(), offset);
                    dynasm!(ops
                        ; ldr Q(r_out), [X(r_addr.r())]
                    );
                }
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
            // Store directly from SIMD instead of falling through to the generic materialize_as_gpr
            // path below
            (ConstOrReg::GPR(r_ptr), ConstOrReg::SIMD(r_value), DataType::U32 | DataType::S32 | DataType::F32) => {
                dynasm!(ops
                    ; str S(r_value), [X(r_ptr), offset as u32]
                );
            }
            (ConstOrReg::GPR(r_ptr), ConstOrReg::SIMD(r_value), DataType::U64 | DataType::S64 | DataType::F64) => {
                dynasm!(ops
                    ; str D(r_value), [X(r_ptr), offset as u32]
                );
            }
            (ptr, value, DataType::U8 | DataType::S8 | DataType::Bool) => {
                let address = self.materialize_as_gpr(ops, lp, ptr);
                let value = self.materialize_as_gpr(ops, lp, value);
                dynasm!(ops
                    ; strb W(value.r()), [X(address.r()), offset as u32]
                )
            }
            (ptr, value, DataType::U16 | DataType::S16) => {
                let address = self.materialize_as_gpr(ops, lp, ptr);
                let value = self.materialize_as_gpr(ops, lp, value);
                dynasm!(ops
                    ; strh W(value.r()), [X(address.r()), offset as u32]
                )
            }
            (ptr, value, DataType::U32 | DataType::S32 | DataType::F32) => {
                let address = self.materialize_as_gpr(ops, lp, ptr);
                let value = self.materialize_as_gpr(ops, lp, value);
                dynasm!(ops
                    ; str W(value.r()), [X(address.r()), offset as u32]
                )
            }
            (ptr, value, DataType::Ptr | DataType::U64 | DataType::S64 | DataType::F64) => {
                let address = self.materialize_as_gpr(ops, lp, ptr);
                let value = self.materialize_as_gpr(ops, lp, value);
                dynasm!(ops
                    ; str X(value.r()), [X(address.r()), offset as u32]
                )
            }
            (ptr, value, tp) if tp.size() == 16 => {
                let value = self.materialize_as_simd(ops, lp, value);
                let r_ptr = self.materialize_as_gpr(ops, lp, ptr);
                if q_offset_is_encodable(offset) {
                    dynasm!(ops
                        ; str Q(value.r()), [X(r_ptr.r()), offset as u32]
                    );
                } else {
                    let r_addr = address_in_gpr(ops, lp, &self.scratch_regs, r_ptr.r(), offset);
                    dynasm!(ops
                        ; str Q(value.r()), [X(r_addr.r())]
                    );
                }
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
            (ConstOrReg::SIMD(r), ConstOrReg::U64(offset), tp) if tp.size() == 16 => {
                dynasm!(ops
                    ; str Q(*r), [sp, self.func.get_stack_offset_for_location(*offset, DataType::U128)]
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
            (Register::SIMD(r_out), ConstOrReg::U64(offset), tp) if tp.size() == 16 => {
                dynasm!(ops
                    ; ldr Q(r_out), [sp, self.func.get_stack_offset_for_location(*offset, DataType::U128)]
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
        r_out: Register,
        n: ConstOrReg,
        amount: ConstOrReg,
        tp: DataType,
    ) -> () {
        let r_out = r_out.expect_gpr();
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
        r_out: Register,
        n: ConstOrReg,
        amount: ConstOrReg,
        tp: DataType,
    ) {
        let r_out = r_out.expect_gpr();
        if let Some(amount) = amount.to_u64_const() {
            let amount = amount as u32;
            let orig_n = n;
            let n = self.materialize_as_gpr(ops, lp, n);
            match tp {
                DataType::U8 => {
                    dynasm!(ops
                        ; and WSP(r_out), W(n.r()), 0xFF
                        ; lsr W(r_out), W(r_out), amount & 0b111
                    );
                }
                DataType::S8 => {
                    // Shift left to put the sign bit in the 32 bit sign bit position, then shift
                    // right.
                    dynasm!(ops
                        ; lsl W(r_out), W(n.r()), 24
                        ; asr W(r_out), W(r_out), (amount & 0b111) + 24
                        ; and WSP(r_out), W(r_out), 0xFF // Mask to 8 bits
                    );
                }
                DataType::U16 => {
                    dynasm!(ops
                        ; and WSP(r_out), W(n.r()), 0xFFFF
                        ; lsr W(r_out), W(r_out), amount & 0b1111
                    );
                }
                DataType::S16 => {
                    // Shift left to put the sign bit in the 32 bit sign bit position, then shift
                    // right.
                    dynasm!(ops
                        ; lsl W(r_out), W(n.r()), 16
                        ; asr W(r_out), W(r_out), (amount & 0b1111) + 16
                        ; and WSP(r_out), W(r_out), 0xFFFF // Mask to 16 bits
                    );
                }
                DataType::U32 => {
                    dynasm!(ops
                        ; lsr W(r_out), W(n.r()), amount & 0b11111
                    );
                }
                DataType::S32 => {
                    dynasm!(ops
                        ; asr W(r_out), W(n.r()), amount & 0b11111
                    );
                }
                DataType::U64 => {
                    dynasm!(ops
                        ; lsr X(r_out), X(n.r()), amount & 0b111111
                    );
                }
                DataType::S64 => {
                    dynasm!(ops
                        ; asr X(r_out), X(n.r()), amount & 0b111111
                    );
                }
                _ => todo!("Unsupported RightShift operation: {:?} >> {:?} with type {}", orig_n, amount, tp),
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

    fn vector_swizzle(
        &self,
        ops: &mut Ops,
        lp: &mut LiteralPool,
        r_out: Register,
        value: ConstOrReg,
        pattern: u64,
        tp: DataType,
    ) {
        let v = match tp {
            DataType::Vector(v) => v,
            _ => todo!("VectorSwizzle on non-vector type {}", tp),
        };
        let src = self.materialize_as_simd(ops, lp, value);
        let literal = Self::add_literal(ops, lp, Constant::U128(lane_swizzle_byte_mask(pattern, v)));
        let r_mask = self.scratch_regs.borrow::<register_type::SIMD>();
        dynasm!(ops
            ; ldr Q(r_mask.r()), =>literal
            ; tbl V(r_out.expect_simd()).B16, {V(src.r()).B16 * 1}, V(r_mask.r()).B16
        );
    }

    fn vector_left_shift_bytes(
        &self,
        ops: &mut Ops,
        lp: &mut LiteralPool,
        r_out: Register,
        n: ConstOrReg,
        bytes: ConstOrReg,
        _tp: DataType,
    ) {
        self.move_to_reg(ops, lp, n, r_out);
        byte_shift_128(ops, lp, &self.scratch_regs, r_out.expect_simd(), bytes, true);
    }

    fn vector_right_shift_bytes(
        &self,
        ops: &mut Ops,
        lp: &mut LiteralPool,
        r_out: Register,
        n: ConstOrReg,
        bytes: ConstOrReg,
        _tp: DataType,
    ) {
        self.move_to_reg(ops, lp, n, r_out);
        byte_shift_128(ops, lp, &self.scratch_regs, r_out.expect_simd(), bytes, false);
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
            (Register::GPR(r_out), DataType::S32, DataType::S8) => {
                let input = self.materialize_as_gpr(ops, lp, input);
                dynasm!(ops
                    ; sxtb W(r_out), W(input.r())
                );
            }
            (Register::GPR(r_out), DataType::S32, DataType::S16) => {
                let input = self.materialize_as_gpr(ops, lp, input);
                dynasm!(ops
                    ; sxth W(r_out), W(input.r())
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
            (Register::GPR(r_out), DataType::U32, DataType::S64) => {
                let input = self.materialize_as_gpr(ops, lp, input);
                dynasm!(ops
                    ; mov W(r_out), W(input.r())
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
            (tp, Register::SIMD(r_out)) if tp.size() == 16 => {
                let a = self.materialize_as_simd(ops, lp, a);
                let b = self.materialize_as_simd(ops, lp, b);
                dynasm!(ops
                    ; and V(r_out).B16, V(a.r()).B16, V(b.r()).B16
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
            (tp, Register::SIMD(r_out)) if tp.size() == 16 => {
                let a = self.materialize_as_simd(ops, lp, a);
                let b = self.materialize_as_simd(ops, lp, b);
                dynasm!(ops
                    ; orr V(r_out).B16, V(a.r()).B16, V(b.r()).B16
                );
            }
            _ => todo!("Unsupported OR operation: {:?} | {:?} with type {:?}", a, b, tp),
        }
    }

    fn not(&self, ops: &mut Ops, lp: &mut LiteralPool, tp: DataType, r_out: Register, a: ConstOrReg) {
        if tp.size() == 16 {
            let a = self.materialize_as_simd(ops, lp, a);
            dynasm!(ops
                ; mvn V(r_out.expect_simd()).B16, V(a.r()).B16
            );
            return;
        }
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
        let arg_regs = assign_argument_registers(&args);
        let moves = args
            .into_iter()
            .zip(arg_regs.into_iter())
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
            let from = *get_return_value_registers()
                .iter()
                .find(|r| r.is_simd() == to.is_simd())
                .unwrap();
            debug!("Moving return value from {} to {}", from, to);
            self.move_to_reg(ops, lp, from.to_const_or_reg(), to);
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
