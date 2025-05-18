use std::collections::HashMap;

use crate::{
    compiler::{move_regs_multi, Compiler, ConstOrReg},
    ir::{BlockReference, CompareType, Constant, DataType, IRFunction, InputSlot, OutputSlot},
    reg_pool::{register_type, RegPool},
    register_allocator::{alloc_for, get_scratch_registers, Register, RegisterAllocations, Value},
};
use dynasmrt::{aarch64::Aarch64Relocation, dynasm, AssemblyOffset, DynasmApi, DynasmLabelApi};

type Ops = dynasmrt::Assembler<Aarch64Relocation>;

// TODO: should I do a literal pool instead?
fn load_64_bit_constant(ops: &mut dynasmrt::aarch64::Assembler, reg: u32, value: u64) {
    if value > 0x0000_FFFF_FFFF_FFFF {
        dynasm!(ops
            ; movk X(reg), ((value >> 48) & 0xFFFF) as u32, lsl #48
            ; movk X(reg), ((value >> 32) & 0xFFFF) as u32, lsl #32
            ; movk X(reg), ((value >> 16) & 0xFFFF) as u32, lsl #16
            ; movk X(reg), (value & 0xFFFF) as u32
        );
    } else if value > 0x0000_0000_FFFF_FFFF {
        dynasm!(ops
            ; movz X(reg), ((value >> 32) & 0xFFFF) as u32, lsl #32
            ; movk X(reg), ((value >> 16) & 0xFFFF) as u32, lsl #16
            ; movk X(reg), (value & 0xFFFF) as u32
        );
    } else if value > 0x0000_0000_0000_FFFF {
        dynasm!(ops
            ; movz X(reg), ((value >> 16) & 0xFFFF) as u32, lsl #16
            ; movk X(reg), (value & 0xFFFF) as u32
        );
    } else {
        dynasm!(ops
            ; movz X(reg), value as u32
        );
    }
}

fn load_32_bit_constant(ops: &mut dynasmrt::aarch64::Assembler, reg: u32, value: u32) {
    if value > 0x0000_FFFF {
        dynasm!(ops
            ; movz X(reg), (value >> 16) & 0xFFFF, lsl #16
            ; movk X(reg), value & 0xFFFF
        );
    } else {
        dynasm!(ops
            ; movz X(reg), value
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

        // Setup stack
        dynasm!(ops
            ; .arch aarch64
        );
        if stack_bytes_used > 0 {
            dynasm!(ops
                ; sub sp, sp, stack_bytes_used.try_into().unwrap()
            )
        }

        Aarch64Compiler {
            entrypoint,
            scratch_regs: RegPool::new(get_scratch_registers()),
            func,
            allocations,
            block_labels,
        }
    }

    fn prologue(&self, ops: &mut Ops) {
        // Save callee-saved registers to stack
        for (reg, stack_location) in &self.allocations.callee_saved {
            match reg {
                Register::GPR(r) => {
                    assert_eq!(reg.size(), 8);
                    dynasm!(ops
                        ; str X(*r as u32), [sp, self.func.get_stack_offset_for_location(*stack_location as u64, DataType::U64)]
                    )
                }
            }
        }
    }

    fn epilogue(&self, _ops: &mut Ops) {
        println!("Epilogue: emitting nothing");
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

    fn call_block(&self, ops: &mut Ops, target: &BlockReference) {
        let moves = target
            .arguments
            .iter()
            .enumerate()
            .map(|(input_index, arg)| {
                let data_type = self.func.blocks[target.block_index].inputs[input_index];

                let in_block_value = Value::BlockInput {
                    block_index: target.block_index,
                    input_index,
                    data_type,
                };

                let block_arg_reg = self.allocations.get(&in_block_value).unwrap();
                (self.to_imm_or_reg(&arg), block_arg_reg)
            })
            .collect::<HashMap<_, _>>();

        if moves.len() > 0 {
            move_regs_multi(moves, |from, to| {
                match (from, to) {
                    (ConstOrReg::U32(c), Register::GPR(r_to)) => {
                        println!("\tMoving constant {} to register {}", c, r_to);
                        load_32_bit_constant(ops, r_to as u32, c);
                        // It was a constant, so no need to remove the source
                    }
                    (ConstOrReg::U64(_), Register::GPR(_)) => todo!("Moving {:?} to {}", from, to),
                    (ConstOrReg::GPR(r_from), Register::GPR(r_to)) => {
                        dynasm!(ops
                            ; mov X(r_to as u32), X(r_from as u32)
                        );
                    }
                }

            });
        }

        // TODO: figure out when we can elide this jump. If it's a jmp instruction to the next block,
        // we definitely can, but it gets more complicated when it's a branch instruction. Maybe the
        // branch instruction should detect if one of the targets is the next block and always put that
        // second. Then we could always elide this jump here.
        // if target.block_index != from_block_index + 1 {
        let target_label = self.block_labels[target.block_index];
        dynasm!(ops
            ; b =>target_label
        )
        // }
    }

    fn branch(&self, ops: &mut Ops, cond: &InputSlot, if_true: &BlockReference, if_false: &BlockReference) {
        let cond = self.to_imm_or_reg(&cond);

        match cond {
            ConstOrReg::GPR(c) => {
                dynasm!(ops
                    ; cbz W(c), >if_false
                );
            }
            _ => todo!("Unsupported branch condition: {:?}", cond),
        }
        self.call_block(ops, if_true);
        dynasm!(ops
            ; if_false:
        );
        self.call_block(ops, if_false);
    }

    fn ret(&self, ops: &mut Ops, value: &Option<InputSlot>) {
        if value.is_none() {
            println!("WARNING: returning values not supported yet")
        }

        // Pop callee-saved regs from stack
        // TODO: move this to the epilogue and emit a jmp to the end of the function here (to make
        // multiple returns more efficient)
        for (reg, stack_location) in &self.allocations.callee_saved {
            match *reg {
                Register::GPR(r) => {
                    assert_eq!(reg.size(), 8);
                    dynasm!(ops
                        ; ldr X(r as u32), [sp, self.func.get_stack_offset_for_location(*stack_location as u64, DataType::U64)]
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

    fn add(
        &self,
        ops: &mut Ops,
        inputs: &Vec<InputSlot>,
        outputs: &Vec<OutputSlot>,
        output_regs: Vec<Option<Register>>,
    ) {
        assert_eq!(inputs.len(), 2);
        assert_eq!(outputs.len(), 1);
        assert_eq!(output_regs.len(), 1);

        let a = self.to_imm_or_reg(&inputs[0]);
        let b = self.to_imm_or_reg(&inputs[1]);

        let tp = outputs[0].tp;

        let r_out = output_regs[0].unwrap();

        match (tp, r_out, a, b) {
            (DataType::U32, Register::GPR(r_out), ConstOrReg::U32(c1), ConstOrReg::U32(c2)) => {
                load_32_bit_constant(ops, r_out as u32, c1 + c2);
            }
            (DataType::U32, Register::GPR(r_out), ConstOrReg::GPR(r), ConstOrReg::U32(c)) => {
                if c < 4096 {
                    dynasm!(ops
                        ; add WSP(r_out as u32), WSP(r), c
                    )
                } else {
                    let r_temp = self.scratch_regs.borrow::<register_type::GPR>();
                    load_32_bit_constant(ops, r_temp.r(), c);
                    dynasm!(ops
                        ; add W(r_out as u32), W(r), W(r_temp.r())
                    )
                }
            }
            (DataType::U32, Register::GPR(r_out), ConstOrReg::GPR(r1), ConstOrReg::GPR(r2)) => {
                dynasm!(ops
                    ; add W(r_out as u32), W(r1), W(r2)
                )
            }
            _ => todo!("Unsupported Add operation: {:?} + {:?} with type {:?}", a, b, tp),
        }
    }

    fn compare(&self, ops: &mut Ops, inputs: &Vec<InputSlot>, output_regs: Vec<Option<Register>>) {
        let a = self.to_imm_or_reg(&inputs[0]);
        let b = self.to_imm_or_reg(&inputs[2]);

        let r_out = output_regs[0].unwrap();

        if let InputSlot::Constant(Constant::CompareType(compare_type)) = inputs[1] {
            // TODO: redo this to do the cmp and the cset in separate match statements to reduce
            // duplication
            match (r_out, a, compare_type, b) {
                (Register::GPR(r_out), ConstOrReg::GPR(r1), CompareType::LessThanUnsigned, ConstOrReg::GPR(r2)) => {
                    dynasm!(ops
                        ; cmp X(r1 as u32), X(r2 as u32)
                        ; cset W(r_out as u32), lo // unsigned "lower"
                    )
                }
                (Register::GPR(r_out), ConstOrReg::GPR(r1), CompareType::LessThanUnsigned, ConstOrReg::U32(c2)) => {
                    if c2 < 4096 {
                        dynasm!(ops
                            ; cmp XSP(r1 as u32), c2
                        )
                    } else {
                        todo!("Too big a constant here, load it to a temp and compare")
                    }
                    dynasm!(ops
                        ; cset W(r_out as u32), lo // unsigned "lower"
                    )
                }
                _ => todo!("Unsupported Compare operation: {:?} {:?} {:?}", a, compare_type, b),
            }
        } else {
            panic!("Expected a compare type constant as the second input to Compare");
        }
    }

    fn load_ptr(
        &self,
        _ops: &mut Ops,
        _inputs: &Vec<InputSlot>,
        _outputs: &Vec<OutputSlot>,
        _output_regs: Vec<Option<Register>>,
    ) {
        todo!("load_ptr")
    }

    fn write_ptr(&self, ops: &mut Ops, inputs: &Vec<InputSlot>) {
        assert_eq!(inputs.len(), 3);
        // ptr, value, type
        let ptr = self.to_imm_or_reg(&inputs[0]);
        let value = self.to_imm_or_reg(&inputs[1]);
        if let InputSlot::Constant(Constant::DataType(tp)) = inputs[2] {
            match (&ptr, &value, tp) {
                (ConstOrReg::U64(ptr), ConstOrReg::U32(value), DataType::U32) => {
                    let r_address = self.scratch_regs.borrow::<register_type::GPR>();
                    let r_value = self.scratch_regs.borrow::<register_type::GPR>();

                    load_64_bit_constant(ops, r_address.r(), *ptr);
                    load_32_bit_constant(ops, r_value.r(), *value);
                    dynasm!(ops
                        ; str W(r_value.r()), [X(r_address.r())]
                    );
                }
                (ConstOrReg::U64(ptr), ConstOrReg::GPR(value), DataType::U32) => {
                    let r_address = self.scratch_regs.borrow::<register_type::GPR>();
                    load_64_bit_constant(ops, r_address.r(), *ptr);
                    dynasm!(ops
                        ; str W((*value) as u32), [X(r_address.r())]
                    )
                }
                (ConstOrReg::GPR(_), ConstOrReg::U32(_), DataType::U32) => todo!(),
                (ConstOrReg::GPR(_), ConstOrReg::GPR(_), DataType::U32) => todo!(),
                _ => todo!("Unsupported WritePtr operation: {:?} = {:?} with type {}", ptr, value, tp),
            }
        } else {
            panic!("Expected a datatype constant as the third input to WritePtr");
        }
    }

    fn spill_to_stack(&self, ops: &mut Ops, inputs: &Vec<InputSlot>) {
        let to_spill = self.to_imm_or_reg(&inputs[0]);
        let stack_offset = self.to_imm_or_reg(&inputs[1]);
        if let InputSlot::Constant(Constant::DataType(tp)) = inputs[2] {
            match (&to_spill, &stack_offset, tp) {
                (ConstOrReg::GPR(r), ConstOrReg::U64(offset), DataType::U32) => {
                    dynasm!(ops
                        ; str W(*r), [sp, self.func.get_stack_offset_for_location(*offset, DataType::U32)]
                    )
                }
                _ => todo!(
                    "Unsupported SpillToStack operation: {:?} to offset {:?} with datatype {}",
                    to_spill,
                    stack_offset,
                    tp
                ),
            }
        } else {
            panic!("Expected a datatype constant as the third input to SpillToStack");
        }
    }

    fn load_from_stack(
        &self,
        ops: &mut Ops,
        inputs: &Vec<InputSlot>,
        outputs: &Vec<OutputSlot>,
        output_regs: Vec<Option<Register>>,
    ) {
        let stack_offset = self.to_imm_or_reg(&inputs[0]);

        let r_out = output_regs[0].unwrap();
        let tp = outputs[0].tp;

        match (r_out, &stack_offset, tp) {
            (Register::GPR(r_out), ConstOrReg::U64(offset), DataType::U32) => {
                dynasm!(ops
                    ; ldr W(r_out as u32), [sp, self.func.get_stack_offset_for_location(*offset, DataType::U32)]
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
}

