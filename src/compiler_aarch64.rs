use std::{collections::HashMap, mem};

use crate::{disassembler::disassemble, ir::{Constant, DataType, IRFunction, InputSlot}, register_allocator::{alloc_for, Register, Value}};
use dynasmrt::{dynasm, DynasmApi, DynasmLabelApi};
use itertools::Either;

fn input_slot_to_imm_or_reg(s : InputSlot, func: &IRFunction, allocations: &HashMap<Value, Register>) -> Either<Constant, Register> {
    match s {
        InputSlot::InstructionOutput { .. } => Either::Right(allocations[&s.to_value(func).unwrap()]),
        InputSlot::BlockInput { .. } => Either::Right(allocations[&s.to_value(func).unwrap()]),
        InputSlot::Constant(constant) => Either::Left(constant),
    }
}

#[derive(Debug)]
enum ConstOrReg {
    U32(u32),
    U64(u64),
    GPR(u32),
}

fn input_slot_to_imm_or_reg_2(s : InputSlot, func: &IRFunction, allocations: &HashMap<Value, Register>) -> ConstOrReg {
    match s {
        InputSlot::InstructionOutput { .. } => {
            match allocations[&s.to_value(func).unwrap()] {
                Register::GPR(r) => ConstOrReg::GPR(r as u32),
            }
        },
        InputSlot::Constant(constant) => {
            match constant {
                Constant::U32(c) => ConstOrReg::U32(c as u32),
                Constant::U8(_) => todo!(),
                Constant::S8(_) => todo!(),
                Constant::U16(_) => todo!(),
                Constant::S16(_) => todo!(),
                Constant::S32(_) => todo!(),
                Constant::U64(c) => ConstOrReg::U64(c as u64),
                Constant::S64(_) => todo!(),
                Constant::F32(_) => todo!(),
                Constant::F64(_) => todo!(),
                Constant::Ptr(c) => ConstOrReg::U64(c as u64),
                Constant::Bool(_) => todo!(),
                Constant::DataType(_) => todo!(),
                Constant::CompareType(_) => todo!(),
            }
        },
        _ => todo!("Unsupported input slot type: {:?}", s),
    }
}

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

fn add_constants(a: Constant, b: Constant) -> u32 {
    match (a, b) {
        (Constant::U32(a), Constant::U32(b)) => a + b,
        _ => todo!("Adding constants other than u32")
    }
}

pub fn compile(func: &mut IRFunction) {
    let allocations = alloc_for(func); // TODO: this will eventually return some data

    let mut ops = dynasmrt::aarch64::Assembler::new().unwrap();

    // Stack bytes used: aligned to 16 bytes
    let misalignment = func.stack_bytes_used % 16;
    let correction = if misalignment == 0 { 0 } else { 16 - misalignment };
    let stack_bytes_used = func.stack_bytes_used + correction;
    println!("Function uses {} bytes of stack, misaligned by {}, corrected to {}", func.stack_bytes_used, misalignment, stack_bytes_used);
    func.stack_bytes_used = stack_bytes_used;

    // Entrypoint to the function - get the offset before appending anything
    let entrypoint = ops.offset();

    // Setup stack
    dynasm!(ops
        ; .arch aarch64
    );
    if stack_bytes_used > 0 {
        dynasm!(ops
            ; sub sp, sp, stack_bytes_used.try_into().unwrap()
        )
    }

    let block_labels = func.blocks.iter().map(|_| ops.new_dynamic_label()).collect::<Vec<_>>();

    for (block_index, block) in func.blocks.iter().enumerate() {
        dynasm!(ops
            ; =>block_labels[block_index]
        );

        for (_instruction_index_in_block, instruction_index) in block.instructions.iter().enumerate() {
            let instruction = &func.instructions[*instruction_index];

            match &instruction.instruction {
                crate::ir::Instruction::Instruction { tp, inputs, outputs } => {
                    match tp {
                        crate::ir::InstructionType::Add => {
                            assert_eq!(inputs.len(), 2);

                            let a = input_slot_to_imm_or_reg(inputs[0], func, &allocations);
                            let b = input_slot_to_imm_or_reg(inputs[1], func, &allocations);

                            let r_out = allocations[&Value::InstructionOutput {
                                instruction_index: *instruction_index,
                                output_index: 0,
                                block_index,
                                data_type: outputs[0].tp,
                            }];


                            match (r_out, a, b) {
                                (Register::GPR(r_out), Either::Left(c), Either::Left(c2)) => {
                                    dynasm!(ops
                                        ; mov X(r_out as u32), add_constants(c, c2).into()
                                    )
                                },
                                (Register::GPR(r_out), Either::Left(Constant::U32(c)), Either::Right(Register::GPR(r))) => {
                                    dynasm!(ops
                                        ; mov X(r_out as u32), c.into()
                                        ; add X(r_out as u32), X(r_out as u32), X(r as u32)
                                    )
                                },
                                (Register::GPR(r_out), Either::Right(Register::GPR(r)), Either::Left(Constant::U32(c))) => {
                                    dynasm!(ops
                                        ; mov X(r_out as u32), c.into()
                                        ; add X(r_out as u32), X(r_out as u32), X(r as u32)
                                    )
                                },
                                (Register::GPR(r_out), Either::Right(Register::GPR(r1)), Either::Right(Register::GPR(r2))) => {
                                    dynasm!(ops
                                        ; add X(r_out as u32), X(r1 as u32), X(r2 as u32)
                                    )
                                }
                                _ => todo!("Unsupported add operation: {:?} + {:?}", a, b),
                            }
                        },
                        crate::ir::InstructionType::Compare => todo!("compare"),
                        crate::ir::InstructionType::LoadPtr => todo!("load_ptr"),
                        crate::ir::InstructionType::WritePtr => {
                            assert_eq!(inputs.len(), 3);
                            // ptr, value, type
                            let ptr = input_slot_to_imm_or_reg_2(inputs[0], func, &allocations);
                            let value = input_slot_to_imm_or_reg_2(inputs[1], func, &allocations);
                            if let InputSlot::Constant(Constant::DataType(tp)) = inputs[2] {
                                match (&ptr, &value, tp) {
                                    (ConstOrReg::U64(ptr), ConstOrReg::U32(value), DataType::U32) => {
                                        println!("TODO: allocate these temporary registers in a real/safe way");
                                        load_64_bit_constant(&mut ops, 0, *ptr);
                                        load_32_bit_constant(&mut ops, 1, *value);
                                        dynasm!(ops
                                            ; str W(1), [X(0)]
                                        )
                                    },
                                    (ConstOrReg::U64(ptr), ConstOrReg::GPR(value), DataType::U32) => {
                                        println!("TODO: allocate these temporary registers in a real/safe way");
                                        load_64_bit_constant(&mut ops, 0, *ptr);
                                        dynasm!(ops
                                            ; str W((*value) as u32), [X(0)]
                                        )
                                    },
                                    (ConstOrReg::GPR(ptr), ConstOrReg::U32(value), DataType::U32) => todo!(),
                                    (ConstOrReg::GPR(ptr), ConstOrReg::GPR(value), DataType::U32) => todo!(),
                                    _ => todo!("Unsupported WritePtr operation: {:?} = {:?} with type {}", ptr, value, tp),
                                }
                            } else {
                                panic!("Expected a datatype constant as the third input to WritePtr");
                            }
                        },
                        crate::ir::InstructionType::SpillToStack => {
                            let to_spill = input_slot_to_imm_or_reg_2(inputs[0], func, &allocations);
                            let stack_offset = input_slot_to_imm_or_reg_2(inputs[1], func, &allocations);
                            if let InputSlot::Constant(Constant::DataType(tp)) = inputs[2] {
                                match (&to_spill, &stack_offset, tp) {
                                    (ConstOrReg::GPR(r), ConstOrReg::U64(offset), DataType::U32) => {
                                        dynasm!(ops
                                            ; str W(*r), [sp, func.get_stack_offset_for_location(*offset, DataType::U32)]
                                        )
                                    },
                                    _ => todo!("Unsupported SpillToStack operation: {:?} to offset {:?} with datatype {}", to_spill, stack_offset, tp),
                                }
                            } else {
                                panic!("Expected a datatype constant as the third input to SpillToStack");
                            }
                        },
                        crate::ir::InstructionType::LoadFromStack => {
                            let stack_offset = input_slot_to_imm_or_reg_2(inputs[0], func, &allocations);

                            let r_out = allocations[&Value::InstructionOutput {
                                instruction_index: *instruction_index,
                                output_index: 0,
                                block_index,
                                data_type: outputs[0].tp,
                            }];

                            if let InputSlot::Constant(Constant::DataType(tp)) = inputs[1] {
                                match (r_out, &stack_offset, tp) {
                                    (Register::GPR(r_out), ConstOrReg::U64(offset), DataType::U32) => {
                                        dynasm!(ops
                                            ; ldr W(r_out as u32), [sp, func.get_stack_offset_for_location(*offset, DataType::U32)]
                                        )
                                    },
                                    _ => todo!("Unsupported LoadFromStack operation: load {} from offset {:?} with datatype {}", r_out, stack_offset, tp),
                                }
                            }
                        },
                    }
                },
                crate::ir::Instruction::Branch { cond, if_true, if_false } => todo!("branch"),
                crate::ir::Instruction::Jump { target } => {
                    if target.arguments.len() > 0 {
                        todo!("Jump with arguments");
                    }

                    // Only need to emit a jump if the target is not the next block - otherwise
                    // it's just a fallthrough
                    if target.block_index != block_index + 1 {
                        let target_label = block_labels[target.block_index];
                        dynasm!(ops
                            ; b =>target_label
                        );
                    }
                },
                crate::ir::Instruction::Return { value } => println!("return"),
            }
        }
    }

    // Fix stack and return
    if stack_bytes_used > 0 {
        dynasm!(ops
            ; add sp, sp, stack_bytes_used.try_into().unwrap()
        );
    }
    dynasm!(ops
        ; ret
    );


    let code = ops.finalize().unwrap();
    let f: extern "C" fn(x: u64) -> u64 = unsafe { mem::transmute(code.ptr(entrypoint)) };

    println!("{}", disassemble(&code, f as u64));

    println!("Running:");
    f(1);
}
