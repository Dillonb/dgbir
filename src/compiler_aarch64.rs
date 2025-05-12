use std::{collections::{HashMap, HashSet}, mem};

use crate::{disassembler::disassemble, ir::{BlockReference, CompareType, Constant, DataType, IRFunction, InputSlot, OutputSlot}, register_allocator::{alloc_for, Register, Value}};
use dynasmrt::{dynasm, DynasmApi, DynasmLabelApi};
use itertools::Itertools;

#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
enum ConstOrReg {
    U32(u32),
    U64(u64),
    GPR(u32),
}

impl ConstOrReg {
    fn to_reg(&self) -> Option<Register> {
        match self {
            ConstOrReg::GPR(r) => Some(Register::GPR(*r as usize)),
            ConstOrReg::U32(_) => None,
            ConstOrReg::U64(_) => None,
        }
    }
}

impl Register {
    fn to_const_or_reg(&self) -> ConstOrReg {
        match self {
            Register::GPR(r) => ConstOrReg::GPR(*r as u32),
        }
    }
}

fn input_slot_to_imm_or_reg(s: &InputSlot, func: &IRFunction, allocations: &HashMap<Value, Register>) -> ConstOrReg {
    match *s {
        InputSlot::InstructionOutput { .. } | InputSlot::BlockInput { .. } => {
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
        // _ => todo!("Unsupported input slot type: {:?}", s),
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

fn compile_add(ops: &mut dynasmrt::aarch64::Assembler, func: &IRFunction, allocations: &HashMap<Value, Register>, inputs: &Vec<InputSlot>, outputs: &Vec<OutputSlot>, output_registers: Vec<Option<&Register>>) {
    assert_eq!(inputs.len(), 2);
    assert_eq!(outputs.len(), 1);
    assert_eq!(output_registers.len(), 1);

    let a = input_slot_to_imm_or_reg(&inputs[0], func, &allocations);
    let b = input_slot_to_imm_or_reg(&inputs[1], func, &allocations);

    let tp = outputs[0].tp;

    let r_out = *output_registers[0].unwrap();

    match (tp, r_out, a, b) {
        (DataType::U32, Register::GPR(r_out), ConstOrReg::U32(c1), ConstOrReg::U32(c2)) => {
            load_32_bit_constant(ops, r_out as u32, c1 + c2);
        },
        (DataType::U32, Register::GPR(r_out), ConstOrReg::GPR(r), ConstOrReg::U32(c)) => {
            if c < 4096 {
                dynasm!(ops
                    ; add WSP(r_out as u32), WSP(r), c
                )
            } else {
                println!("TODO: allocate these temporary registers in a real/safe way");
                load_32_bit_constant(ops, 0, c);
                dynasm!(ops
                    ; add W(r_out as u32), W(r), W(0)
                )
            }
        },
        (DataType::U32, Register::GPR(r_out), ConstOrReg::GPR(r1), ConstOrReg::GPR(r2)) => {
            dynasm!(ops
                ; add W(r_out as u32), W(r1), W(r2)
            )
        },
        _ => todo!("Unsupported Add operation: {:?} + {:?} with type {:?}", a, b, tp),
    }
}

fn compile_compare(ops: &mut dynasmrt::aarch64::Assembler, func: &IRFunction, allocations: &HashMap<Value, Register>, inputs: &Vec<InputSlot>, output_registers: Vec<Option<&Register>>) {
    let a = input_slot_to_imm_or_reg(&inputs[0], func, &allocations);
    let b = input_slot_to_imm_or_reg(&inputs[2], func, &allocations);

    let r_out = *output_registers[0].unwrap();

    if let InputSlot::Constant(Constant::CompareType(compare_type)) = inputs[1] {
        // TODO: redo this to do the cmp and the cset in separate match statements to reduce
        // duplication
        match (r_out, a, compare_type, b) {
            (Register::GPR(r_out), ConstOrReg::GPR(r1), CompareType::LessThanUnsigned, ConstOrReg::GPR(r2)) => {
                dynasm!(ops
                    ; cmp X(r1 as u32), X(r2 as u32)
                    ; cset W(r_out as u32), lo // unsigned "lower"
                )
            },
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
            },
            _ => todo!("Unsupported Compare operation: {:?} {:?} {:?}", a, compare_type, b),
        }
    } else {
        panic!("Expected a compare type constant as the second input to Compare");
    }
}

fn compile_write_ptr(ops: &mut dynasmrt::aarch64::Assembler, func: &IRFunction, allocations: &HashMap<Value, Register>, inputs: &Vec<InputSlot>) {
    assert_eq!(inputs.len(), 3);
    // ptr, value, type
    let ptr = input_slot_to_imm_or_reg(&inputs[0], func, &allocations);
    let value = input_slot_to_imm_or_reg(&inputs[1], func, &allocations);
    if let InputSlot::Constant(Constant::DataType(tp)) = inputs[2] {
        match (&ptr, &value, tp) {
            (ConstOrReg::U64(ptr), ConstOrReg::U32(value), DataType::U32) => {
                println!("TODO: allocate these temporary registers in a real/safe way");
                load_64_bit_constant(ops, 0, *ptr);
                load_32_bit_constant(ops, 1, *value);
                dynasm!(ops
                ; str W(1), [X(0)]
            )
            },
            (ConstOrReg::U64(ptr), ConstOrReg::GPR(value), DataType::U32) => {
                println!("TODO: allocate these temporary registers in a real/safe way");
                load_64_bit_constant(ops, 0, *ptr);
                dynasm!(ops
                ; str W((*value) as u32), [X(0)]
            )
            },
            (ConstOrReg::GPR(_), ConstOrReg::U32(_), DataType::U32) => todo!(),
            (ConstOrReg::GPR(_), ConstOrReg::GPR(_), DataType::U32) => todo!(),
            _ => todo!("Unsupported WritePtr operation: {:?} = {:?} with type {}", ptr, value, tp),
        }
    } else {
        panic!("Expected a datatype constant as the third input to WritePtr");
    }
}

fn compile_spill_to_stack(ops: &mut dynasmrt::aarch64::Assembler, func: &IRFunction, allocations: &HashMap<Value, Register>, inputs: &Vec<InputSlot>) {
    let to_spill = input_slot_to_imm_or_reg(&inputs[0], func, &allocations);
    let stack_offset = input_slot_to_imm_or_reg(&inputs[1], func, &allocations);
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
}

fn compile_load_from_stack(ops: &mut dynasmrt::aarch64::Assembler, func: &IRFunction, allocations: &HashMap<Value, Register>, inputs: &Vec<InputSlot>, outputs: &Vec<OutputSlot>, output_registers: Vec<Option<&Register>>) {
    let stack_offset = input_slot_to_imm_or_reg(&inputs[0], func, &allocations);

    let r_out = *output_registers[0].unwrap();
    let tp = outputs[0].tp;

    match (r_out, &stack_offset, tp) {
        (Register::GPR(r_out), ConstOrReg::U64(offset), DataType::U32) => {
            dynasm!(ops
            ; ldr W(r_out as u32), [sp, func.get_stack_offset_for_location(*offset, DataType::U32)]
        )
        },
        _ => todo!("Unsupported LoadFromStack operation: load {} from offset {:?} with datatype {}", r_out, stack_offset, tp),
    }
}

fn calculate_callee_saved_regs(func: &mut IRFunction, allocations: &HashMap<Value, Register>) -> Vec<(Register, usize)> {
    allocations
        .iter()
        .map(|(_, reg)| reg)
        .unique()
        .flat_map(|reg| {
            if !reg.is_volatile() {
                Some((*reg, func.new_sized_stack_location(reg.size())))
            } else {
                None
            }
        })
        .collect::<Vec<_>>()
}

fn save_callee_regs_to_stack(ops: &mut dynasmrt::aarch64::Assembler, func: &IRFunction, callee_saved: &Vec<(Register, usize)>) {
    for (reg, stack_location) in callee_saved {
        match *reg {
            Register::GPR(r) => {
                assert_eq!(reg.size(), 8);
                dynasm!(ops
                    ; str X(r as u32), [sp, func.get_stack_offset_for_location(*stack_location as u64, DataType::U64)]
                )
            },
        }
    }
}

fn pop_callee_regs_from_stack(ops: &mut dynasmrt::aarch64::Assembler, func: &IRFunction, callee_saved: &Vec<(Register, usize)>) {
    for (reg, stack_location) in callee_saved {
        match *reg {
            Register::GPR(r) => {
                assert_eq!(reg.size(), 8);
                dynasm!(ops
                    ; ldr X(r as u32), [sp, func.get_stack_offset_for_location(*stack_location as u64, DataType::U64)]
                )
            },
        }
    }
}

/// Moves a set of values into a set of registers. The sets can overlap, but the algorithm assumes
/// each move is unique. That is, {A->C, B->C} is not allowed, but {A->B, B->C, C->A} is allowed.
/// Moves to self (A->A) will be ignored.
fn move_regs_multi(ops: &mut dynasmrt::aarch64::Assembler, mut moves: HashMap<ConstOrReg, Register>) {
    println!("Begin move_regs_multi");
    let mut pending_move_targets = HashSet::new();
    let mut pending_move_sources = HashSet::new();

    for (from, to) in &moves {
        // assert true: A register can be the target of only one move per run of this algorithm
        assert_eq!(pending_move_targets.insert(*to), true);
        if let ConstOrReg::GPR(r) = *from {
            // assert true: A register can be the source of only one move per run of this algorithm
            assert_eq!(pending_move_sources.insert(Register::GPR(r as usize)), true);
        }
    }

    // Used to reorder the moves in case there's a conflict
    let mut postponed_moves = Vec::new();
    while moves.len() > 0 {
        postponed_moves.push(*moves.keys().next().unwrap());

        while let Some(from) = postponed_moves.pop() {
            let to = moves[&from];
            println!("Moving {:?} to {:?}", from, to);
            println!("\tPending move targets: {:?}", pending_move_targets);
            println!("\tPending move sources: {:?}", pending_move_sources);
            if Some(to) == from.to_reg() {
                println!("Moving {:?} to itself (eliding the move and not outputting anything)", from);
                // If it's a move to self, no-op.
                moves.remove(&from);
                pending_move_targets.remove(&to);
                pending_move_sources.remove(&to);
            } else if pending_move_sources.contains(&to) {
                if postponed_moves.contains(&to.to_const_or_reg()) {
                    // What I think I have to do here is:
                    //
                    // Allocate a temporary register for the move target.
                    //
                    // Move the target into the temporary register.
                    //
                    // Remove the move of the target to _its target_ from the move queue and replace
                    // it with a move from the temp reg to the target's target.
                    //
                    // Add this move back to postponed_moves and `continue`
                    panic!("Would overwrite a pending move target - we have a cycle. Need to allocate temp regs to fix this.");
                } else {
                    println!("\tWould overwrite pending move source {:?} with {:?}, postponing the move (this does not conflict with an already postponed move)", to, from);
                    // We couldn't make this move, so we need to add it back to the list of moves
                    postponed_moves.push(from);
                    // But do the conflicting one first
                    postponed_moves.push(to.to_const_or_reg());
                }
            } else {
                // It is safe to do the move. It's not a self-move, and it doesn't conflict with any other moves.
                match (from, to) {
                    (ConstOrReg::U32(c), Register::GPR(r_to)) => {
                        println!("\tMoving constant {} to register {}", c, r_to);
                        load_32_bit_constant(ops, r_to as u32, c);
                        moves.remove(&from);
                        pending_move_targets.remove(&to);
                        // It was a constant, so no need to remove the source
                    },
                    (ConstOrReg::U64(_), Register::GPR(_)) => todo!("Moving {:?} to {}", from, to),
                    (ConstOrReg::GPR(r_from), Register::GPR(r_to)) => {
                        dynasm!(ops
                            ; mov X(r_to as u32), X(r_from as u32)
                        );
                        moves.remove(&from);
                        pending_move_targets.remove(&to);
                        pending_move_sources.remove(&from.to_reg().unwrap());
                    },
                }
            }
        }
    }
    println!();
}

fn call_block(ops: &mut dynasmrt::aarch64::Assembler, func: &IRFunction, allocations: &HashMap<Value, Register>, target: &BlockReference, block_labels: &Vec<dynasmrt::DynamicLabel>, from_block_index: usize) {
    let moves = target
        .arguments
        .iter()
        .enumerate()
        .map(|(input_index, arg)| {
            let data_type = func.blocks[target.block_index].inputs[input_index];

            let in_block_value = Value::BlockInput {
                block_index: target.block_index,
                input_index,
                data_type,
            };

            let block_arg_reg = allocations.get(&in_block_value).unwrap();
            (input_slot_to_imm_or_reg(&arg, func, &allocations), *block_arg_reg)
        })
        .collect::<HashMap<_, _>>();

    if moves.len() > 0 {
        move_regs_multi(ops, moves);
    }

    // TODO: figure out when we can elide this jump. If it's a jmp instruction to the next block,
    // we definitely can, but it gets more complicated when it's a branch instruction. Maybe the
    // branch instruction should detect if one of the targets is the next block and always put that
    // second. Then we could always elide this jump here.
    // if target.block_index != from_block_index + 1 {
        let target_label = block_labels[target.block_index];
        dynasm!(ops
            ; b =>target_label
        )
    // }
}

pub fn compile(func: &mut IRFunction) {
    let allocations = alloc_for(func); // TODO: this will eventually return some data

    println!("Allocations:");
    allocations.iter().sorted_by_key(|(v, _)| **v).for_each(|(k, v)| {
        println!("{} \t {:?}", k, v);
    });

    let mut ops = dynasmrt::aarch64::Assembler::new().unwrap();

    let callee_saved = calculate_callee_saved_regs(func, &allocations);


    // Stack bytes used: aligned to 16 bytes
    let misalignment = func.stack_bytes_used % 16;
    let correction = if misalignment == 0 { 0 } else { 16 - misalignment };
    let stack_bytes_used = func.stack_bytes_used + correction;
    println!("Function uses {} bytes of stack, misaligned by {}, corrected to {}", func.stack_bytes_used, misalignment, stack_bytes_used);
    func.stack_bytes_used = stack_bytes_used;

    // Entrypoint to the function - get the offset before appending anything
    let entrypoint = ops.offset();

    println!("Compiling function:\n{}", func);

    // Setup stack
    dynasm!(ops
        ; .arch aarch64
    );
    if stack_bytes_used > 0 {
        dynasm!(ops
            ; sub sp, sp, stack_bytes_used.try_into().unwrap()
        )
    }

    save_callee_regs_to_stack(&mut ops, func, &callee_saved);

    let block_labels = func.blocks.iter().map(|_| ops.new_dynamic_label()).collect::<Vec<_>>();

    for (block_index, block) in func.blocks.iter().enumerate() {
        dynasm!(ops
            ; =>block_labels[block_index]
        );

        for (_instruction_index_in_block, instruction_index) in block.instructions.iter().enumerate() {
            let instruction = &func.instructions[*instruction_index];

            match &instruction.instruction {
                crate::ir::Instruction::Instruction { tp, inputs, outputs } => {
                    let output_registers = outputs
                        .iter()
                        .enumerate()
                        .map(|(output_index, output)| {
                            allocations.get(&Value::InstructionOutput {
                                instruction_index: *instruction_index,
                                output_index,
                                block_index,
                                data_type: output.tp,
                            })
                        }).collect::<Vec<_>>();

                    match tp {
                        crate::ir::InstructionType::Add => compile_add(&mut ops, func, &allocations, inputs, outputs, output_registers),
                        crate::ir::InstructionType::Compare => compile_compare(&mut ops, func, &allocations, inputs, output_registers),
                        crate::ir::InstructionType::LoadPtr => todo!("load_ptr"),
                        crate::ir::InstructionType::WritePtr => compile_write_ptr(&mut ops, func, &allocations, inputs),
                        crate::ir::InstructionType::SpillToStack => compile_spill_to_stack(&mut ops, func, &allocations, inputs),
                        crate::ir::InstructionType::LoadFromStack => compile_load_from_stack(&mut ops, func, &allocations, inputs, outputs, output_registers),
                    }
                },
                crate::ir::Instruction::Branch { cond, if_true, if_false } => {
                    let cond = input_slot_to_imm_or_reg(&cond, func, &allocations);

                    match cond {
                        ConstOrReg::GPR(c) => {
                            dynasm!(ops
                                ; cbz W(c), >if_false
                            );
                        },
                        _ => todo!("Unsupported branch condition: {:?}", cond),
                    }
                    call_block(&mut ops, func, &allocations, if_true, &block_labels, block_index);
                    dynasm!(ops
                        ; if_false:
                    );
                    call_block(&mut ops, func, &allocations, if_false, &block_labels, block_index);
                },
                crate::ir::Instruction::Jump { target } => {
                    call_block(&mut ops, func, &allocations, target, &block_labels, block_index);
                },
                crate::ir::Instruction::Return { .. } => {
                    pop_callee_regs_from_stack(&mut ops, func, &callee_saved);
                    // Fix sp
                    if stack_bytes_used > 0 {
                        dynasm!(ops
                            ; add sp, sp, stack_bytes_used.try_into().unwrap()
                        );
                    }
                    dynasm!(ops
                        ; ret
                    );
                },
            }
        }
    }


    let code = ops.finalize().unwrap();
    let f: extern "C" fn(x: u64) -> u64 = unsafe { mem::transmute(code.ptr(entrypoint)) };

    println!("{}", disassemble(&code, f as u64));

    println!("Running:");
    f(1);
}
