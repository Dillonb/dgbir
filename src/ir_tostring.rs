use crate::ir::{
    Constant, IRBasicBlock, IRFunction, IndexedInstruction, InputSlot, Instruction, InstructionType, OutputSlot
};

fn outputs_tostring(block_index: usize, instr_index: usize, inputs: &Vec<OutputSlot>) -> String {
    inputs
        .iter()
        .enumerate()
        .map(|(output_index, slot)| {
            if output_index == 0 {
                return format!("b{}v{} : {:?}", block_index, instr_index, slot.tp);
            } else {
                return format!("b{}v{}_{} : {:?}", block_index, instr_index, output_index, slot.tp);
            }
        })
        .collect::<Vec<String>>()
        .join(", ")
}

fn inputs_tostring(inputs: &Vec<InputSlot>) -> String {
    inputs
        .iter()
        .map(|slot| match slot {
            InputSlot::Constant(constant) => match constant {
                Constant::CompareType(tp) => return format!("{:?}", tp),
                _ => return format!("{:?}", constant),
            },
            InputSlot::InstructionOutput {
                block_index,
                instruction_index,
                tp: _,
                output_index,
            } => {
                if *output_index == 0 {
                    return format!("b{}v{}", block_index, instruction_index);
                } else {
                    return format!("b{}v{}_{}", block_index, instruction_index, output_index);
                }
            }
            InputSlot::BlockInput {
                block_index,
                input_index,
                ..
            } => {
                return format!("b{}i{}", block_index, input_index);
            }
        })
        .collect::<Vec<String>>()
        .join(", ")
}

fn unindexed_instruction_tostring(tp: &InstructionType, inputs: &Vec<InputSlot>) -> String {
    format!("{:?}({})", tp, inputs_tostring(&inputs))
}

pub fn instruction_tostring(instr: &IndexedInstruction) -> String {
    match &instr.instruction {
        Instruction::Instruction {
            tp,
            inputs,
            outputs,
        } => {
            let instr_string = unindexed_instruction_tostring(tp, inputs);
            let outputs_string = outputs_tostring(instr.block_index, instr.index, &outputs);
            if outputs_string.is_empty() {
                return instr_string;
            } else {
                return format!("{} = {}", outputs_string, instr_string);
            }
        }
        Instruction::Branch { .. } => "Branch(TODO)".to_string(),
        Instruction::Jump { .. } => "Jump(TODO)".to_string(),
        Instruction::Return { .. } => "Return(TODO)".to_string(),
    }
}

pub fn block_args_tostring(block: &IRBasicBlock) -> String {
    block
        .inputs
        .iter()
        .enumerate()
        .map(|(index, tp)| {
            return format!("b{}i{} : {:?}", block.index, index, tp);
        })
        .collect::<Vec<String>>()
        .join(", ")
}

pub fn block_tostring(block: &IRBasicBlock) -> String {
    return block
        .instructions
        .iter()
        .map(|instruction| instruction_tostring(instruction))
        .collect::<Vec<String>>()
        .join("\n");
}

pub fn func_tostring(func: &IRFunction) -> String {
    return func
        .blocks
        .iter()
        .enumerate()
        .map(|(index, block)| {
            format!(
                "block_{}({}):\n{}\n",
                index,
                block_args_tostring(block),
                block_tostring(block)
            )
        })
        .collect::<Vec<String>>()
        .join("\n");
}
