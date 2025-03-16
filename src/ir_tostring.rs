use crate::ir::{IRBlock, IndexedInstruction, InputSlot, Instruction, OutputSlot};

fn outputs_tostring(instr_index: usize, inputs: &Vec<OutputSlot>) -> String {
    inputs
        .iter()
        .enumerate()
        .map(|(output_index, slot)| {
            if output_index == 0 {
                return format!("v{} : {:?}", instr_index, slot.tp);
            } else {
                return format!("v{}_{} : {:?}", instr_index, output_index, slot.tp);
            }
        })
        .collect::<Vec<String>>()
        .join(", ")
}

fn inputs_tostring(inputs: &Vec<InputSlot>) -> String {
    inputs
        .iter()
        .map(|slot| match slot {
            InputSlot::Constant(constant) => {
                return format!("{:?}", constant);
            }
            InputSlot::DataType(data_type) => {
                return format!("{:?}", data_type);
            }
            InputSlot::InstructionOutput {
                instruction_index,
                tp: _,
                output_index,
            } => {
                if *output_index == 0 {
                    return format!("v{}", instruction_index);
                } else {
                    return format!("v{}_{}", instruction_index, output_index);
                }
            }
        })
        .collect::<Vec<String>>()
        .join(", ")
}

fn unindexed_instruction_tostring(instr: &Instruction) -> String {
    format!("{:?}({})", instr.tp, inputs_tostring(&instr.inputs))
}

pub fn instruction_tostring(instr: &IndexedInstruction) -> String {
    let outputs = outputs_tostring(instr.index, &instr.instruction.outputs);
    let instr_string = unindexed_instruction_tostring(&instr.instruction);
    if outputs.is_empty() {
        return instr_string;
    } else {
        return format!("{} = {}", outputs, instr_string);
    }
}

pub fn block_tostring(block: &IRBlock) -> String {
    return block
        .instructions
        .iter()
        .map(|instruction| instruction_tostring(instruction))
        .collect::<Vec<String>>()
        .join("\n");
}
