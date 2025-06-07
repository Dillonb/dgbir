use std::fmt::Display;

use super::*;

impl Display for DataType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl Display for InputSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InputSlot::InstructionOutput {
                instruction_index,
                output_index,
                ..
            } => {
                if *output_index > 0 {
                    write!(f, "v{}_{}", instruction_index, output_index)
                } else {
                    write!(f, "v{}", instruction_index)
                }
            }
            InputSlot::BlockInput {
                block_index,
                input_index,
                ..
            } => write!(f, "b{}i{}", block_index, input_index),
            InputSlot::Constant(constant) => write!(f, "{:?}", constant),
        }
    }
}

impl Display for BlockReference {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inputs = self
            .arguments
            .iter()
            .map(|input| format!("{input}"))
            .collect::<Vec<String>>()
            .join(", ");

        write!(f, "block_{}({})", self.block_index, inputs)
    }
}

impl Display for IndexedInstruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match &self.instruction {
                Instruction::Instruction { tp, inputs, outputs } => {
                    let outputs = outputs
                        .iter()
                        .enumerate()
                        .map(|(i, output)| {
                            if i > 0 {
                                format!("v{}_{} : {:?}", self.index, i, output.tp)
                            } else {
                                format!("v{} : {:?}", self.index, output.tp)
                            }
                        })
                        .collect::<Vec<String>>()
                        .join(", ");

                    let inputs = inputs
                        .iter()
                        .map(|input| format!("{input}"))
                        .collect::<Vec<String>>()
                        .join(", ");

                    if outputs.is_empty() {
                        format!("{tp:?}({inputs})")
                    } else {
                        format!("{outputs} = {tp:?}({inputs})")
                    }
                }
                Instruction::Branch {
                    cond,
                    if_true,
                    if_false,
                } => {
                    format!("Branch({} ? {} : {})", cond, if_true, if_false)
                }
                Instruction::Jump { target } => {
                    format!("Jump {target}")
                }
                Instruction::Return { value } => {
                    match value {
                        Some(value) => format!("Return {value}"),
                        None => "Return".to_string(),
                    }
                }
            }
        )
    }
}

impl Display for IRBasicBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "block_{}()", self.index)
    }
}

impl Display for IRFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Forward to IRFunctionInternal formatter
        self.func.borrow().fmt(f)
    }
}

impl Display for IRFunctionInternal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.blocks
            .iter()
            .map(|block| {
                let block_name = format!("block_{}", block.index);
                let inputs = block
                    .inputs
                    .iter()
                    .enumerate()
                    .map(|(i, tp)| format!("b{}i{}: {:?}", block.index, i, tp))
                    .collect::<Vec<String>>()
                    .join(", ");

                let instructions = block
                    .instructions
                    .iter()
                    .map(|instruction_index| &self.instructions[*instruction_index])
                    .map(|instruction| format!("  {instruction}"))
                    .collect::<Vec<String>>()
                    .join("\n");

                format!("{block_name}({inputs}):\n{instructions}")
            })
            .collect::<Vec<String>>()
            .join("\n\n")
            .fmt(f)
    }
}
