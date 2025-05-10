use std::{
    cmp::{max, min},
    collections::HashMap,
    fmt::Display,
};

use crate::ir::{IRFunction, InputSlot};

use itertools::Itertools;

#[derive(Hash, PartialEq, Eq, Debug, Clone, Copy)]
enum Value {
    InstructionOutput {
        block_index: usize,
        instruction_index: usize,
        output_index: usize,
    },
    BlockInput {
        block_index: usize,
        input_index: usize,
    },
}

impl Value {
    /// Get the "first usage" of this value
    fn into_usage(&self) -> Usage {
        match self {
            Value::InstructionOutput {
                block_index,
                instruction_index,
                ..
            } => Usage {
                block_index: *block_index,
                instruction_index: *instruction_index,
            },
            Value::BlockInput { block_index, .. } => Usage {
                block_index: *block_index,
                instruction_index: 0,
            },
        }
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::InstructionOutput {
                block_index,
                instruction_index,
                output_index,
            } => {
                if *output_index == 0 {
                    write!(f, "b{}v{}", block_index, instruction_index)
                } else {
                    write!(f, "b{}v{}o{}", block_index, instruction_index, output_index)
                }
            }
            Value::BlockInput {
                block_index,
                input_index,
            } => {
                write!(f, "b{}i{}", block_index, input_index)
            }
        }
    }
}

impl InputSlot {
    fn to_value(self) -> Option<Value> {
        match self {
            InputSlot::InstructionOutput {
                block_index,
                instruction_index,
                output_index,
                ..
            } => Some(Value::InstructionOutput {
                block_index,
                instruction_index,
                output_index,
            }),
            InputSlot::BlockInput {
                block_index,
                input_index,
                ..
            } => Some(Value::BlockInput {
                block_index,
                input_index,
            }),
            InputSlot::Constant(_) => None,
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
struct Usage {
    block_index: usize,
    instruction_index: usize,
}

impl PartialOrd for Usage {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self.block_index == other.block_index {
            // If the values are in the same block, compare the instruction index
            Some(self.instruction_index.cmp(&other.instruction_index))
        } else {
            // Otherwise, compare the block index
            Some(self.block_index.cmp(&other.block_index))
        }
    }
}

impl Ord for Usage {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if self.block_index == other.block_index {
            // If the values are in the same block, compare the instruction index
            self.instruction_index.cmp(&other.instruction_index)
        } else {
            // Otherwise, compare the block index
            self.block_index.cmp(&other.block_index)
        }
    }
}

impl Display for Usage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Last used at block {} instruction index {}",
            self.block_index, self.instruction_index
        )
    }
}

struct Lifetimes {
    last_used: HashMap<Value, Usage>,
    interference: HashMap<Value, Vec<Value>>,
}

impl Lifetimes {
    /// Is a value active at a given index?
    fn is_active(&self, value: Value, at_block_index: usize, at_instruction_index: usize) -> bool {
        let last_usage = &self.last_used[&value];
        match value {
            Value::InstructionOutput {
                block_index,
                instruction_index,
                ..
            } => {
                at_block_index >= block_index
                    && at_instruction_index > instruction_index
                    && at_block_index <= last_usage.block_index
                    && at_instruction_index <= last_usage.instruction_index
            }
            Value::BlockInput { block_index, .. } => {
                at_block_index >= block_index && at_block_index <= last_usage.block_index
            }
        }
    }
}

fn calculate_lifetimes(func: IRFunction) -> Lifetimes {
    println!("Calculating lifetimes");
    let mut last_used = HashMap::new();
    let mut interference = HashMap::new();

    func.blocks
        .iter()
        .enumerate()
        .flat_map(|(block_index, block)| {
            block.instructions.iter().enumerate().map(
                move |(instruction_index_in_block, instruction_index)| {
                    (block_index, instruction_index_in_block, instruction_index)
                },
            )
        })
        .for_each(
            |(block_index, instruction_index_in_block, instruction_index)| match &func.instructions
                [*instruction_index]
                .instruction
            {
                crate::ir::Instruction::Instruction { inputs, .. } => {
                    inputs
                        .iter()
                        .map(|input| input.to_value())
                        .for_each(|input| {
                            if let Some(value) = input {
                                last_used.insert(
                                    value,
                                    Usage {
                                        block_index,
                                        instruction_index: instruction_index_in_block,
                                    },
                                );
                            };
                        });
                }
                crate::ir::Instruction::Branch { cond, .. } => {
                    if let Some(value) = cond.to_value() {
                        last_used.insert(
                            value,
                            Usage {
                                block_index,
                                instruction_index: instruction_index_in_block,
                            },
                        );
                    };
                }
                crate::ir::Instruction::Jump { .. } => {}
                crate::ir::Instruction::Return { value } => value.into_iter().for_each(|input| {
                    if let Some(input) = input.to_value() {
                        last_used.insert(
                            input,
                            Usage {
                                block_index,
                                instruction_index: instruction_index_in_block,
                            },
                        );
                    }
                }),
            },
        );

    last_used.keys().combinations(2).for_each(|x| {
        let a = x[0];
        let b = x[1];

        let a_first = a.into_usage();
        let b_first = b.into_usage();

        let a_last = &last_used[a];
        let b_last = &last_used[b];

        // If the live ranges of the two values overlap, add them to the interference graph
        let overlap = max(a_first, b_first) <= *min(a_last, b_last);
        if overlap {
            interference.entry(*a).or_insert_with(Vec::new).push(*b);
            interference.entry(*b).or_insert_with(Vec::new).push(*a);
        }
    });

    Lifetimes {
        last_used,
        interference,
    }
}

pub fn alloc_for(func: IRFunction) {
    let lifetimes = calculate_lifetimes(func);
    for (value, usage) in lifetimes.last_used.iter() {
        println!("{}: {}", value, usage);
    }

    // The number of registers needed is the maximum number of values that are active at once - or
    // the node with the most edges in the interference graph
    let num_regs_needed = lifetimes
        .interference
        .values()
        .map(|v| v.len())
        .max()
        .unwrap_or(0);

    println!("Number of registers needed: {num_regs_needed}");
}
