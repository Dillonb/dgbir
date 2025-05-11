use std::{
    cmp::{max, min},
    collections::HashMap,
    fmt::Display,
};

use crate::ir::{const_ptr, Constant, DataType, IRFunction, IndexedInstruction, InputSlot, Instruction, InstructionType, OutputSlot};

use itertools::Itertools;

#[derive(Hash, PartialEq, Eq, Debug, Clone, Copy)]
pub enum Value {
    InstructionOutput {
        /// Which block this value is in
        block_index: usize,
        /// Index of the instruction in the function
        instruction_index: usize,
        /// Which output this value is referencing
        output_index: usize,
        /// The type of the value
        data_type: DataType,
    },
    BlockInput {
        /// Which block this input is in
        block_index: usize,
        /// Which input this value is referencing
        input_index: usize,
        /// The type of the value
        data_type: DataType,
    },
}

impl Value {
    /// Get the "first usage" of this value
    fn into_usage(&self, func: &IRFunction) -> Usage {
        match self {
            Value::InstructionOutput {
                block_index,
                instruction_index,
                ..
            } => Usage {
                block_index: *block_index,
                instruction_index: *instruction_index,
                instruction_index_in_block: func.get_index_in_block(*block_index, *instruction_index).unwrap(),
            },
            Value::BlockInput { block_index, .. } => Usage {
                block_index: *block_index,
                instruction_index: 0,
                instruction_index_in_block: 0,
            },
        }
    }

    fn into_inputslot(&self) -> InputSlot {
        match self {
            Value::InstructionOutput { instruction_index, output_index, data_type, .. } => {
                InputSlot::InstructionOutput {
                    instruction_index: *instruction_index,
                    output_index: *output_index,
                    tp: *data_type,
                }
            },
            Value::BlockInput { block_index, input_index, data_type } => {
                InputSlot::BlockInput {
                    block_index: *block_index,
                    input_index: *input_index,
                    tp: *data_type,
                }
            }
        }
    }

    fn data_type(&self) -> DataType {
        match self {
            Value::InstructionOutput { data_type, .. } => *data_type,
            Value::BlockInput { data_type, .. } => *data_type,
        }
    }
}

impl PartialOrd for Value {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Value {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self, other) {
            (Value::InstructionOutput { block_index, instruction_index, output_index, .. }, Value::InstructionOutput { block_index: other_block_index, instruction_index: other_instruction_index, output_index: other_output_index, ..  }) => {
                if block_index == other_block_index {
                    if instruction_index == other_instruction_index {
                        return output_index.cmp(other_output_index);
                    } else {
                        return instruction_index.cmp(other_instruction_index);
                    }
                } else {
                    return block_index.cmp(other_block_index);
                }
            },
            (Value::InstructionOutput { block_index, .. }, Value::BlockInput { block_index: other_block_index, .. }) => {
                if block_index == other_block_index {
                    // When both are in the same block, instruction outputs are always greater than
                    // block inputs
                    return std::cmp::Ordering::Greater;
                } else {
                    return block_index.cmp(other_block_index);
                }
            },
            (Value::BlockInput { block_index, .. }, Value::InstructionOutput { block_index: other_block_index, .. }) => {
                if block_index == other_block_index {
                    // When both are in the same block, instruction outputs are always greater than
                    // block inputs
                    return std::cmp::Ordering::Less;
                } else {
                    return block_index.cmp(other_block_index);
                }
            },
            (Value::BlockInput { block_index, input_index, .. }, Value::BlockInput { block_index: other_block_index, input_index: other_input_index, .. }) => {
                if block_index == other_block_index {
                    return input_index.cmp(other_input_index);
                } else {
                    return block_index.cmp(other_block_index);
                }
            },
        }
    }
}


impl InputSlot {
    fn references_value(&self, value: &Value) -> bool {
        match self {
            InputSlot::InstructionOutput { instruction_index, output_index, .. } => {
                match value {
                    Value::InstructionOutput { instruction_index: v_instruction_index, output_index: v_output_index, .. } => {
                        *instruction_index == *v_instruction_index && *output_index == *v_output_index
                    },
                    _ => false,
                }
            },
            InputSlot::BlockInput { block_index, input_index, .. } => {
                match value {
                    Value::BlockInput { block_index: v_block_index, input_index: v_input_index, .. } => {
                        *block_index == *v_block_index && *input_index == *v_input_index
                    },
                    _ => false,
                }
            },
            InputSlot::Constant(..) => false,
        }
    }

    pub fn to_value(self, func: &IRFunction) -> Option<Value> {
        match self {
            InputSlot::InstructionOutput {
                instruction_index,
                output_index,
                tp,
                ..
            } => {
                let block_index = func.instructions[instruction_index].block_index;
                Some(Value::InstructionOutput {
                    block_index,
                    instruction_index,
                    output_index,
                    data_type: tp,
                })
            },
            InputSlot::BlockInput {
                block_index,
                input_index,
                tp,
                ..
            } => Some(Value::BlockInput {
                block_index,
                input_index,
                data_type: tp,
            }),
            InputSlot::Constant(_) => None,
        }
    }
}

struct IRFunctionValueIterator<'a> {
    pub function: &'a IRFunction,
    block_index: usize,
    block_input_index: usize,
    instruction_index: usize,
    output_index: usize,
}

impl Iterator for IRFunctionValueIterator<'_> {
    type Item = Value;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.block_index >= self.function.blocks.len() {
                return None;
            }

            let block = &self.function.blocks[self.block_index];

            // If we're not done iterating through the inputs, self.block_input_index will be less than
            // the number of inputs
            if self.block_input_index < block.inputs.len() {
                self.block_input_index += 1;
                return Some(Value::BlockInput {
                    data_type: block.inputs[self.block_input_index],
                    block_index: self.block_index,
                    input_index: self.block_input_index,
                });
            }

            let fn_instruction_index = block.instructions[self.instruction_index];
            let instruction = &self.function.instructions[fn_instruction_index];

            match &instruction.instruction {
                crate::ir::Instruction::Instruction { outputs, .. } => {
                    let v = if self.output_index < outputs.len() {
                        Some(Value::InstructionOutput {
                            block_index: instruction.block_index,
                            instruction_index: instruction.index,
                            output_index: self.output_index,
                            data_type: outputs[self.output_index].tp,
                        })
                    } else {
                        None
                    };

                    self.output_index = 0;
                    self.instruction_index += 1;
                    if self.instruction_index >= block.instructions.len() {
                        self.block_index += 1;
                        self.block_input_index = 0;
                        self.instruction_index = 0;
                    }

                    if v.is_some() {
                        return v;
                    }
                },

                // These aren't values, so skip
                crate::ir::Instruction::Branch { .. } | crate::ir::Instruction::Jump { .. } | crate::ir::Instruction::Return { .. } => {
                    self.output_index = 0;
                    self.instruction_index += 1;
                    if self.instruction_index >= block.instructions.len() {
                        self.block_index += 1;
                        self.block_input_index = 0;
                        self.instruction_index = 0;
                    }
                }
            }
        }
    }
}

impl IRFunction {
    fn value_iter(&self) -> IRFunctionValueIterator {
        IRFunctionValueIterator {
            function: self,
            block_index: 0,
            block_input_index: 0,
            instruction_index: 0,
            output_index: 0,
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
                data_type,
                ..
            } => {
                if *output_index == 0 {
                    write!(f, "v{}(b{}):{}", instruction_index, block_index, data_type)
                } else {
                    write!(f, "v{}(b{})o{}:{}", instruction_index, block_index, output_index, data_type)
                }
            }
            Value::BlockInput {
                block_index,
                input_index,
                data_type
            } => {
                write!(f, "b{}i{}:{}", block_index, input_index, data_type)
            }
        }
    }
}

#[derive(Debug, Eq, PartialEq, Clone, Copy)]
struct Usage {
    block_index: usize,
    instruction_index: usize,
    instruction_index_in_block : usize
}

impl Usage {
    fn recalculate_index_in_block(&self, func: &IRFunction) -> Self {
        Usage {
            block_index: self.block_index,
            instruction_index: self.instruction_index,
            instruction_index_in_block: func.get_index_in_block(self.block_index, self.instruction_index).unwrap(),
        }
    }
}

impl PartialOrd for Usage {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self.block_index == other.block_index {
            // If the values are in the same block, compare the instruction index
            Some(self.instruction_index_in_block.cmp(&other.instruction_index_in_block))
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
            self.instruction_index_in_block.cmp(&other.instruction_index_in_block)
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
            "used at block {} instruction index {} (v{})",
            self.block_index, self.instruction_index_in_block, self.instruction_index
        )
    }
}

struct Lifetimes {
    last_used: HashMap<Value, Usage>,
    interference: HashMap<Value, Vec<Value>>,
    /// A list of all usages of a value. Guaranteed to be sorted.
    all_usages: HashMap<Value, Vec<Usage>>,
}

fn calculate_lifetimes(func: &IRFunction) -> Lifetimes {
    println!("Calculating lifetimes");
    let mut last_used = HashMap::new();
    let mut all_usages : HashMap<Value, Vec<Usage>> = HashMap::new();
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
                        .map(|input| input.to_value(&func))
                        .for_each(|input| {
                            if let Some(value) = input {
                                let u = Usage {
                                    block_index,
                                    instruction_index: *instruction_index,
                                    instruction_index_in_block,
                                };
                                println!("{} {}", value, u);
                                last_used.insert(value, u);
                                all_usages
                                    .entry(value)
                                    .or_insert_with(Vec::new)
                                    .push(u);
                            };
                        });
                }
                crate::ir::Instruction::Branch { cond, .. } => {
                    if let Some(value) = cond.to_value(&func) {
                        let u = Usage {
                            block_index,
                            instruction_index: *instruction_index,
                            instruction_index_in_block,
                        };
                        println!("{} {}", value, u);
                        last_used.insert(
                            value,
                            u,
                        );
                        all_usages
                            .entry(value)
                            .or_insert_with(Vec::new)
                            .push(u);
                    };
                }
                crate::ir::Instruction::Jump { .. } => {}
                crate::ir::Instruction::Return { value } => value.into_iter().for_each(|input| {
                    if let Some(value) = input.to_value(&func) {
                        let u = Usage {
                            block_index,
                            instruction_index: *instruction_index,
                            instruction_index_in_block,
                        };
                        println!("{} {}", value, u);
                        last_used.insert(
                            value,
                            u,
                        );
                        all_usages
                            .entry(value)
                            .or_insert_with(Vec::new)
                            .push(u);
                    }
                }),
            },
        );

    last_used.keys().combinations(2).for_each(|x| {
        let a = x[0];
        let b = x[1];

        let a_first = a.into_usage(func);
        let b_first = b.into_usage(func);

        let a_last = &last_used[a];
        let b_last = &last_used[b];

        // If the live ranges of the two values overlap, add them to the interference graph
        let overlap = max(a_first, b_first) <= *min(a_last, b_last);
        // println!("Checking interference between {} and {}", a, b);

        if overlap {
            // println!("\t{} and {} interfere", a, b);
            interference.entry(*a).or_insert_with(Vec::new).push(*b);
            interference.entry(*b).or_insert_with(Vec::new).push(*a);
        }
    });

    Lifetimes {
        last_used,
        all_usages,
        interference,
    }
}

#[derive(Hash, PartialEq, Eq, Debug, Clone, Copy)]
pub enum Register {
    GPR(usize),
}

impl Display for Register {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Register::GPR(r) => write!(f, "GPR({})", r),
        }
    }
}

fn get_registers() -> Vec<Register> {
    // Callee-saved registers
    #[cfg(target_arch = "aarch64")]
    vec![19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
    // // vec![19, 20, 21, 22, 23] // Removed a bunch to test register spilling
    // vec![19, 20, 21] // Removed a bunch to test register spilling
        .into_iter()
        .map(|r| Register::GPR(r))
        .collect()
}

impl IRFunction {
    fn new_stack_location(&mut self, tp: DataType) -> usize {
        let bytes_needed = tp.size();

        // Align the stack to this data type
        self.stack_bytes_used += bytes_needed - (self.stack_bytes_used % bytes_needed);

        return self.stack_bytes_used - bytes_needed;
    }

    pub fn get_stack_offset_for_location(&self, location: u64, tp: DataType) -> u32 {
        (self.stack_bytes_used - location as usize - tp.size()) as u32
    }

    fn get_index_in_block(&self, block_index: usize, instruction_index: usize) -> Option<usize> {
        self.blocks[block_index]
            .instructions
            .iter()
            .position(|i| {
                *i == instruction_index
            })
    }

    fn spill(&mut self, to_spill: &Value, final_usage_pre_spill: &Usage, usages_post_spill: Vec<&Usage>) {
        let to_spill_inputslot = to_spill.into_inputslot();
        println!("Final usage of {} before spill: {}", to_spill, final_usage_pre_spill);
        println!("Usages after spill: {}", usages_post_spill.iter().map(|u| u.to_string()).join(", "));

        let stack_location = self.new_stack_location(to_spill.data_type());

        let spill_instr_index = self.instructions.len();
        let spill_instr = IndexedInstruction {
            block_index: final_usage_pre_spill.block_index,
            index: spill_instr_index,
            instruction: Instruction::Instruction {
                tp: InstructionType::SpillToStack,
                inputs: vec![to_spill_inputslot, const_ptr(stack_location), InputSlot::Constant(Constant::DataType(to_spill.data_type()))],
                outputs: vec![]
            },
        };
        self.instructions.push(spill_instr);

        {
            let i = final_usage_pre_spill.instruction_index_in_block + 1; // Insert immediately after the last usage pre-spill
            self.blocks[final_usage_pre_spill.block_index].instructions.splice(i..i, [
                spill_instr_index
            ]);
        }

        // Now, insert a reload instruction before the first usage after the spill

        let first_usage_post_spill = usages_post_spill[0].recalculate_index_in_block(self);

        let reload_instr_index = self.instructions.len();
        let reload_instr = IndexedInstruction {
            block_index: first_usage_post_spill.block_index,
            index: reload_instr_index,
            instruction: Instruction::Instruction {
                tp: InstructionType::LoadFromStack,
                inputs: vec![const_ptr(stack_location), InputSlot::Constant(Constant::DataType(to_spill.data_type()))],
                outputs: vec![ OutputSlot { tp: to_spill.data_type() }],
            },
        };
        self.instructions.push(reload_instr);

        {
            let i = first_usage_post_spill.instruction_index_in_block; // Insert immediately before the first usage post-spill
            self.blocks[first_usage_post_spill.block_index].instructions.splice(i..i, [
                reload_instr_index
            ]);
        }

        let reloaded_inputslot = InputSlot::InstructionOutput {
            instruction_index: reload_instr_index,
            output_index: 0,
            tp: to_spill.data_type(),
        };

        // Then rewrite all following usages to use that reload:
        for usage in usages_post_spill {
            let instruction = &mut self.instructions[usage.instruction_index];
            match &mut instruction.instruction {
                Instruction::Instruction { inputs, .. } => {
                    let indices = inputs
                        .into_iter()
                        .positions(|input| input.references_value(to_spill))
                        .collect::<Vec<usize>>();

                    for i in indices {
                            inputs[i] = reloaded_inputslot;
                    }
                },
                Instruction::Branch { cond, .. } => {
                    if cond.references_value(to_spill) {
                        *cond = reloaded_inputslot;
                    }
                },
                Instruction::Jump { .. } => {},
                Instruction::Return { value } => {
                    if let Some(v) = value {
                        if v.references_value(to_spill) {
                            *v = reloaded_inputslot;
                        }
                    }
                },
            }
        }

        println!("{}", self);
    }
}

pub fn alloc_for(func: &mut IRFunction) -> HashMap<Value, Register> {
    let mut done = false;
    let mut allocations = HashMap::new();
    while !done {
        allocations.clear();
        let mut to_spill = None;
        let lifetimes = calculate_lifetimes(&func);
        for (value, usage) in lifetimes.last_used.iter() {
            println!("{}: last {}", value, usage);
        }


        println!("Iterating through all values:");

        for value in func.value_iter() {
            println!("{} {}", value, lifetimes.last_used[&value]);
            let interference = &lifetimes.interference[&value];
            println!("\tInterference: {}", interference.iter().join(", "));

            let mut found_reg = false;
            for reg in get_registers() {
                // Check if the register is already allocated to an interfering value
                let already_allocated = interference.iter().flat_map(|iv| allocations.get(iv)).any(|r| *r == reg);
                if already_allocated {
                    println!("\t\tRegister {} is already allocated to an interfering value", reg);
                } else {
                    println!("\t\tRegister {} is available", reg);
                    allocations.insert(value, reg);
                    found_reg = true;
                    break;
                }
            }

            if !found_reg {
                println!("Interfering values with allocated registers (active values):");
                to_spill = interference
                    .iter()
                    .filter(|iv| allocations.get(iv).is_some())
                    .flat_map(|iv| {
                        lifetimes.all_usages[iv].iter().find(|u| u >= &&value.into_usage(func)).map(|u| {
                            (*u, *iv) // This does a copy so we don't have to deal with lifetimes
                        })
                    })
                    .max_by(|(u1, _iv1), (u2, _iv2)| {
                        u1.cmp(u2)
                    })
                    .clone();
                if to_spill.is_none() {
                    panic!("Couldn't find a value to spill!");
                }
                let (to_spill_next_used, to_spill) = to_spill.unwrap();
                if to_spill_next_used == value.into_usage(func) {
                    panic!("Tried to spill a value next used at the same time as the one we're allocating (??? are we out of registers?)");
                }
                println!("Spilling value {} to stack - next {}", to_spill, to_spill_next_used);
                break;
            }
        }

        if to_spill.is_none() {
            done = true;
        } else {
            let (to_spill_next_used, to_spill) = to_spill.unwrap();
            let to_spill_first_usage = to_spill.into_usage(func);

            let final_usage_pre_spill = lifetimes.all_usages[&to_spill]
                .iter()
                .filter(|u| u < &&to_spill_next_used)
                .last()
                .unwrap_or(&to_spill_first_usage);

            let usages_post_spill = lifetimes.all_usages[&to_spill]
                .iter()
                .filter(|u| u >= &&to_spill_next_used)
                .collect::<Vec<_>>();

            func.spill(&to_spill, final_usage_pre_spill, usages_post_spill);
        }
    }

    println!("Allocations:\n{}", allocations.iter().sorted_by_key(|v| v.0).map(|(v, r)| format!("\t{} -> {}", v, r)).join("\n"));
    allocations
}
