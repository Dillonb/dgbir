use std::{
    cmp::{max, min},
    collections::BTreeMap,
    fmt::Display,
    iter,
};

use crate::{
    abi::{get_registers, is_register_volatile},
    compiler::ConstOrReg,
    ir::{
        const_ptr, CompareType, Constant, DataType, IRFunctionInternal, IndexedInstruction, InputSlot, Instruction,
        InstructionType, OutputSlot, RoundType,
    },
};

use itertools::Itertools;

#[derive(Hash, PartialEq, Eq, Debug, Clone, Copy, PartialOrd, Ord)]
pub enum Register {
    GPR(usize),
    SIMD(usize),
}

impl Register {
    pub fn can_hold_datatype(&self, tp: DataType) -> bool {
        match self {
            Register::GPR(_) => {
                const VALID_GPR_TYPES: [DataType; 10] = [
                    DataType::U8,
                    DataType::S8,
                    DataType::U16,
                    DataType::S16,
                    DataType::U32,
                    DataType::S32,
                    DataType::U64,
                    DataType::S64,
                    DataType::Bool,
                    DataType::Ptr,
                ];
                return VALID_GPR_TYPES.contains(&tp);
            }
            Register::SIMD(_) => {
                const VALID_SIMD_TYPES: [DataType; 2] = [DataType::F32, DataType::F64];
                return VALID_SIMD_TYPES.contains(&tp);
            }
        }
    }

    pub fn is_gpr(&self) -> bool {
        match self {
            Register::GPR(_) => true,
            _ => false,
        }
    }

    pub fn expect_gpr(&self) -> usize {
        match self {
            Register::GPR(r) => *r,
            _ => panic!("Expected GPR, found {:?}", self),
        }
    }

    pub fn is_simd(&self) -> bool {
        match self {
            Register::SIMD(_) => true,
            _ => false,
        }
    }

    pub fn expect_simd(&self) -> usize {
        match self {
            Register::SIMD(r) => *r,
            _ => panic!("Expected SIMD, found {:?}", self),
        }
    }

    pub fn is_same_type_as(&self, other: &Register) -> bool {
        match (self, other) {
            (Register::GPR(_), Register::GPR(_)) => true,
            (Register::GPR(_), _) => false,

            (Register::SIMD(_), Register::SIMD(_)) => true,
            (Register::SIMD(_), _) => false,
        }
    }

    pub fn to_const_or_reg(&self) -> ConstOrReg {
        match self {
            Register::GPR(r) => ConstOrReg::GPR(*r as u32),
            Register::SIMD(r) => ConstOrReg::SIMD(*r as u32),
        }
    }
    pub fn is_volatile(&self) -> bool {
        is_register_volatile(*self)
    }

    pub fn size(&self) -> usize {
        match self {
            Register::GPR(_) => {
                #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
                return 8;
            }
            Register::SIMD(_) => {
                #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
                return 16; // 128-bit SIMD registers (AVX's YMM or ZMM registers are unsupported)
            }
        }
    }

    pub fn index(&self) -> usize {
        match self {
            Register::GPR(r) => *r,
            Register::SIMD(r) => *r,
        }
    }
}

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
    fn into_usage(&self, func: &IRFunctionInternal) -> Usage {
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
            Value::InstructionOutput {
                instruction_index,
                output_index,
                data_type,
                ..
            } => InputSlot::InstructionOutput {
                instruction_index: *instruction_index,
                output_index: *output_index,
                tp: *data_type,
            },
            Value::BlockInput {
                block_index,
                input_index,
                data_type,
            } => InputSlot::BlockInput {
                block_index: *block_index,
                input_index: *input_index,
                tp: *data_type,
            },
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
            (
                Value::InstructionOutput {
                    block_index,
                    instruction_index,
                    output_index,
                    ..
                },
                Value::InstructionOutput {
                    block_index: other_block_index,
                    instruction_index: other_instruction_index,
                    output_index: other_output_index,
                    ..
                },
            ) => {
                if block_index == other_block_index {
                    if instruction_index == other_instruction_index {
                        return output_index.cmp(other_output_index);
                    } else {
                        return instruction_index.cmp(other_instruction_index);
                    }
                } else {
                    return block_index.cmp(other_block_index);
                }
            }
            (
                Value::InstructionOutput { block_index, .. },
                Value::BlockInput {
                    block_index: other_block_index,
                    ..
                },
            ) => {
                if block_index == other_block_index {
                    // When both are in the same block, instruction outputs are always greater than
                    // block inputs
                    return std::cmp::Ordering::Greater;
                } else {
                    return block_index.cmp(other_block_index);
                }
            }
            (
                Value::BlockInput { block_index, .. },
                Value::InstructionOutput {
                    block_index: other_block_index,
                    ..
                },
            ) => {
                if block_index == other_block_index {
                    // When both are in the same block, instruction outputs are always greater than
                    // block inputs
                    return std::cmp::Ordering::Less;
                } else {
                    return block_index.cmp(other_block_index);
                }
            }
            (
                Value::BlockInput {
                    block_index,
                    input_index,
                    ..
                },
                Value::BlockInput {
                    block_index: other_block_index,
                    input_index: other_input_index,
                    ..
                },
            ) => {
                if block_index == other_block_index {
                    return input_index.cmp(other_input_index);
                } else {
                    return block_index.cmp(other_block_index);
                }
            }
        }
    }
}

impl InputSlot {
    fn references_value(&self, value: &Value) -> bool {
        match self {
            InputSlot::InstructionOutput {
                instruction_index,
                output_index,
                ..
            } => match value {
                Value::InstructionOutput {
                    instruction_index: v_instruction_index,
                    output_index: v_output_index,
                    ..
                } => *instruction_index == *v_instruction_index && *output_index == *v_output_index,
                _ => false,
            },
            InputSlot::BlockInput {
                block_index,
                input_index,
                ..
            } => match value {
                Value::BlockInput {
                    block_index: v_block_index,
                    input_index: v_input_index,
                    ..
                } => *block_index == *v_block_index && *input_index == *v_input_index,
                _ => false,
            },
            InputSlot::Constant(..) => false,
        }
    }

    pub fn to_value(self, func: &IRFunctionInternal) -> Option<Value> {
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
            }
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

    pub fn tp(&self) -> DataType {
        match self {
            InputSlot::InstructionOutput { tp, .. } => *tp,
            InputSlot::BlockInput { tp, .. } => *tp,
            InputSlot::Constant(c) => c.get_type(),
        }
    }

    pub fn expect_constant_round_type(&self) -> RoundType {
        match self {
            InputSlot::Constant(Constant::RoundType(rt)) => *rt,
            _ => {
                panic!("Expected a RoundType constant, found {:?}", self);
            }
        }
    }

    pub fn expect_constant_data_type(&self) -> DataType {
        if let InputSlot::Constant(Constant::DataType(data_type)) = self {
            *data_type
        } else {
            panic!("Expected data type constant, got {:?}", self);
        }
    }

    pub fn expect_constant_cmp_type(&self) -> CompareType {
        if let InputSlot::Constant(Constant::CompareType(cmp_type)) = self {
            *cmp_type
        } else {
            panic!("Expected compare type constant, got {:?}", self);
        }
    }

    pub fn expect_constant_u64(&self) -> u64 {
        if let InputSlot::Constant(c) = self {
            match c {
                Constant::U64(value) => *value,
                Constant::U32(value) => *value as u64,
                Constant::U8(value) => *value as u64,
                Constant::S64(value) if *value >= 0 => *value as u64,
                Constant::S16(value) if *value >= 0 => *value as u64,
                Constant::S8(value) if *value >= 0 => *value as u64,
                Constant::Ptr(value) => *value as u64,
                _ => panic!("Expected unsigned, positive, or ptr constant, got {:?}", self),
            }
        } else {
            panic!("Expected u64 constant, got {:?}", self);
        }
    }
}

struct IRFunctionValueIterator<'a> {
    pub function: &'a IRFunctionInternal,
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
                let result = Some(Value::BlockInput {
                    data_type: block.inputs[self.block_input_index],
                    block_index: self.block_index,
                    input_index: self.block_input_index,
                });
                self.block_input_index += 1;
                return result;
            }

            let fn_instruction_index = block.instructions[self.instruction_index];
            let instruction = &self.function.instructions[fn_instruction_index];

            match &instruction.instruction {
                Instruction::Instruction { outputs, .. } => {
                    let v = if self.output_index < outputs.len() {
                        let temp = Some(Value::InstructionOutput {
                            block_index: instruction.block_index,
                            instruction_index: instruction.index,
                            output_index: self.output_index,
                            data_type: outputs[self.output_index].tp,
                        });
                        self.output_index += 1;
                        return temp;
                    } else {
                        // We're out of range for our outputs, move to the next instruction
                        self.output_index = 0;
                        self.instruction_index += 1;
                        if self.instruction_index >= block.instructions.len() {
                            self.block_index += 1;
                            self.block_input_index = 0;
                            self.instruction_index = 0;
                        }

                        None
                    };

                    if v.is_some() {
                        return v;
                    }
                }

                // These aren't values, so skip
                Instruction::Branch { .. } | Instruction::Jump { .. } | Instruction::Return { .. } => {
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
                    write!(f, "v{}_{}(b{}):{}", instruction_index, output_index, block_index, data_type)
                }
            }
            Value::BlockInput {
                block_index,
                input_index,
                data_type,
            } => {
                write!(f, "b{}i{}:{}", block_index, input_index, data_type)
            }
        }
    }
}

#[derive(Debug, Eq, PartialEq, Clone, Copy)]
pub struct Usage {
    block_index: usize,
    instruction_index: usize,
    instruction_index_in_block: usize,
}

impl Usage {
    fn recalculate_index_in_block(&self, func: &IRFunctionInternal) -> Self {
        Usage {
            block_index: self.block_index,
            instruction_index: self.instruction_index,
            instruction_index_in_block: func
                .get_index_in_block(self.block_index, self.instruction_index)
                .unwrap(),
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

#[derive(Clone)]
pub struct Lifetimes {
    #[allow(dead_code)] // Maybe I'll need this later
    pub last_used: BTreeMap<Value, Usage>,
    pub interference: BTreeMap<Value, Vec<Value>>,
    /// A list of all usages of a value. Guaranteed to be sorted.
    pub all_usages: BTreeMap<Value, Vec<Usage>>,
}

impl Lifetimes {
    pub fn get_active_at_index(
        &self,
        func: &IRFunctionInternal,
        block_index: usize,
        instruction_index_in_block: usize,
    ) -> Vec<Value> {
        self.last_used
            .iter()
            .filter_map(|(value, last_usage)| {
                let first_usage = value.into_usage(func);

                let first_used_before = first_usage.block_index < block_index
                    || (first_usage.block_index == block_index
                        && first_usage.instruction_index_in_block <= instruction_index_in_block);

                let last_used_after = last_usage.block_index > block_index
                    || (last_usage.block_index == block_index
                        && last_usage.instruction_index_in_block >= instruction_index_in_block);

                if first_used_before && last_used_after {
                    Some(*value)
                } else {
                    None
                }
            })
            .collect()
    }
}

fn calculate_lifetimes(func: &IRFunctionInternal) -> Lifetimes {
    let mut last_used = BTreeMap::new();
    let mut all_usages: BTreeMap<Value, Vec<Usage>> = BTreeMap::new();
    let mut interference = BTreeMap::new();

    func.blocks
        .iter()
        .enumerate()
        .flat_map(|(block_index, block)| {
            block
                .instructions
                .iter()
                .enumerate()
                .map(move |(instruction_index_in_block, instruction_index)| {
                    (block_index, instruction_index_in_block, instruction_index)
                })
        })
        .for_each(|(block_index, instruction_index_in_block, instruction_index)| {
            match &func.instructions[*instruction_index].instruction {
                Instruction::Instruction { inputs, .. } => {
                    inputs.iter().map(|input| input.to_value(&func)).for_each(|input| {
                        if let Some(value) = input {
                            let u = Usage {
                                block_index,
                                instruction_index: *instruction_index,
                                instruction_index_in_block,
                            };
                            last_used.insert(value, u);
                            all_usages.entry(value).or_insert_with(Vec::new).push(u);
                        };
                    });
                }
                Instruction::Branch {
                    cond,
                    if_true,
                    if_false,
                } => {
                    if_true
                        .arguments
                        .iter()
                        .chain(if_false.arguments.iter())
                        .chain(iter::once(cond))
                        .flat_map(|i| i.to_value(&func))
                        .for_each(|value| {
                            let usage = Usage {
                                block_index,
                                instruction_index: *instruction_index,
                                instruction_index_in_block,
                            };
                            last_used.insert(value, usage);
                            all_usages.entry(value).or_insert_with(Vec::new).push(usage);
                        });
                }
                Instruction::Jump { target } => {
                    target
                        .arguments
                        .iter()
                        .flat_map(|i| i.to_value(&func))
                        .for_each(|value| {
                            let usage = Usage {
                                block_index,
                                instruction_index: *instruction_index,
                                instruction_index_in_block,
                            };
                            last_used.insert(value, usage);
                            all_usages.entry(value).or_insert_with(Vec::new).push(usage);
                        });
                }
                Instruction::Return { value } => value.into_iter().for_each(|input| {
                    if let Some(value) = input.to_value(&func) {
                        let u = Usage {
                            block_index,
                            instruction_index: *instruction_index,
                            instruction_index_in_block,
                        };
                        last_used.insert(value, u);
                        all_usages.entry(value).or_insert_with(Vec::new).push(u);
                    }
                }),
            }
        });

    last_used.keys().combinations(2).for_each(|x| {
        let a = x[0];
        let b = x[1];

        let a_first = a.into_usage(func);
        let b_first = b.into_usage(func);

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
        all_usages,
        interference,
    }
}

impl Display for Register {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Register::GPR(r) => write!(f, "GPR({})", r),
            Register::SIMD(r) => write!(f, "SIMD({})", r),
        }
    }
}

impl IRFunctionInternal {
    pub fn new_sized_stack_location(&mut self, bytes_needed: usize) -> usize {
        // Align the stack to this data type
        self.stack_bytes_used += bytes_needed - (self.stack_bytes_used % bytes_needed);

        return self.stack_bytes_used - bytes_needed;
    }
    pub fn new_stack_location(&mut self, tp: DataType) -> usize {
        return self.new_sized_stack_location(tp.size());
    }

    pub fn get_stack_offset_for_location(&self, location: u64, tp: DataType) -> u32 {
        (self.stack_bytes_used - location as usize - tp.size()) as u32
    }

    fn get_index_in_block(&self, block_index: usize, instruction_index: usize) -> Option<usize> {
        self.blocks[block_index]
            .instructions
            .iter()
            .position(|i| *i == instruction_index)
    }

    fn spill(&mut self, to_spill: &Value, final_usage_pre_spill: &Usage, usages_post_spill: Vec<&Usage>) {
        let to_spill_inputslot = to_spill.into_inputslot();

        let stack_location = self.new_stack_location(to_spill.data_type());

        let spill_instr_index = self.instructions.len();
        let spill_instr = IndexedInstruction {
            block_index: final_usage_pre_spill.block_index,
            index: spill_instr_index,
            instruction: Instruction::Instruction {
                tp: InstructionType::SpillToStack,
                inputs: vec![
                    to_spill_inputslot,
                    const_ptr(stack_location),
                    InputSlot::Constant(Constant::DataType(to_spill.data_type())),
                ],
                outputs: vec![],
            },
        };
        self.instructions.push(spill_instr);

        {
            let i = final_usage_pre_spill.instruction_index_in_block + 1; // Insert immediately after the last usage pre-spill
            self.blocks[final_usage_pre_spill.block_index]
                .instructions
                .splice(i..i, [spill_instr_index]);
        }

        // Now, insert a reload instruction before the first usage after the spill

        let first_usage_post_spill = usages_post_spill[0].recalculate_index_in_block(self);

        let reload_instr_index = self.instructions.len();
        let reload_instr = IndexedInstruction {
            block_index: first_usage_post_spill.block_index,
            index: reload_instr_index,
            instruction: Instruction::Instruction {
                tp: InstructionType::LoadFromStack,
                inputs: vec![const_ptr(stack_location)],
                outputs: vec![OutputSlot {
                    tp: to_spill.data_type(),
                }],
            },
        };
        self.instructions.push(reload_instr);

        {
            let i = first_usage_post_spill.instruction_index_in_block; // Insert immediately before the first usage post-spill
            self.blocks[first_usage_post_spill.block_index]
                .instructions
                .splice(i..i, [reload_instr_index]);
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
                }
                Instruction::Branch { cond, .. } => {
                    if cond.references_value(to_spill) {
                        *cond = reloaded_inputslot;
                    }
                }
                Instruction::Jump { .. } => {}
                Instruction::Return { value } => {
                    if let Some(v) = value {
                        if v.references_value(to_spill) {
                            *v = reloaded_inputslot;
                        }
                    }
                }
            }
        }
    }

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

#[derive(Clone)]
pub struct RegisterAllocations {
    pub allocations: BTreeMap<Value, Register>,
    pub callee_saved: Vec<(Register, usize)>,
    pub lifetimes: Lifetimes,
}
impl RegisterAllocations {
    pub fn get(&self, value: &Value) -> Option<Register> {
        self.allocations.get(value).map(|r| *r)
    }
}

/// Allocates registers for the given function. This will modify the function to spill values as
/// needed. Also calculates which callee-saved registers are needed and reserves space on the stack
/// for them.
pub fn alloc_for(func: &mut IRFunctionInternal) -> RegisterAllocations {
    let mut done = false;
    let mut allocations = BTreeMap::new();
    while !done {
        allocations.clear();
        let mut to_spill = None;
        let lifetimes = calculate_lifetimes(&func);

        for value in func.value_iter() {
            if !lifetimes
                .all_usages
                .get(&value)
                .map(|usages| usages.is_empty())
                .unwrap_or(true)
            {
                let interference = lifetimes.interference.get(&value);

                let mut found_reg = false;
                for reg in get_registers()
                    .iter()
                    .filter(|r| r.can_hold_datatype(value.data_type()))
                {
                    // Check if the register is already allocated to an interfering value
                    let already_allocated = interference.is_some()
                        && interference
                            .unwrap()
                            .iter()
                            .flat_map(|iv| allocations.get(iv))
                            .any(|r| r == reg);
                    if !already_allocated {
                        allocations.insert(value, *reg);
                        found_reg = true;
                        break;
                    }
                }

                if !found_reg {
                    to_spill = interference
                        .unwrap() // If we couldn't find a register, there must be interference
                        .iter()
                        // Limit to only values that have been allocated to registers, and only to
                        // registers that can hold the datatype of the value we're allocating
                        .filter(|iv| {
                            allocations
                                .get(iv)
                                .map(|r| r.can_hold_datatype(value.data_type()))
                                .is_some()
                        })
                        // Pair each interfering value with its next usage
                        .flat_map(|iv| {
                            lifetimes.all_usages[iv]
                                .iter()
                                .find(|u| u >= &&value.into_usage(&func))
                                .map(|u| (*u, *iv))
                        })
                        // Find the one with the farthest out next usage
                        .max_by(|(u1, _iv1), (u2, _iv2)| u1.cmp(u2))
                        .clone();
                    if to_spill.is_none() {
                        panic!("Couldn't find a value to spill!");
                    }
                    let (to_spill_next_used, _to_spill) = to_spill.unwrap();
                    if to_spill_next_used == value.into_usage(&func) {
                        panic!("Tried to spill a value next used at the same time as the one we're allocating (??? are we out of registers?)");
                    }
                    break;
                }
            }
        }

        if to_spill.is_none() {
            done = true;
        } else {
            let (to_spill_next_used, to_spill) = to_spill.unwrap();
            let to_spill_first_usage = to_spill.into_usage(&func);

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

    let callee_saved = allocations
        .iter()
        .map(|(_, reg)| *reg)
        .unique()
        .flat_map(|reg| {
            if !reg.is_volatile() {
                Some((reg, func.new_sized_stack_location(reg.size())))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    let lifetimes = calculate_lifetimes(&func);
    RegisterAllocations {
        allocations,
        callee_saved,
        lifetimes,
    }
}
