use std::{
    cmp::{max, min},
    collections::HashMap,
    fmt::Display,
    iter,
};

use crate::ir::{
    const_ptr, Constant, DataType, IRFunction, IndexedInstruction, InputSlot, Instruction, InstructionType, OutputSlot,
};

use itertools::Itertools;

#[derive(Hash, PartialEq, Eq, Debug, Clone, Copy)]
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

    pub fn is_simd(&self) -> bool {
        match self {
            Register::SIMD(_) => true,
            _ => false,
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
}

// X64
#[allow(dead_code)]
const RAX: Register = Register::GPR(0);
#[allow(dead_code)]
const RCX: Register = Register::GPR(1);
#[allow(dead_code)]
const RDX: Register = Register::GPR(2);
#[allow(dead_code)]
const RBX: Register = Register::GPR(3);
#[allow(dead_code)]
const RSP: Register = Register::GPR(4);
#[allow(dead_code)]
const RBP: Register = Register::GPR(5);
#[allow(dead_code)]
const RSI: Register = Register::GPR(6);
#[allow(dead_code)]
const RDI: Register = Register::GPR(7);
#[allow(dead_code)]
const R8: Register = Register::GPR(8);
#[allow(dead_code)]
const R9: Register = Register::GPR(9);
#[allow(dead_code)]
const R10: Register = Register::GPR(10);
#[allow(dead_code)]
const R11: Register = Register::GPR(11);
#[allow(dead_code)]
const R12: Register = Register::GPR(12);
#[allow(dead_code)]
const R13: Register = Register::GPR(13);
#[allow(dead_code)]
const R14: Register = Register::GPR(14);
#[allow(dead_code)]
const R15: Register = Register::GPR(15);
#[allow(dead_code)]
const XMM0: Register = Register::SIMD(0);
#[allow(dead_code)]
const XMM1: Register = Register::SIMD(1);
#[allow(dead_code)]
const XMM2: Register = Register::SIMD(2);
#[allow(dead_code)]
const XMM3: Register = Register::SIMD(3);
#[allow(dead_code)]
const XMM4: Register = Register::SIMD(4);
#[allow(dead_code)]
const XMM5: Register = Register::SIMD(5);
#[allow(dead_code)]
const XMM6: Register = Register::SIMD(6);
#[allow(dead_code)]
const XMM7: Register = Register::SIMD(7);
#[allow(dead_code)]
const XMM8: Register = Register::SIMD(8);
#[allow(dead_code)]
const XMM9: Register = Register::SIMD(9);
#[allow(dead_code)]
const XMM10: Register = Register::SIMD(10);
#[allow(dead_code)]
const XMM11: Register = Register::SIMD(11);
#[allow(dead_code)]
const XMM12: Register = Register::SIMD(12);
#[allow(dead_code)]
const XMM13: Register = Register::SIMD(13);
#[allow(dead_code)]
const XMM14: Register = Register::SIMD(14);
#[allow(dead_code)]
const XMM15: Register = Register::SIMD(15);
#[allow(dead_code)]
// AArch64
#[allow(dead_code)]
const X0: Register = Register::GPR(0);
#[allow(dead_code)]
const X1: Register = Register::GPR(1);
#[allow(dead_code)]
const X2: Register = Register::GPR(2);
#[allow(dead_code)]
const X3: Register = Register::GPR(3);
#[allow(dead_code)]
const X4: Register = Register::GPR(4);
#[allow(dead_code)]
const X5: Register = Register::GPR(5);
#[allow(dead_code)]
const X6: Register = Register::GPR(6);
#[allow(dead_code)]
const X7: Register = Register::GPR(7);
#[allow(dead_code)]
const X8: Register = Register::GPR(8);
#[allow(dead_code)]
const X9: Register = Register::GPR(9);
#[allow(dead_code)]
const X10: Register = Register::GPR(10);
#[allow(dead_code)]
const X11: Register = Register::GPR(11);
#[allow(dead_code)]
const X12: Register = Register::GPR(12);
#[allow(dead_code)]
const X13: Register = Register::GPR(13);
#[allow(dead_code)]
const X14: Register = Register::GPR(14);
#[allow(dead_code)]
const X15: Register = Register::GPR(15);
#[allow(dead_code)]
const X16: Register = Register::GPR(16);
#[allow(dead_code)]
const X17: Register = Register::GPR(17);
#[allow(dead_code)]
const X18: Register = Register::GPR(18);
#[allow(dead_code)]
const X19: Register = Register::GPR(19);
#[allow(dead_code)]
const X20: Register = Register::GPR(20);
#[allow(dead_code)]
const X21: Register = Register::GPR(21);
#[allow(dead_code)]
const X22: Register = Register::GPR(22);
#[allow(dead_code)]
const X23: Register = Register::GPR(23);
#[allow(dead_code)]
const X24: Register = Register::GPR(24);
#[allow(dead_code)]
const X25: Register = Register::GPR(25);
#[allow(dead_code)]
const X26: Register = Register::GPR(26);
#[allow(dead_code)]
const X27: Register = Register::GPR(27);
#[allow(dead_code)]
const X28: Register = Register::GPR(28);
#[allow(dead_code)]
const X29: Register = Register::GPR(29);
#[allow(dead_code)]
const X30: Register = Register::GPR(30);
#[allow(dead_code)]
const SP: Register = Register::GPR(31);

#[allow(dead_code)]
const V0: Register = Register::SIMD(0);
#[allow(dead_code)]
const V1: Register = Register::SIMD(1);
#[allow(dead_code)]
const V2: Register = Register::SIMD(2);
#[allow(dead_code)]
const V3: Register = Register::SIMD(3);
#[allow(dead_code)]
const V4: Register = Register::SIMD(4);
#[allow(dead_code)]
const V5: Register = Register::SIMD(5);
#[allow(dead_code)]
const V6: Register = Register::SIMD(6);
#[allow(dead_code)]
const V7: Register = Register::SIMD(7);
#[allow(dead_code)]
const V8: Register = Register::SIMD(8);
#[allow(dead_code)]
const V9: Register = Register::SIMD(9);
#[allow(dead_code)]
const V10: Register = Register::SIMD(10);
#[allow(dead_code)]
const V11: Register = Register::SIMD(11);
#[allow(dead_code)]
const V12: Register = Register::SIMD(12);
#[allow(dead_code)]
const V13: Register = Register::SIMD(13);
#[allow(dead_code)]
const V14: Register = Register::SIMD(14);
#[allow(dead_code)]
const V15: Register = Register::SIMD(15);
#[allow(dead_code)]
const V16: Register = Register::SIMD(16);
#[allow(dead_code)]
const V17: Register = Register::SIMD(17);
#[allow(dead_code)]
const V18: Register = Register::SIMD(18);
#[allow(dead_code)]
const V19: Register = Register::SIMD(19);
#[allow(dead_code)]
const V20: Register = Register::SIMD(20);
#[allow(dead_code)]
const V21: Register = Register::SIMD(21);
#[allow(dead_code)]
const V22: Register = Register::SIMD(22);
#[allow(dead_code)]
const V23: Register = Register::SIMD(23);
#[allow(dead_code)]
const V24: Register = Register::SIMD(24);
#[allow(dead_code)]
const V25: Register = Register::SIMD(25);
#[allow(dead_code)]
const V26: Register = Register::SIMD(26);
#[allow(dead_code)]
const V27: Register = Register::SIMD(27);
#[allow(dead_code)]
const V28: Register = Register::SIMD(28);
#[allow(dead_code)]
const V29: Register = Register::SIMD(29);
#[allow(dead_code)]
const V30: Register = Register::SIMD(30);
#[allow(dead_code)]
const V31: Register = Register::SIMD(31);

fn get_registers() -> Vec<Register> {
    // Callee-saved registers
    #[cfg(target_arch = "aarch64")]
    {
        vec![
            X19, X20, X21, X22, X23, X24, X25, X26, X27, X28, V8, V9, V10, V11, V12, V13, V14, V15,
        ]
    }

    // For x64, it matters whether we're on Linux or Windows
    #[cfg(target_arch = "x86_64")]
    {
        // rbx, rsp, rbp, r12, r13, r14, r15
        // but we can't use rbp and rsp - so we just have to use rbx, r12, r13, r14, r15
        #[cfg(target_os = "linux")]
        {
            vec![
                RBX, R12, R13, R14, R15,
                // These aren't callee-saved, but none are on this ABI. So, we make do.
                // Use the regs not also used for function arguments.
                XMM8, XMM9, XMM10, XMM11, XMM12, XMM13, XMM14, XMM15,
            ]
        }
        #[cfg(target_os = "windows")]
        {
            todo!("Preserved regs on x64 Windows")
        }
    }
}

pub fn get_scratch_registers() -> Vec<Register> {
    #[cfg(target_arch = "aarch64")]
    {
        vec![
            X9, X10, X11, X12, X13, X14, X15, V16, V17, V18, V19, V20, V21, V22, V23, V24, V25, V26, V27, V28, V29,
            V30, V31,
        ]
    }

    // For x64, it matters whether we're on Linux or Windows
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(target_os = "linux")]
        {
            vec![
                RAX, RDI, RSI, RDX, RCX, R8, R9, R10, R11, XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7,
            ]
        }
        #[cfg(target_os = "windows")]
        {
            todo!("Scratch regs on x64 Windows")
        }
    }
}

pub fn get_function_argument_registers() -> Vec<Register> {
    #[cfg(target_arch = "aarch64")]
    {
        vec![X0, X1, X2, X3, X4, X5, X6, X7, V0, V1, V2, V3, V4, V5, V6, V7]
    }
    // For x64, it matters whether we're on Linux or Windows
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(target_os = "linux")]
        {
            vec![
                RDI, RSI, RDX, RCX, R8, R9, XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7,
            ]
        }
        #[cfg(target_os = "windows")]
        {
            vec![RCX, RDX, R8, R9] // TODO SIMD registers
        }
    }
}

pub fn get_return_value_registers() -> Vec<Register> {
    #[cfg(target_arch = "aarch64")]
    {
        // Technically, it's x0-x7,v0-v7, but we only support returning one value
        return vec![X0, V0];
    }
    #[cfg(target_arch = "x86_64")]
    {
        vec![RAX, XMM0]
    }
}

impl Register {
    pub fn is_volatile(&self) -> bool {
        match *self {
            Register::GPR(r) => {
                #[cfg(target_arch = "aarch64")]
                {
                    r < 19 || r > 28
                }
                #[cfg(target_arch = "x86_64")]
                {
                    #[cfg(target_os = "linux")]
                    {
                        // rax, rdi, rsi, rdx, rcx, r8, r9, r10, r11
                        r == 0 || r == 7 || r == 6 || r == 2 || r == 1 || r == 8 || r == 9 || r == 10 || r == 11
                    }
                }
            }
            Register::SIMD(_r) => {
                #[cfg(target_arch = "aarch64")]
                {
                    _r < 8 || _r > 15
                }
                #[cfg(target_arch = "x86_64")]
                {
                    #[cfg(target_os = "linux")]
                    {
                        true // All SIMD registers are volatile in SYSTEM-V
                    }
                }
            }
        }
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
                    write!(f, "v{}(b{})o{}:{}", instruction_index, block_index, output_index, data_type)
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
struct Usage {
    block_index: usize,
    instruction_index: usize,
    instruction_index_in_block: usize,
}

impl Usage {
    fn recalculate_index_in_block(&self, func: &IRFunction) -> Self {
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

struct Lifetimes {
    #[allow(dead_code)] // Maybe I'll need this later
    last_used: HashMap<Value, Usage>,
    interference: HashMap<Value, Vec<Value>>,
    /// A list of all usages of a value. Guaranteed to be sorted.
    all_usages: HashMap<Value, Vec<Usage>>,
}

fn calculate_lifetimes(func: &IRFunction) -> Lifetimes {
    let mut last_used = HashMap::new();
    let mut all_usages: HashMap<Value, Vec<Usage>> = HashMap::new();
    let mut interference = HashMap::new();

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

impl IRFunction {
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

pub struct RegisterAllocations {
    pub allocations: HashMap<Value, Register>,
    pub callee_saved: Vec<(Register, usize)>,
}
impl RegisterAllocations {
    pub fn get(&self, value: &Value) -> Option<Register> {
        self.allocations.get(value).map(|r| *r)
    }
}

/// Allocates registers for the given function. This will modify the function to spill values as
/// needed. Also calculates which callee-saved registers are needed and reserves space on the stack
/// for them.
pub fn alloc_for(func: &mut IRFunction) -> RegisterAllocations {
    let mut done = false;
    let mut allocations = HashMap::new();
    while !done {
        allocations.clear();
        let mut to_spill = None;
        let lifetimes = calculate_lifetimes(&func);

        for value in func.value_iter() {
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
                            .find(|u| u >= &&value.into_usage(func))
                            .map(|u| (*u, *iv))
                    })
                    // Find the one with the farthest out next usage
                    .max_by(|(u1, _iv1), (u2, _iv2)| u1.cmp(u2))
                    .clone();
                if to_spill.is_none() {
                    panic!("Couldn't find a value to spill!");
                }
                let (to_spill_next_used, _to_spill) = to_spill.unwrap();
                if to_spill_next_used == value.into_usage(func) {
                    panic!("Tried to spill a value next used at the same time as the one we're allocating (??? are we out of registers?)");
                }
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

    RegisterAllocations {
        allocations,
        callee_saved,
    }
}
