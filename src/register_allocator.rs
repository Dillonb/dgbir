use std::{collections::BTreeMap, fmt::Display};

use crate::{
    abi::is_register_volatile,
    compiler::ConstOrReg,
    ir::{CompareType, Constant, DataType, IRFunctionInternal, InputSlot, MultiplyType, PackType, RoundingMode, VectorHalf},
};

mod linear_scan;

pub type RegisterIndex = u8;

#[derive(Hash, PartialEq, Eq, Debug, Clone, Copy, PartialOrd, Ord)]
pub enum Register {
    GPR(RegisterIndex),
    SIMD(RegisterIndex),
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
                const VALID_SIMD_TYPES: [DataType; 4] = [DataType::F32, DataType::F64, DataType::U128, DataType::S128];
                return tp.is_vector() || VALID_SIMD_TYPES.contains(&tp);
            }
        }
    }

    pub fn is_gpr(&self) -> bool {
        match self {
            Register::GPR(_) => true,
            _ => false,
        }
    }

    pub fn expect_gpr(&self) -> RegisterIndex {
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

    pub fn expect_simd(&self) -> RegisterIndex {
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
            Register::GPR(r) => ConstOrReg::GPR(*r),
            Register::SIMD(r) => ConstOrReg::SIMD(*r),
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

    pub fn index(&self) -> RegisterIndex {
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

    pub fn expect_constant_vector_half(&self) -> VectorHalf {
        match self {
            InputSlot::Constant(Constant::VectorHalf(h)) => *h,
            _ => {
                panic!("Expected a VectorHalf constant, found {:?}", self);
            }
        }
    }

    pub fn expect_constant_pack_type(&self) -> PackType {
        match self {
            InputSlot::Constant(Constant::PackType(p)) => *p,
            _ => {
                panic!("Expected a PackType constant, found {:?}", self);
            }
        }
    }

    pub fn expect_constant_multiply_type(&self) -> MultiplyType {
        match self {
            InputSlot::Constant(Constant::MultiplyType(mt)) => *mt,
            _ => {
                panic!("Expected a MultiplyType constant, found {:?}", self);
            }
        }
    }

    pub fn expect_constant_rounding_mode(&self) -> RoundingMode {
        match self {
            InputSlot::Constant(Constant::RoundingMode(rt)) => *rt,
            _ => {
                panic!("Expected a RoundingMode constant, found {:?}", self);
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

    pub fn block_referenced(&self, func: &IRFunctionInternal) -> Option<usize> {
        match self {
            InputSlot::InstructionOutput { instruction_index, .. } => {
                Some(func.instructions[*instruction_index].block_index)
            }
            InputSlot::BlockInput { block_index, .. } => Some(*block_index),
            InputSlot::Constant(_) => None,
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
    /// The block this usage is in.
    pub fn block_index(&self) -> usize {
        self.block_index
    }

    /// The index of the using instruction within its block.
    pub fn instruction_index_in_block(&self) -> usize {
        self.instruction_index_in_block
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

/// Where every value is live, in the allocator's linearized program-point space.
///
/// This exists solely to answer [`Lifetimes::get_active_at_index`]; the representation is private,
/// so a different allocator is free to store whatever it needs behind that method.
#[derive(Clone)]
pub struct Lifetimes {
    /// Live ranges over the linearized program-point space, sorted by `start`.
    pub(crate) live_ranges: Vec<LiveRange>,
    /// `block_starts[b]` is the program point of the first instruction of block `b`.
    pub(crate) block_starts: Vec<u32>,
}

/// A half-open-free (i.e. inclusive on both ends) live range in the linearized program point space.
#[derive(Clone, Copy)]
pub(crate) struct LiveRange {
    pub(crate) start: u32,
    pub(crate) end: u32,
    /// The program point at which this value is defined, or `u32::MAX` for block inputs.
    pub(crate) def: u32,
    pub(crate) value: Value,
}

impl Lifetimes {
    /// All values that are live across the given instruction, excluding any value that is defined
    /// by that very instruction.
    ///
    /// Used by the compiler to decide which volatile registers must be preserved around a call.
    pub fn get_active_at_index(
        &self,
        _func: &IRFunctionInternal,
        block_index: usize,
        instruction_index_in_block: usize,
    ) -> Vec<Value> {
        let Some(block_start) = self.block_starts.get(block_index) else {
            return Vec::new();
        };
        let point = block_start.saturating_add(instruction_index_in_block as u32);

        // `live_ranges` is sorted by `start`, so everything that could contain `point` lies in the
        // prefix ending at the first range starting after it.
        let upper = self.live_ranges.partition_point(|r| r.start <= point);

        let mut out = Vec::new();
        for range in &self.live_ranges[..upper] {
            if range.end >= point && range.def != point {
                out.push(range.value);
            }
        }
        out.sort();
        out
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
        assert!(bytes_needed.is_power_of_two(), "Stack allocations must be a power of two in size");

        // Align to the size of the allocation
        self.stack_bytes_used = (self.stack_bytes_used + bytes_needed - 1) & !(bytes_needed - 1);

        let location_offset = self.stack_bytes_used;
        self.stack_bytes_used += bytes_needed;
        return location_offset;
    }

    pub fn new_stack_location(&mut self, tp: DataType) -> usize {
        return self.new_sized_stack_location(tp.size());
    }

    pub fn get_stack_offset_for_location(&self, location: u64, tp: DataType) -> u32 {
        (self.stack_bytes_used - location as usize - tp.size()) as u32
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
    linear_scan::regalloc(func)
}
