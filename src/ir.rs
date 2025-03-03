use std::cell::RefCell;
// #[macro_use] extern crate maplit;

#[derive(Debug, Clone)]
pub enum DataType {
    U8,
    S8,
    U16,
    S16,
    U32,
    S32,
    U64,
    S64,
    F32,
    F64,
    Ptr,
    Flags,
}

#[derive(Debug)]
pub enum Constant {
    U8(u8),
    S8(i8),
    U16(u16),
    S16(i16),
    U32(u32),
    S32(i32),
    U64(u64),
    S64(i64),
    F32(f32),
    F64(f64),
    Ptr(usize),
}

#[derive(Debug)]
pub enum InstructionType {
    Add,
    LoadPtr,
}

#[derive(Debug)]
pub enum InputSlot {
    /// References the output of another instruction.
    InstructionOutput {
        instruction_index: usize,
        output_index: usize,
    },
    Constant(Constant),
    DataType(DataType),
}

pub struct InputSlotSchema {
    name: String,
}

pub struct InstructionSchema {
    inputs: Vec<InputSlotSchema>,
}

#[derive(Debug)]
pub struct Instruction {
    /// The index of the instruction in the IR vec
    index: usize,
    instruction_type: InstructionType,
    inputs: Vec<InputSlot>,
}

#[derive(Debug)]
pub struct IRContext {
    /// Memory addresses available to the IR
    inputs: Vec<usize>,
}

pub struct IRBlock {
    context: RefCell<IRContext>,
    instructions: Vec<Instruction>,
}
