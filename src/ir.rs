use colored::Colorize;
use std::{
    cell::RefCell,
    ops::{Index, IndexMut},
};

#[derive(Debug, Clone, Copy)]
pub enum DataType {
    // None,
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
    Bool,
    Ptr,
}

#[derive(Debug, Clone, Copy)]
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
    Bool(bool),
    DataType(DataType),
    CompareType(CompareType),
}

#[derive(Debug, Clone, Copy)]
pub enum CompareType {
    Equal,
    NotEqual,
    LessThanSigned,
    GreaterThanSigned,
    LessThanOrEqualSigned,
    GreaterThanOrEqualSigned,
    LessThanUnsigned,
    GreaterThanUnsigned,
    LessThanOrEqualUnsigned,
    GreaterThanOrEqualUnsigned,
}

#[derive(Debug)]
pub enum InstructionType {
    Add,
    Compare,
    LoadPtr,
    WritePtr,
}

#[derive(Debug, Clone, Copy)]
pub enum InputSlot {
    /// References the output of another instruction.
    InstructionOutput {
        block_index: usize,
        instruction_index: usize,
        tp: DataType,
        output_index: usize,
    },
    /// References an input to the block.
    BlockInput {
        block_index: usize,
        input_index: usize,
        tp: DataType,
    },
    Constant(Constant),
}
impl std::fmt::Display for InputSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InputSlot::InstructionOutput {
                block_index,
                instruction_index,
                tp,
                output_index,
            } => writeln!(
                f,
                "block_{}_i{}: type: {:?}, out_indx:{}",
                block_index, instruction_index, tp, output_index
            )?,
            InputSlot::BlockInput {
                block_index,
                input_index,
                tp,
            } => writeln!(
                f,
                "block_{}, input_index {}, type: {:?}",
                block_index, input_index, tp
            )?,
            InputSlot::Constant(constant) => writeln!(f, "{:?}", constant)?,
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct OutputSlot {
    pub tp: DataType,
}

#[derive(Debug, Clone)]
pub struct BlockReference {
    pub block_index: usize,
    pub arguments: Vec<InputSlot>,
}

#[derive(Debug)]
pub enum Instruction {
    Instruction {
        tp: InstructionType,
        inputs: Vec<InputSlot>,
        outputs: Vec<OutputSlot>,
    },
    Branch {
        cond: InputSlot,
        if_true: BlockReference,
        if_false: BlockReference,
    },
    Jump {
        target: BlockReference,
    },
    Return {
        value: Option<InputSlot>,
    },
}

impl std::fmt::Display for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Instruction::Instruction {
                tp,
                inputs,
                outputs,
            } => {
                writeln!(f, "type: {:?}", tp)?;
                writeln!(f, "inputs:")?;
                for i in inputs {
                    writeln!(f, "{i}")?;
                }
                writeln!(f, "outputs:")?;
                for o in outputs {
                    writeln!(f, "{:?}", o)?;
                }
            }
            Instruction::Branch {
                cond,
                if_true,
                if_false,
            } => writeln!(f, "{:?},\n{:?},\n{:?}", cond, if_true, if_false)?,
            Instruction::Jump { target } => writeln!(f, "{:?}", target)?,
            Instruction::Return { value } => writeln!(f, "{:?}", value)?,
            _ => panic!("you forgot to implement display branch arm for this instruction"),
        }

        Ok(())
    }
}

#[derive(Debug)]
pub struct InstructionOutput {
    outputs: Vec<InputSlot>,
}

impl InstructionOutput {
    /// Gets a specific output
    pub fn at(&self, index: usize) -> InputSlot {
        self.outputs[index]
    }

    /// Convenience function to get the first output
    pub fn val(&self) -> InputSlot {
        self.outputs[0]
    }
}

pub fn const_u32(value: u32) -> InputSlot {
    InputSlot::Constant(Constant::U32(value))
}

pub fn const_f32(value: f32) -> InputSlot {
    InputSlot::Constant(Constant::F32(value))
}

pub fn const_ptr(value: usize) -> InputSlot {
    InputSlot::Constant(Constant::Ptr(value))
}

#[derive(Debug)]
pub struct IndexedInstruction {
    /// Which block this instruction belongs to
    pub block_index: usize,
    /// The index of the instruction in the IR vec
    pub index: usize,
    pub instruction: Instruction,
}
impl std::fmt::Display for IndexedInstruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", self.instruction)?;
        Ok(())
    }
}

#[derive(Debug)]
pub struct IRContext {
    /// Memory addresses available to the IR
    pub inputs: Vec<usize>,
}

#[derive(Debug)]
pub struct IRFunction {
    pub blocks: Vec<IRBasicBlock>,
}

impl std::fmt::Display for IRFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, b) in self.blocks.iter().enumerate() {
            writeln!(
                f,
                "{}\n{}\n{}\n",
                format!("═══════════block_{i}══════════").red().on_green(),
                b,
                format!("══════════════════════════════").red().on_green()
            )?
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct IRBasicBlock {
    pub is_closed: bool,
    pub index: usize,
    pub inputs: Vec<DataType>,
    pub instructions: Vec<IndexedInstruction>,
}

impl std::fmt::Display for IRBasicBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_closed {
            writeln!(f, "{}", format!("closed").red())?
        } else {
            writeln!(f, "{}", format!("open").green())?
        }

        writeln!(f, "{}", format!("block inputs: ").magenta())?;
        for i in self.inputs.iter() {
            writeln!(f, "┌──────────────────┐")?;
            writeln!(f, "{:?},", i)?;
            writeln!(f, "└──────────────────┘")?;
        }

        writeln!(f, "{}", format!("list of instructions:").magenta())?;
        for i in self.instructions.iter() {
            writeln!(f, "┌────────────────────────────┐")?;
            writeln!(f, "{i}")?;
            writeln!(f, "└────────────────────────────┘")?;
        }

        Ok(())
    }
}

pub struct IRBlockHandle {
    pub index: usize,
    pub inputs: Vec<DataType>,
}

impl IRBlockHandle {
    pub fn call(&self, vec: Vec<InputSlot>) -> BlockReference {
        BlockReference {
            block_index: self.index,
            arguments: vec,
        }
    }

    /// Gets an input from a block to be used in another instruction
    pub fn input(&self, i: usize) -> InputSlot {
        InputSlot::BlockInput {
            block_index: self.index,
            input_index: i,
            tp: self.inputs[i],
        }
    }
}

impl IRContext {
    pub fn new() -> RefCell<Self> {
        Into::into(IRContext { inputs: Vec::new() })
    }
}

impl IRFunction {
    pub fn new(_context: RefCell<IRContext>) -> Self {
        IRFunction { blocks: Vec::new() }
    }

    pub fn new_block(&mut self, inputs: Vec<DataType>) -> IRBlockHandle {
        let index = self.blocks.len();
        self.blocks.push(IRBasicBlock {
            is_closed: false,
            index,
            inputs: inputs.clone(),
            instructions: Vec::new(),
        });
        return IRBlockHandle { index, inputs };
    }

    pub fn validate(&self) {
        for block in &self.blocks {
            if !block.is_closed {
                panic!("Unclosed block: block_{}", block.index);
            }
        }
    }
}

impl Index<&IRBlockHandle> for IRFunction {
    type Output = IRBasicBlock;
    fn index(&self, index: &IRBlockHandle) -> &Self::Output {
        &self.blocks[index.index]
    }
}

impl IndexMut<&IRBlockHandle> for IRFunction {
    fn index_mut(&mut self, index: &IRBlockHandle) -> &mut Self::Output {
        &mut self.blocks[index.index]
    }
}

impl IRBasicBlock {
    pub fn append_obj(&mut self, instruction: Instruction) -> usize {
        if self.is_closed {
            panic!("Cannot append to a closed block");
        }

        let index = self.instructions.len();

        // Close the block if necessary
        match instruction {
            Instruction::Branch { .. } => {
                self.is_closed = true;
            }
            Instruction::Jump { .. } => {
                self.is_closed = true;
            }
            Instruction::Return { .. } => {
                self.is_closed = true;
            }
            Instruction::Instruction { .. } => {}
        }

        self.instructions.push(IndexedInstruction {
            block_index: self.index,
            index,
            instruction,
        });
        return index;
    }

    pub fn append(
        &mut self,
        tp: InstructionType,
        inputs: Vec<InputSlot>,
        outputs: Vec<OutputSlot>,
    ) -> InstructionOutput {
        let index = self.append_obj(Instruction::Instruction {
            tp,
            inputs,
            outputs: outputs.clone(),
        });

        return InstructionOutput {
            outputs: outputs
                .iter()
                .enumerate()
                .map(|(i, output)| InputSlot::InstructionOutput {
                    block_index: self.index,
                    instruction_index: index,
                    tp: output.tp,
                    output_index: i,
                })
                .collect(),
        };
    }

    pub fn add(
        &mut self,
        result_tp: DataType,
        arg1: InputSlot,
        arg2: InputSlot,
    ) -> InstructionOutput {
        self.append(
            InstructionType::Add,
            vec![arg1, arg2],
            vec![OutputSlot { tp: result_tp }],
        )
    }

    pub fn write_ptr(
        &mut self,
        tp: DataType,
        ptr: InputSlot,
        value: InputSlot,
    ) -> InstructionOutput {
        self.append(
            InstructionType::WritePtr,
            vec![ptr, value, InputSlot::Constant(Constant::DataType(tp))],
            vec![],
        );
        return InstructionOutput { outputs: vec![] };
    }

    pub fn compare(&mut self, x: InputSlot, tp: CompareType, y: InputSlot) -> InstructionOutput {
        self.append(
            InstructionType::Compare,
            vec![x, InputSlot::Constant(Constant::CompareType(tp)), y],
            vec![OutputSlot { tp: DataType::Bool }],
        )
    }

    pub fn branch(&mut self, cond: InputSlot, if_true: BlockReference, if_false: BlockReference) {
        self.append_obj(Instruction::Branch {
            cond,
            if_true,
            if_false,
        });
    }

    pub fn jump(&mut self, target: BlockReference) {
        self.append_obj(Instruction::Jump { target });
    }

    pub fn ret(&mut self, input: Option<InputSlot>) {
        self.append_obj(Instruction::Return { value: input });
    }
}
