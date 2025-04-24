use std::cell::RefCell;

mod ir_display;

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

#[derive(Debug)]
pub struct IRContext {
    /// Memory addresses available to the IR
    pub inputs: Vec<usize>,
}

#[derive(Debug)]
pub struct IRFunction {
    pub blocks: Vec<IRBasicBlock>,
}

#[derive(Debug)]
pub struct IRBasicBlock {
    pub is_closed: bool,
    pub index: usize,
    pub inputs: Vec<DataType>,
    pub instructions: Vec<IndexedInstruction>,
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

    pub fn append_obj(&mut self, block_handle: &IRBlockHandle, instruction: Instruction) -> usize {
        let block = &mut self.blocks[block_handle.index];
        if block.is_closed {
            panic!("Cannot append to a closed block");
        }

        let index = block.instructions.len();

        // Close the block if necessary
        match instruction {
            Instruction::Branch { .. } => {
                block.is_closed = true;
            }
            Instruction::Jump { .. } => {
                block.is_closed = true;
            }
            Instruction::Return { .. } => {
                block.is_closed = true;
            }
            Instruction::Instruction { .. } => {}
        }

        block.instructions.push(IndexedInstruction {
            block_index: block.index,
            index,
            instruction,
        });
        return index;
    }

    pub fn append(
        &mut self,
        block_handle: &IRBlockHandle,
        tp: InstructionType,
        inputs: Vec<InputSlot>,
        outputs: Vec<OutputSlot>,
    ) -> InstructionOutput {
        let index = self.append_obj(
            block_handle,
            Instruction::Instruction {
                tp,
                inputs,
                outputs: outputs.clone(),
            },
        );

        let block = &mut self.blocks[block_handle.index];

        return InstructionOutput {
            outputs: outputs
                .iter()
                .enumerate()
                .map(|(i, output)| InputSlot::InstructionOutput {
                    block_index: block.index,
                    instruction_index: index,
                    tp: output.tp,
                    output_index: i,
                })
                .collect(),
        };
    }

    pub fn add(
        &mut self,
        block_handle: &IRBlockHandle,
        result_tp: DataType,
        arg1: InputSlot,
        arg2: InputSlot,
    ) -> InstructionOutput {
        self.append(
            block_handle,
            InstructionType::Add,
            vec![arg1, arg2],
            vec![OutputSlot { tp: result_tp }],
        )
    }

    pub fn write_ptr(
        &mut self,
        block_handle: &IRBlockHandle,
        tp: DataType,
        ptr: InputSlot,
        value: InputSlot,
    ) -> InstructionOutput {
        self.append(
            block_handle,
            InstructionType::WritePtr,
            vec![ptr, value, InputSlot::Constant(Constant::DataType(tp))],
            vec![],
        );
        return InstructionOutput { outputs: vec![] };
    }

    pub fn compare(
        &mut self,
        block_handle: &IRBlockHandle,
        x: InputSlot,
        tp: CompareType,
        y: InputSlot,
    ) -> InstructionOutput {
        self.append(
            block_handle,
            InstructionType::Compare,
            vec![x, InputSlot::Constant(Constant::CompareType(tp)), y],
            vec![OutputSlot { tp: DataType::Bool }],
        )
    }

    pub fn branch(
        &mut self,
        block_handle: &IRBlockHandle,
        cond: InputSlot,
        if_true: BlockReference,
        if_false: BlockReference,
    ) {
        self.append_obj(
            block_handle,
            Instruction::Branch {
                cond,
                if_true,
                if_false,
            },
        );
    }

    pub fn jump(&mut self, block_handle: &IRBlockHandle, target: BlockReference) {
        self.append_obj(block_handle, Instruction::Jump { target });
    }

    pub fn ret(&mut self, block_handle: &IRBlockHandle, input: Option<InputSlot>) {
        self.append_obj(block_handle, Instruction::Return { value: input });
    }
}
