use itertools::Itertools;
use petgraph::{
    algo::dominators::{self, Dominators},
    graph::{Graph, NodeIndex},
};
use std::{cell::RefCell, collections::HashSet, rc::Rc};

use ordered_float::OrderedFloat;

mod ir_display;
mod ir_emitters;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
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
    U128,
    F32,
    F64,
    Bool,
    Ptr,
}

impl DataType {
    pub fn size(&self) -> usize {
        match self {
            DataType::U8 => 1,
            DataType::S8 => 1,
            DataType::U16 => 2,
            DataType::S16 => 2,
            DataType::U32 => 4,
            DataType::S32 => 4,
            DataType::U64 => 8,
            DataType::S64 => 8,
            DataType::U128 => 16,
            DataType::F32 => 4,
            DataType::F64 => 8,
            DataType::Bool => 1,
            #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
            DataType::Ptr => 8,
        }
    }

    pub fn is_signed(&self) -> bool {
        match self {
            // Integers
            DataType::S8 | DataType::S16 | DataType::S32 | DataType::S64 => true,
            DataType::U8 | DataType::U16 | DataType::U32 | DataType::U64 | DataType::U128 => false,

            // Consider floats to be signed since unsigned floats don't exist
            DataType::F32 | DataType::F64 => true,

            // ???
            DataType::Bool => false,
            DataType::Ptr => false,
        }
    }

    pub fn is_integer(&self) -> bool {
        match self {
            DataType::U8
            | DataType::S8
            | DataType::U16
            | DataType::S16
            | DataType::U32
            | DataType::S32
            | DataType::U64
            | DataType::S64
            | DataType::U128 => true,
            DataType::F32 | DataType::F64 => false,
            DataType::Bool => true,
            DataType::Ptr => true,
        }
    }

    pub fn is_float(&self) -> bool {
        match self {
            DataType::F32 | DataType::F64 => true,
            _ => false,
        }
    }

    fn half_type(&self) -> DataType {
        match self {
            DataType::U8 | DataType::S8 => panic!("Cannot take half type of U8 or S8"),
            DataType::U16 => DataType::U8,
            DataType::S16 => DataType::S8,
            DataType::U32 => DataType::U16,
            DataType::S32 => DataType::S16,
            DataType::U64 => DataType::U32,
            DataType::S64 => DataType::S32,
            DataType::U128 => DataType::U64,
            DataType::F32 => panic!("Cannot take half type of F32"),
            DataType::F64 => DataType::F32,
            DataType::Bool => panic!("Cannot take half type of Bool"),
            DataType::Ptr => panic!("Cannot take half type of Ptr"),
        }
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum Constant {
    U8(u8),
    S8(i8),
    U16(u16),
    S16(i16),
    U32(u32),
    S32(i32),
    U64(u64),
    S64(i64),
    F32(OrderedFloat<f32>),
    F64(OrderedFloat<f64>),
    Ptr(usize),
    Bool(bool),
    DataType(DataType),
    CompareType(CompareType),
    RoundingMode(RoundingMode),
}

impl Constant {
    pub fn get_type(&self) -> DataType {
        match self {
            Constant::U8(_) => DataType::U8,
            Constant::S8(_) => DataType::S8,
            Constant::U16(_) => DataType::U16,
            Constant::S16(_) => DataType::S16,
            Constant::U32(_) => DataType::U32,
            Constant::S32(_) => DataType::S32,
            Constant::U64(_) => DataType::U64,
            Constant::S64(_) => DataType::S64,
            Constant::F32(_) => DataType::F32,
            Constant::F64(_) => DataType::F64,
            Constant::Ptr(_) => DataType::Ptr,
            Constant::Bool(_) => DataType::Bool,
            _ => panic!("Invalid constant type"),
        }
    }

    pub fn size(&self) -> usize {
        self.get_type().size()
    }

    pub fn into_inputslot(&self) -> InputSlot {
        InputSlot::Constant(*self)
    }
}

pub enum MultiplyType {
    Split,
    Combined,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum CompareType {
    Equal,
    NotEqual,
    LessThan,
    GreaterThan,
    LessThanOrEqual,
    GreaterThanOrEqual,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum RoundingMode {
    Up,
    Down,
    Nearest,
    Truncate,
}

#[derive(Debug)]
pub enum InstructionType {
    Add,
    LeftShift,
    RightShift,
    Compare,
    LoadPtr,
    WritePtr,
    SpillToStack,
    LoadFromStack,
    Convert,
    And,
    Or,
    Not,
    Xor,
    Subtract,
    Multiply,
    Divide,
    SquareRoot,
    AbsoluteValue,
    Negate,
    CallFunction,
}

#[derive(Debug, Clone, Copy)]
pub enum InputSlot {
    /// References the output of another instruction.
    InstructionOutput {
        /// The index of the instruction in the FUNCTION
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
    #[cfg(feature = "ir_comments")]
    Comment(String),
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

impl Instruction {
    pub fn all_inputslots(&self) -> Vec<InputSlot> {
        match self {
            Instruction::Comment(_) => vec![],
            Instruction::Instruction { inputs, .. } => inputs.clone(),
            Instruction::Branch {
                cond,
                if_true,
                if_false,
            } => {
                let mut slots = vec![*cond];
                slots.extend(if_true.arguments.iter());
                slots.extend(if_false.arguments.iter());
                slots
            }
            Instruction::Jump { target } => target.arguments.clone(),
            Instruction::Return { value } => value.iter().cloned().collect(),
        }
    }
    pub fn references_values_in_blocks(&self, func: &IRFunctionInternal) -> Vec<usize> {
        self.all_inputslots()
            .iter()
            .flat_map(|slot| slot.block_referenced(func))
            .dedup()
            .collect_vec()
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

pub fn const_u16(value: u16) -> InputSlot {
    InputSlot::Constant(Constant::U16(value))
}

pub fn const_s16(value: i16) -> InputSlot {
    InputSlot::Constant(Constant::S16(value))
}

pub fn const_u32(value: u32) -> InputSlot {
    InputSlot::Constant(Constant::U32(value))
}

pub fn const_s32(value: i32) -> InputSlot {
    InputSlot::Constant(Constant::S32(value))
}

pub fn const_u64(value: u64) -> InputSlot {
    InputSlot::Constant(Constant::U64(value))
}

pub fn const_s64(value: i64) -> InputSlot {
    InputSlot::Constant(Constant::S64(value))
}

pub fn const_f32(value: f32) -> InputSlot {
    InputSlot::Constant(Constant::F32(OrderedFloat(value)))
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
pub struct IRFunctionInternal {
    pub block_graph: Graph<usize, ()>,
    pub stack_bytes_used: usize,
    pub blocks: Vec<IRBasicBlock>,
    pub instructions: Vec<IndexedInstruction>,
}

#[derive(Debug)]
pub struct IRFunction {
    pub func: Rc<RefCell<IRFunctionInternal>>,
}

#[derive(Debug)]
pub struct IRBasicBlock {
    pub is_closed: bool,
    pub index: usize,
    pub inputs: Vec<DataType>,
    pub instructions: Vec<usize>,
}

pub struct IRBlockHandle {
    pub index: usize,
    pub inputs: Vec<DataType>,
    func: Rc<RefCell<IRFunctionInternal>>,
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
        IRFunction {
            func: Rc::new(RefCell::new(IRFunctionInternal {
                block_graph: Graph::new(),
                blocks: Vec::new(),
                instructions: Vec::new(),
                stack_bytes_used: 0,
            })),
        }
    }

    pub fn new_block(&self, inputs: Vec<DataType>) -> IRBlockHandle {
        let index = self.func.borrow().blocks.len();
        {
            let mut func = self.func.borrow_mut();
            func.blocks.push(IRBasicBlock {
                is_closed: false,
                index,
                inputs: inputs.clone(),
                instructions: Vec::new(),
            });
            func.block_graph.add_node(index);
        }
        return IRBlockHandle {
            index,
            inputs,
            func: self.func.clone(),
        };
    }

    pub fn validate(&self) {
        self.func.borrow().validate();
    }
}

pub struct IRFunctionBlockDominanceGraph {
    doms: Dominators<NodeIndex>,
}

impl IRFunctionBlockDominanceGraph {
    /// Get a set of all dominators of a given block
    pub fn get_dominators_of_block(&self, block_index: usize) -> HashSet<usize> {
        self.doms
            .dominators((block_index as u32).into())
            .unwrap_or_else(|| panic!("Unreachable block (no dominators found) block index: {}", block_index))
            .map(|n| n.index() as usize)
            .collect::<std::collections::HashSet<_>>()
    }

    /// Returns true if block `a` dominates block `b`
    /// TODO: Optimize, cache in a HashMap?
    pub fn dominates(&self, a: usize, b: usize) -> bool {
        self.doms
            .dominators((b as u32).into())
            .unwrap_or_else(|| panic!("Unreachable block (no dominators found) block index: {}", b))
            .find(|n| n.index() as usize == a)
            .is_some()
    }
}

impl IRFunctionInternal {
    pub fn calculate_dominance_graph(&self) -> IRFunctionBlockDominanceGraph {
        IRFunctionBlockDominanceGraph {
            doms: dominators::simple_fast(&self.block_graph, 0.into()),
        }
    }

    pub fn validate(&self) {
        // Ensure all blocks are closed
        for block in &self.blocks {
            if !block.is_closed {
                panic!("Unclosed block: block_{}", block.index);
            }
        }

        // Ensure all instructions only reference values from blocks that fully dominate their originating block
        let function_block_dominators = self.calculate_dominance_graph();

        for (block_index, block) in self.blocks.iter().enumerate() {
            let doms = function_block_dominators.get_dominators_of_block(block_index);

            block.instructions.iter().for_each(|i_in_block| {
                let instr = &self.instructions[*i_in_block];
                let blocks_referenced = instr.instruction.references_values_in_blocks(&self);

                for block in blocks_referenced {
                    if !doms.contains(&block) {
                        println!("{}", self);
                        panic!(
                            "Instruction '{}' references a value from block b{} which may not be initialized.",
                            instr, block
                        );
                    }
                }
            });
        }
    }

    pub fn append_obj(&mut self, block_handle: &IRBlockHandle, instruction: Instruction) -> usize {
        let block = &mut self.blocks[block_handle.index];
        if block.is_closed {
            panic!("Cannot append to a closed block");
        }

        let index = self.instructions.len();

        // Close the block if necessary
        // Add edges to the block graph
        match &instruction {
            #[cfg(feature = "ir_comments")]
            Instruction::Comment(_) => {}
            Instruction::Branch { if_true, if_false, .. } => {
                self.block_graph
                    .add_edge((block_handle.index as u32).into(), (if_true.block_index as u32).into(), ());
                self.block_graph
                    .add_edge((block_handle.index as u32).into(), (if_false.block_index as u32).into(), ());
                block.is_closed = true;
            }
            Instruction::Jump { target } => {
                self.block_graph
                    .add_edge((block_handle.index as u32).into(), (target.block_index as u32).into(), ());
                block.is_closed = true;
            }
            Instruction::Return { .. } => {
                block.is_closed = true;
            }
            Instruction::Instruction { .. } => {}
        }

        self.instructions.push(IndexedInstruction {
            block_index: block.index,
            index,
            instruction,
        });

        block.instructions.push(index);

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

        return InstructionOutput {
            outputs: outputs
                .iter()
                .enumerate()
                .map(|(i, output)| InputSlot::InstructionOutput {
                    instruction_index: index,
                    tp: output.tp,
                    output_index: i,
                })
                .collect(),
        };
    }
}
