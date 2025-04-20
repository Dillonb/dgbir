use std::{cell::RefCell, collections::BTreeSet};

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
    Ptr,
    Flags,
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
    Label(Label),
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
    ConditionalBranch,
    Compare,
    LoadPtr,
    Phi,
    WritePtr,
}

#[derive(Debug, Clone, Copy)]
pub enum InputSlot {
    /// References the output of another instruction.
    InstructionOutput {
        instruction_index: usize,
        tp: DataType,
        output_index: usize,
    },
    Constant(Constant),
}

// pub struct InputSlotSchema {
//     name: String,
// }

// pub struct OutputSlotSchema {
//     name: String
// }

// pub struct InstructionSchema {
//     inputs: Vec<InputSlotSchema>,
//     outputs: Vec<OutputSlotSchema>,
// }

#[derive(Debug, Clone, Copy)]
pub struct OutputSlot {
    pub tp: DataType,
}

#[derive(Debug)]
pub struct Instruction {
    pub tp: InstructionType,
    pub inputs: Vec<InputSlot>,
    pub outputs: Vec<OutputSlot>,
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

#[derive(Debug, Clone, Copy)]
pub struct Label {
    pub index: usize,
}

impl PartialEq for Label {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}

impl Eq for Label {}

impl PartialOrd for Label {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Label {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.index.cmp(&other.index)
    }
}

#[derive(Debug)]
pub struct IndexedInstruction {
    /// The index of the instruction in the IR vec
    pub index: usize,
    pub instruction: Instruction,
}

#[derive(Debug)]
pub struct IRContext {
    /// Memory addresses available to the IR
    pub inputs: Vec<usize>,
}

impl IRContext {
    pub fn new() -> RefCell<Self> {
        Into::into(IRContext { inputs: Vec::new() })
    }
}

#[derive(Debug)]
pub struct IRBlock {
    pub context: RefCell<IRContext>,
    pub instructions: Vec<IndexedInstruction>,
    pub labels: BTreeSet<Label>,
}

impl IRBlock {
    pub fn new(context: RefCell<IRContext>) -> Self {
        IRBlock {
            context,
            instructions: Vec::new(),
            labels: BTreeSet::new(),
        }
    }

    pub fn append_obj(&mut self, instruction: Instruction) -> usize {
        let index = self.instructions.len();
        self.instructions
            .push(IndexedInstruction { index, instruction });
        return index;
    }

    pub fn append(
        &mut self,
        tp: InstructionType,
        inputs: Vec<InputSlot>,
        outputs: Vec<OutputSlot>,
    ) -> InstructionOutput {
        let index = self.append_obj(Instruction {
            tp,
            inputs,
            outputs: outputs.clone(),
        });

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

    pub fn const_u32(value: u32) -> InputSlot {
        InputSlot::Constant(Constant::U32(value))
    }

    pub fn const_f32(value: f32) -> InputSlot {
        InputSlot::Constant(Constant::F32(value))
    }

    pub fn const_ptr(value: usize) -> InputSlot {
        InputSlot::Constant(Constant::Ptr(value))
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

    pub fn label(&mut self) -> Label {
        let index = self.instructions.len();
        let label = Label { index };
        self.labels.insert(label.clone());
        return label;
    }

    pub fn phi(&mut self, tp: DataType, inputs: Vec<InputSlot>) -> InstructionOutput {
        self.append(InstructionType::Phi, inputs, vec![OutputSlot { tp }])
    }

    pub fn compare(&mut self, x: InputSlot, tp: CompareType, y: InputSlot) -> InstructionOutput {
        self.append(
            InstructionType::Compare,
            vec![x, InputSlot::Constant(Constant::CompareType(tp)), y],
            vec![OutputSlot {
                tp: DataType::Flags,
            }],
        )
    }

    pub fn conditional_branch(
        &mut self,
        cond: InputSlot,
        label: Label,
    ) -> () {
        // Should branches have an output?
        self.append(
            InstructionType::ConditionalBranch,
            vec![
                cond,
                InputSlot::Constant(Constant::Label(label)),
            ],
            vec![],
        );
    }

    pub fn add_phi_input(&mut self, phi_node: &InstructionOutput, val: InputSlot) {
        let InputSlot::InstructionOutput {
            instruction_index, ..
        } = phi_node.val()
        else {
            panic!("This is not a phi node: not an InstructionOutput");
        };

        let instruction = &mut self.instructions[instruction_index];

        let IndexedInstruction {
            instruction:
                Instruction {
                    tp: InstructionType::Phi,
                    inputs,
                    ..
                },
            ..
        } = instruction
        else {
            panic!("This is not a phi node: not a Phi instruction");
        };

        inputs.push(val);
    }
}
