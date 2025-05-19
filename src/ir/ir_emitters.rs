use super::*;

impl IRFunction {
    pub fn add(
        &mut self,
        block_handle: &IRBlockHandle,
        result_tp: DataType,
        arg1: InputSlot,
        arg2: InputSlot,
    ) -> InstructionOutput {
        self.append(block_handle, InstructionType::Add, vec![arg1, arg2], vec![OutputSlot { tp: result_tp }])
    }

    pub fn load_constant(&mut self, block_handle: &IRBlockHandle, value: Constant) -> InstructionOutput {
        self.append(
            block_handle,
            InstructionType::LoadConstant,
            vec![InputSlot::Constant(value)],
            vec![OutputSlot { tp: value.get_type() }],
        )
    }

    pub fn write_ptr(
        &mut self,
        block_handle: &IRBlockHandle,
        tp: DataType,
        ptr: InputSlot,
        offset: usize,
        value: InputSlot,
    ) -> InstructionOutput {
        self.append(
            block_handle,
            InstructionType::WritePtr,
            vec![
                ptr,
                InputSlot::Constant(Constant::U64(offset as u64)),
                value,
                InputSlot::Constant(Constant::DataType(tp)),
            ],
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
