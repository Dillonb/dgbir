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

    pub fn left_shift(
        &mut self,
        block_handle: &IRBlockHandle,
        result_tp: DataType,
        arg1: InputSlot,
        arg2: InputSlot,
    ) -> InstructionOutput {
        self.append(block_handle, InstructionType::LeftShift, vec![arg1, arg2], vec![OutputSlot { tp: result_tp }])
    }

    pub fn right_shift(
        &mut self,
        block_handle: &IRBlockHandle,
        result_tp: DataType,
        arg1: InputSlot,
        arg2: InputSlot,
    ) -> InstructionOutput {
        self.append(block_handle, InstructionType::RightShift, vec![arg1, arg2], vec![OutputSlot { tp: result_tp }])
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

    pub fn convert(&mut self, block_handle: &IRBlockHandle, tp: DataType, value: InputSlot) -> InstructionOutput {
        self.append(block_handle, InstructionType::Convert, vec![value], vec![OutputSlot { tp }])
    }

    pub fn and(
        &mut self,
        block_handle: &IRBlockHandle,
        result_tp: DataType,
        arg1: InputSlot,
        arg2: InputSlot,
    ) -> InstructionOutput {
        self.append(block_handle, InstructionType::And, vec![arg1, arg2], vec![OutputSlot { tp: result_tp }])
    }
    // InstructionType::Or
    pub fn or(
        &mut self,
        block_handle: &IRBlockHandle,
        result_tp: DataType,
        arg1: InputSlot,
        arg2: InputSlot,
    ) -> InstructionOutput {
        self.append(block_handle, InstructionType::Or, vec![arg1, arg2], vec![OutputSlot { tp: result_tp }])
    }

    pub fn not(&mut self, block_handle: &IRBlockHandle, result_tp: DataType, arg: InputSlot) -> InstructionOutput {
        self.append(block_handle, InstructionType::Not, vec![arg], vec![OutputSlot { tp: result_tp }])
    }

    pub fn xor(
        &mut self,
        block_handle: &IRBlockHandle,
        result_tp: DataType,
        arg1: InputSlot,
        arg2: InputSlot,
    ) -> InstructionOutput {
        self.append(block_handle, InstructionType::Xor, vec![arg1, arg2], vec![OutputSlot { tp: result_tp }])
    }

    pub fn subtract(
        &mut self,
        block_handle: &IRBlockHandle,
        result_tp: DataType,
        minuend: InputSlot,
        subtrahend: InputSlot,
    ) -> InstructionOutput {
        self.append(
            block_handle,
            InstructionType::Subtract,
            vec![minuend, subtrahend],
            vec![OutputSlot { tp: result_tp }],
        )
    }

    pub fn multiply(
        &mut self,
        block_handle: &IRBlockHandle,
        result_tp: DataType,
        arg1: InputSlot,
        arg2: InputSlot,
    ) -> InstructionOutput {
        self.append(block_handle, InstructionType::Multiply, vec![arg1, arg2], vec![OutputSlot { tp: result_tp }])
    }

    /// Divides two values and returns the quotient and remainder.
    pub fn divide(
        &mut self,
        block_handle: &IRBlockHandle,
        result_tp: DataType,
        dividend: InputSlot,
        divisor: InputSlot,
    ) -> InstructionOutput {
        self.append(
            block_handle,
            InstructionType::Divide,
            vec![dividend, divisor],
            vec![OutputSlot { tp: result_tp }, OutputSlot { tp: result_tp }],
        )
    }

    pub fn square_root(
        &mut self,
        block_handle: &IRBlockHandle,
        result_tp: DataType,
        value: InputSlot,
    ) -> InstructionOutput {
        self.append(block_handle, InstructionType::SquareRoot, vec![value], vec![OutputSlot { tp: result_tp }])
    }

    pub fn absolute_value(
        &mut self,
        block_handle: &IRBlockHandle,
        result_tp: DataType,
        value: InputSlot,
    ) -> InstructionOutput {
        self.append(block_handle, InstructionType::AbsoluteValue, vec![value], vec![OutputSlot { tp: result_tp }])
    }

    pub fn negate(&mut self, block_handle: &IRBlockHandle, result_tp: DataType, value: InputSlot) -> InstructionOutput {
        self.append(block_handle, InstructionType::Negate, vec![value], vec![OutputSlot { tp: result_tp }])
    }

    pub fn call_function(
        &mut self,
        block_handle: &IRBlockHandle,
        address: InputSlot,
        return_tp: DataType,
        args: Vec<InputSlot>,
    ) -> InstructionOutput {
        self.append(
            block_handle,
            InstructionType::CallFunction,
            std::iter::once(address).chain(args).collect(),
            vec![OutputSlot { tp: return_tp }],
        )
    }
}
