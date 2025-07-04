use super::*;

impl IRBlockHandle {
    fn append(&self, tp: InstructionType, inputs: Vec<InputSlot>, outputs: Vec<OutputSlot>) -> InstructionOutput {
        self.func.borrow_mut().append(self, tp, inputs, outputs)
    }

    pub fn add(&self, result_tp: DataType, arg1: InputSlot, arg2: InputSlot) -> InstructionOutput {
        self.append(InstructionType::Add, vec![arg1, arg2], vec![OutputSlot { tp: result_tp }])
    }

    pub fn left_shift(&mut self, result_tp: DataType, arg1: InputSlot, arg2: InputSlot) -> InstructionOutput {
        self.append(InstructionType::LeftShift, vec![arg1, arg2], vec![OutputSlot { tp: result_tp }])
    }

    pub fn right_shift(&mut self, result_tp: DataType, arg1: InputSlot, arg2: InputSlot) -> InstructionOutput {
        self.append(InstructionType::RightShift, vec![arg1, arg2], vec![OutputSlot { tp: result_tp }])
    }

    pub fn write_ptr(&mut self, tp: DataType, ptr: InputSlot, offset: usize, value: InputSlot) -> InstructionOutput {
        self.append(
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

    pub fn load_ptr(&mut self, tp: DataType, ptr: InputSlot, offset: usize) -> InstructionOutput {
        self.append(
            InstructionType::LoadPtr,
            vec![ptr, InputSlot::Constant(Constant::U64(offset as u64))],
            vec![OutputSlot { tp }],
        )
    }

    pub fn compare(&mut self, x: InputSlot, tp: CompareType, y: InputSlot) -> InstructionOutput {
        self.append(
            InstructionType::Compare,
            vec![x, InputSlot::Constant(Constant::CompareType(tp)), y],
            vec![OutputSlot { tp: DataType::Bool }],
        )
    }

    pub fn branch(&mut self, cond: InputSlot, if_true: BlockReference, if_false: BlockReference) {
        self.func.borrow_mut().append_obj(
            self,
            Instruction::Branch {
                cond,
                if_true,
                if_false,
            },
        );
    }

    pub fn jump(&mut self, target: BlockReference) {
        self.func.borrow_mut().append_obj(self, Instruction::Jump { target });
    }

    pub fn ret(&mut self, input: Option<InputSlot>) {
        self.func
            .borrow_mut()
            .append_obj(self, Instruction::Return { value: input });
    }

    /// Converts a value to a specified data type. Uses the output of the input slot as the source
    /// data type.
    pub fn convert(&mut self, tp: DataType, value: InputSlot) -> InstructionOutput {
        self.append(InstructionType::Convert, vec![value], vec![OutputSlot { tp }])
    }

    /// Converts a value from one data type to another. Takes a specific source data type. Useful
    /// for sign extending values from instructions that output unsigned types.
    pub fn convert_from(&mut self, from_tp: DataType, to_tp: DataType, value: InputSlot) -> InstructionOutput {
        self.append(
            InstructionType::Convert,
            vec![value, Constant::DataType(from_tp).into_inputslot()],
            vec![OutputSlot { tp: to_tp }],
        )
    }

    pub fn and(&mut self, result_tp: DataType, arg1: InputSlot, arg2: InputSlot) -> InstructionOutput {
        self.append(InstructionType::And, vec![arg1, arg2], vec![OutputSlot { tp: result_tp }])
    }

    pub fn or(&mut self, result_tp: DataType, arg1: InputSlot, arg2: InputSlot) -> InstructionOutput {
        self.append(InstructionType::Or, vec![arg1, arg2], vec![OutputSlot { tp: result_tp }])
    }

    pub fn not(&mut self, result_tp: DataType, arg: InputSlot) -> InstructionOutput {
        self.append(InstructionType::Not, vec![arg], vec![OutputSlot { tp: result_tp }])
    }

    pub fn xor(&mut self, result_tp: DataType, arg1: InputSlot, arg2: InputSlot) -> InstructionOutput {
        self.append(InstructionType::Xor, vec![arg1, arg2], vec![OutputSlot { tp: result_tp }])
    }

    pub fn subtract(&mut self, result_tp: DataType, minuend: InputSlot, subtrahend: InputSlot) -> InstructionOutput {
        self.append(InstructionType::Subtract, vec![minuend, subtrahend], vec![OutputSlot { tp: result_tp }])
    }

    pub fn multiply(
        &mut self,
        result_tp: DataType,
        arg_type: DataType,
        mult_type: MultiplyType,
        arg1: InputSlot,
        arg2: InputSlot,
    ) -> InstructionOutput {
        let slot = OutputSlot { tp: result_tp };
        let outputs = match mult_type {
            MultiplyType::Split => vec![slot, slot],
            MultiplyType::Combined => vec![slot],
        };
        self.append(InstructionType::Multiply, vec![arg1, arg2, Constant::DataType(arg_type).into_inputslot()], outputs)
    }

    /// Divides two values and returns the quotient and remainder.
    pub fn divide(&mut self, result_tp: DataType, dividend: InputSlot, divisor: InputSlot) -> InstructionOutput {
        self.append(
            InstructionType::Divide,
            vec![dividend, divisor],
            vec![OutputSlot { tp: result_tp }, OutputSlot { tp: result_tp }],
        )
    }

    pub fn square_root(&mut self, result_tp: DataType, value: InputSlot) -> InstructionOutput {
        self.append(InstructionType::SquareRoot, vec![value], vec![OutputSlot { tp: result_tp }])
    }

    pub fn absolute_value(&mut self, result_tp: DataType, value: InputSlot) -> InstructionOutput {
        self.append(InstructionType::AbsoluteValue, vec![value], vec![OutputSlot { tp: result_tp }])
    }

    pub fn negate(&mut self, result_tp: DataType, value: InputSlot) -> InstructionOutput {
        self.append(InstructionType::Negate, vec![value], vec![OutputSlot { tp: result_tp }])
    }

    pub fn call_function(
        &mut self,
        address: InputSlot,
        return_tp: Option<DataType>,
        args: Vec<InputSlot>,
    ) -> InstructionOutput {
        self.append(
            InstructionType::CallFunction,
            std::iter::once(address).chain(args).collect(),
            return_tp.map(|tp| OutputSlot { tp }).into_iter().collect(),
        )
    }
}
