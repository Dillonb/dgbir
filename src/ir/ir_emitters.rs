use super::*;
use crate::abi::IsSimd;
use log::warn;

impl IRBlockHandle {
    fn append(&self, tp: InstructionType, inputs: Vec<InputSlot>, outputs: Vec<OutputSlot>) -> InstructionOutput {
        self.func.borrow_mut().append(self, tp, inputs, outputs)
    }

    #[cfg(feature = "ir_comments")]
    pub fn comment(&self, text: impl AsRef<str>) {
        let text = text.as_ref();
        self.func
            .borrow_mut()
            .append_obj(self, Instruction::Comment(text.to_string()));
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

    /// Shifts the whole vector left by `bytes`, ignoring its lane structure. Shifting by the width
    /// of the vector or more gives zero.
    pub fn vector_left_shift_bytes(
        &mut self,
        result_tp: DataType,
        arg1: InputSlot,
        bytes: InputSlot,
    ) -> InstructionOutput {
        self.append(InstructionType::VectorLeftShiftBytes, vec![arg1, bytes], vec![OutputSlot { tp: result_tp }])
    }

    /// Shifts the whole vector right by `bytes`, ignoring its lane structure. Shifting by the width
    /// of the vector or more gives zero.
    pub fn vector_right_shift_bytes(
        &mut self,
        result_tp: DataType,
        arg1: InputSlot,
        bytes: InputSlot,
    ) -> InstructionOutput {
        self.append(InstructionType::VectorRightShiftBytes, vec![arg1, bytes], vec![OutputSlot { tp: result_tp }])
    }

    /// Rearranges the lanes of `value`. Nibble `i` of `pattern` is the index of the source lane
    /// that ends up in lane `i` of the result, counting from the low end of the vector.
    pub fn vector_swizzle(&mut self, result_tp: DataType, value: InputSlot, pattern: u64) -> InstructionOutput {
        self.append(
            InstructionType::VectorSwizzle,
            vec![value, InputSlot::Constant(Constant::U64(pattern))],
            vec![OutputSlot { tp: result_tp }],
        )
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

    pub fn compare(&mut self, dtp: DataType, x: InputSlot, ctp: CompareType, y: InputSlot) -> InstructionOutput {
        self.append(
            InstructionType::Compare,
            vec![
                InputSlot::Constant(Constant::DataType(dtp)),
                x,
                InputSlot::Constant(Constant::CompareType(ctp)),
                y,
            ],
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
        let slot = match mult_type {
            MultiplyType::Split => OutputSlot {
                tp: result_tp.half_type(),
            },
            MultiplyType::Combined => OutputSlot { tp: result_tp },
        };
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

    pub fn call_function(&mut self, func: ExternalFunction, args: Vec<InputSlot>) -> InstructionOutput {
        check_call_signature(&func, &args);
        self.append(
            InstructionType::CallFunction,
            std::iter::once(InputSlot::Constant(Constant::Ptr(func.address)))
                .chain(args)
                .collect(),
            func.returns.map(|tp| OutputSlot { tp }).into_iter().collect(),
        )
    }
}

fn check_call_signature(func: &ExternalFunction, args: &[InputSlot]) {
    let address = func.address;
    let (expected, got) = (func.params.len(), args.len());
    assert!(expected == got, "Function at {address:#x} takes {expected} arguments, called with {got}");

    for (i, (param, arg)) in func.params.iter().zip(args).enumerate() {
        let arg = arg.tp();
        assert!(
            param.is_simd() == arg.is_simd(),
            "Argument {i} of the function at {address:#x} is {param}, but a {arg} was passed"
        );
        if param.size() != arg.size() {
            warn!("Argument {i} of the function at {address:#x} is {param}, but a {arg} was passed");
        }
    }
}
