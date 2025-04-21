use crate::ir::{
    CompareType, Constant, DataType, IRBasicBlock, InputSlot, Instruction, InstructionType,
};

/// Used to simplify code around integer math when width is not as important.
enum MiniConstant {
    U64(u64),
    S64(i64),
    F32(f32),
    F64(f64),
}

fn constant_to_mini_constant(val: &Constant) -> MiniConstant {
    match val {
        Constant::U8(v) => MiniConstant::U64(*v as u64),
        Constant::S8(v) => MiniConstant::S64(*v as i64),
        Constant::U16(v) => MiniConstant::U64(*v as u64),
        Constant::S16(v) => MiniConstant::S64(*v as i64),
        Constant::U32(v) => MiniConstant::U64(*v as u64),
        Constant::S32(v) => MiniConstant::S64(*v as i64),
        Constant::U64(v) => MiniConstant::U64(*v),
        Constant::S64(v) => MiniConstant::S64(*v),
        Constant::F32(v) => MiniConstant::F32(*v),
        Constant::F64(v) => MiniConstant::F64(*v),
        _ => panic!("Unsupported constant type for addition: {:?}", val),
    }
}

fn mini_constant_to_constant(val: &MiniConstant, tp: DataType) -> Constant {
    match (val, tp) {
        // U64 can be converted to any unsigned integer type
        (MiniConstant::U64(v), DataType::U8) => Constant::U8(*v as u8),
        (MiniConstant::U64(v), DataType::U16) => Constant::U16(*v as u16),
        (MiniConstant::U64(v), DataType::U32) => Constant::U32(*v as u32),
        (MiniConstant::U64(v), DataType::U64) => Constant::U64(*v),
        (MiniConstant::U64(_), b) => unimplemented!("Cannot convert U64 to {:?}", b),

        // S64 can be converted to any signed integer type
        (MiniConstant::S64(v), DataType::S8) => Constant::S8(*v as i8),
        (MiniConstant::S64(v), DataType::S16) => Constant::S16(*v as i16),
        (MiniConstant::S64(v), DataType::S32) => Constant::S32(*v as i32),
        (MiniConstant::S64(v), DataType::S64) => Constant::S64(*v),
        (MiniConstant::S64(_), b) => unimplemented!("Cannot convert S64 to {:?}", b),

        // Floats just have to be the right type
        (MiniConstant::F32(v), DataType::F32) => Constant::F32(*v),
        (MiniConstant::F32(_), b) => unimplemented!("Cannot convert F32 to {:?}", b),
        (MiniConstant::F64(v), DataType::F64) => Constant::F64(*v),
        (MiniConstant::F64(_), b) => unimplemented!("Cannot convert F64 to {:?}", b),
    }
}

fn constant_to_u64(c: &Constant) -> u64 {
    match c {
        Constant::U8(v) => *v as u64,
        Constant::U16(v) => *v as u64,
        Constant::U32(v) => *v as u64,
        Constant::U64(v) => *v,
        Constant::S8(v) => *v as u64,
        Constant::S16(v) => *v as u64,
        Constant::S32(v) => *v as u64,
        Constant::S64(v) => *v as u64,
        Constant::F32(v) => v.to_bits() as u64,
        Constant::F64(v) => v.to_bits(),
        Constant::Ptr(v) => *v as u64,
        Constant::Bool(v) => *v as u64,
        Constant::DataType(_) => unimplemented!("DataType is not supported"),
        Constant::CompareType(_) => unimplemented!("CompareType is not supported"),
    }
}

fn get_constant_inputs(inputs: &Vec<InputSlot>, _results: &Vec<Vec<Constant>>) -> Vec<Constant> {
    inputs
        .iter()
        .map(|input| match input {
            InputSlot::Constant(constant) => constant.clone(),
            InputSlot::BlockInput { .. } =>  todo!("BlockInput not implemented"),
            InputSlot::InstructionOutput { .. } => todo!("InstructionOutput not implemented"),
        })
        .collect()
}

fn evaluate_instr(
    _block: &IRBasicBlock,
    _pc: &mut usize,
    _instruction_last_executed: &Vec<i32>,
    results: &mut Vec<Vec<Constant>>,
    instruction: &Instruction,
) -> Vec<Constant> {
    match instruction {
        Instruction::Instruction { tp: instr_tp, inputs, outputs } => {
                let const_inputs = get_constant_inputs(&inputs, results);
                match instr_tp {
                    InstructionType::Add => {
                        let result = const_inputs
                            .into_iter()
                            .map(|val| constant_to_mini_constant(&val))
                            .reduce(|acc, val| match (acc, val) {
                                // Pure integer addition
                                (MiniConstant::U64(a), MiniConstant::U64(b)) => {
                                    MiniConstant::U64(a.wrapping_add(b))
                                }
                                (MiniConstant::U64(a), MiniConstant::S64(b)) => {
                                    MiniConstant::U64(a.wrapping_add_signed(b))
                                }
                                (MiniConstant::S64(a), MiniConstant::U64(b)) => {
                                    MiniConstant::S64(a.wrapping_add_unsigned(b))
                                }
                                (MiniConstant::S64(a), MiniConstant::S64(b)) => {
                                    MiniConstant::S64(a.wrapping_add(b))
                                }

                                // Mixed integers and floats: int result
                                (MiniConstant::U64(a), MiniConstant::F32(b)) => {
                                    MiniConstant::U64(a.wrapping_add_signed(b as i64))
                                }
                                (MiniConstant::U64(a), MiniConstant::F64(b)) => {
                                    MiniConstant::U64(a.wrapping_add_signed(b as i64))
                                }
                                (MiniConstant::S64(a), MiniConstant::F32(b)) => {
                                    MiniConstant::S64(a.wrapping_add(b as i64))
                                }
                                (MiniConstant::S64(a), MiniConstant::F64(b)) => {
                                    MiniConstant::S64(a.wrapping_add(b as i64))
                                }

                                // Mixed integers and floats: float result
                                (MiniConstant::F32(a), MiniConstant::U64(b)) => MiniConstant::F32(a + b as f32),
                                (MiniConstant::F32(a), MiniConstant::S64(b)) => MiniConstant::F32(a + b as f32),
                                (MiniConstant::F64(a), MiniConstant::U64(b)) => MiniConstant::F64(a + b as f64),
                                (MiniConstant::F64(a), MiniConstant::S64(b)) => MiniConstant::F64(a + b as f64),

                                // Pure float addition
                                (MiniConstant::F32(a), MiniConstant::F32(b)) => MiniConstant::F32(a + b),
                                (MiniConstant::F32(a), MiniConstant::F64(b)) => MiniConstant::F32(a + b as f32),
                                (MiniConstant::F64(a), MiniConstant::F32(b)) => MiniConstant::F64(a + b as f64),
                                (MiniConstant::F64(a), MiniConstant::F64(b)) => MiniConstant::F64(a + b),
                            })
                            .unwrap();

                        return vec![mini_constant_to_constant(
                            &result,
                            outputs[0].tp,
                        )];
                    }
                    InstructionType::LoadPtr => todo!("LoadPtr"),
                    InstructionType::WritePtr => {
                        assert_eq!(const_inputs.len(), 3);

                        let ptr = match const_inputs[0] {
                            Constant::Ptr(p) => p,
                            _ => panic!("Expected pointer as first input"),
                        };
                        let value = constant_to_u64(&const_inputs[1]);
                        let raw_ptr = ptr as *mut u8;
                        let tp = match const_inputs[2] {
                            Constant::DataType(tp) => tp,
                            _ => panic!("Expected DataType as third input"),
                        };
                        unsafe {
                            match tp {
                                DataType::U8 => *raw_ptr.cast() = value as u8,
                                DataType::S8 => *raw_ptr.cast() = value as i8,
                                DataType::U16 => *raw_ptr.cast() = value as u16,
                                DataType::S16 => *raw_ptr.cast() = value as i16,
                                DataType::U32 => *raw_ptr.cast() = value as u32,
                                DataType::S32 => *raw_ptr.cast() = value as i32,
                                DataType::U64 => *raw_ptr.cast() = value,
                                DataType::S64 => *raw_ptr.cast() = value as i64,

                                DataType::Bool => *raw_ptr.cast() = value != 0,

                                // For floats, the constant_to_u64() above will extract the bits, we just need
                                // to write the correct number of bytes here.
                                DataType::F32 => *raw_ptr.cast() = value as u32,
                                DataType::F64 => *raw_ptr.cast() = value as u64,

                                DataType::Ptr => *raw_ptr.cast() = value as usize,
                            }
                        }

                        return vec![];
                    }
                    InstructionType::Compare => {
                        assert_eq!(const_inputs.len(), 3);
                        let a = constant_to_u64(&const_inputs[0]);
                        let c = match const_inputs[1] {
                            Constant::CompareType(c) => c,
                            _ => panic!("Expected CompareType as second input"),
                        };
                        let b = constant_to_u64(&const_inputs[2]);

                        let result = match c {
                            CompareType::Equal => a == b,
                            CompareType::NotEqual => a != b,

                            CompareType::LessThanSigned => todo!("LessThanSigned"),
                            CompareType::GreaterThanSigned => todo!("GreaterThanSigned"),
                            CompareType::LessThanOrEqualSigned => todo!("LessThanOrEqualSigned"),
                            CompareType::GreaterThanOrEqualSigned => todo!("GreaterThanOrEqualSigned"),

                            CompareType::LessThanUnsigned => a < b,
                            CompareType::GreaterThanUnsigned => a > b,
                            CompareType::LessThanOrEqualUnsigned => a <= b,
                            CompareType::GreaterThanOrEqualUnsigned => a >= b,
                        };

                        return vec![Constant::Bool(result)];
                    }
                }
            }
        Instruction::Branch { .. } => todo!(),
        Instruction::Jump { .. } => todo!(),
        Instruction::Return { .. } => todo!(),
    }
}

pub fn interpret_block(block: &IRBasicBlock) {
    let mut results: Vec<Vec<Constant>> = vec![vec![]; block.instructions.len()];
    let mut pc = 0;

    // Data for phi nodes
    let mut instructions_executed = 0;
    let mut instruction_last_executed: Vec<i32> = vec![0; block.instructions.len()];

    while pc < block.instructions.len() {
        let instruction = &block.instructions[pc];
        let instruction_index = pc;
        pc += 1;
        results[instruction_index] = evaluate_instr(
            block,
            &mut pc,
            &instruction_last_executed,
            &mut results,
            &instruction.instruction,
        );

        // Data for phi nodes
        instructions_executed += 1;
        instruction_last_executed[instruction_index] = instructions_executed;
    }
}
