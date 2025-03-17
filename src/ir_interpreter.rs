use crate::ir::{Constant, DataType, IRBlock, InputSlot, Instruction};

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
        Constant::DataType(_) => unimplemented!("DataType is not supported"),
    }
}

fn get_constant_inputs(inputs: &Vec<InputSlot>, results: &Vec<Vec<Constant>>) -> Vec<Constant> {
    inputs
        .iter()
        .map(|input| match input {
            InputSlot::Constant(constant) => constant.clone(),
            InputSlot::DataType(data_type) => Constant::DataType(data_type.clone()),
            InputSlot::InstructionOutput {
                instruction_index,
                tp: _,
                output_index,
            } => results[*instruction_index][*output_index].clone(),
        })
        .collect()
}

fn evaluate_instr(
    _block: &IRBlock,
    results: &mut Vec<Vec<Constant>>,
    instruction: &Instruction,
) -> Vec<Constant> {
    let inputs = get_constant_inputs(&instruction.inputs, results);
    match instruction.tp {
        crate::ir::InstructionType::Add => {
            let result = inputs
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
                instruction.outputs[0].tp,
            )];
        }
        crate::ir::InstructionType::LoadPtr => todo!("LoadPtr"),
        crate::ir::InstructionType::WritePtr => {
            assert_eq!(inputs.len(), 3);

            let ptr = match inputs[0] {
                Constant::Ptr(p) => p,
                _ => panic!("Expected pointer as first input"),
            };
            let value = constant_to_u64(&inputs[1]);
            let raw_ptr = ptr as *mut u8;
            let tp = match inputs[2] {
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

                    // For floats, the constant_to_u64() above will extract the bits, we just need
                    // to write the correct number of bytes here.
                    DataType::F32 => *raw_ptr.cast() = value as u32,
                    DataType::F64 => *raw_ptr.cast() = value as u64,

                    DataType::Ptr => *raw_ptr.cast() = value as usize,
                    DataType::Flags => unimplemented!("Flags write"),
                }
            }

            return vec![];
        }
    }
}

pub fn interpret_block(block: &IRBlock) {
    let mut results: Vec<Vec<Constant>> = vec![];

    block.instructions.iter().for_each(|instruction| {
        let result = evaluate_instr(&block, &mut results, &instruction.instruction);
        results.push(result);
    })
}
