use crate::ir::{Constant, DataType, IRBlock, InputSlot, Instruction};

fn get_zero(tp: DataType) -> Constant {
    match tp {
        DataType::U8 => Constant::U8(0),
        DataType::U16 => Constant::U16(0),
        DataType::U32 => Constant::U32(0),
        DataType::U64 => Constant::U64(0),
        DataType::S8 => Constant::S8(0),
        DataType::S16 => Constant::S16(0),
        DataType::S32 => Constant::S32(0),
        DataType::S64 => Constant::S64(0),
        DataType::F32 => Constant::F32(0.0),
        DataType::F64 => Constant::F64(0.0),
        DataType::Ptr => Constant::Ptr(0),
        // DataType::None => unimplemented!(),
        DataType::Flags => unimplemented!(),
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
        Constant::F32(_) => unimplemented!("Floats are not supported"),
        Constant::F64(_) => unimplemented!("Floats are not supported"),
        Constant::Ptr(v) => *v as u64,
    }
}

fn get_constant_inputs(inputs: &Vec<InputSlot>, results: &Vec<Vec<Constant>>) -> Vec<Constant> {
    inputs
        .iter()
        .map(|input| match input {
            InputSlot::Constant(constant) => constant.clone(),
            InputSlot::InstructionOutput {
                instruction_index,
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
                .iter()
                .map(|x| constant_to_u64(x))
                .fold(0 as u64, |acc, input| acc.wrapping_add(input));

            let result_const = match instruction.data_tp {
                // DataType::None => todo!(),
                DataType::U8 => Constant::U8(result as u8),
                DataType::S8 => Constant::S8(result as i8),
                DataType::U16 => Constant::U16(result as u16),
                DataType::S16 => Constant::S16(result as i16),
                DataType::U32 => Constant::U32(result as u32),
                DataType::S32 => Constant::S32(result as i32),
                DataType::U64 => Constant::U64(result),
                DataType::S64 => Constant::S64(result as i64),
                DataType::F32 => todo!(),
                DataType::F64 => todo!(),
                DataType::Ptr => todo!(),
                DataType::Flags => todo!(),
            };

            return vec![result_const];
        }
        crate::ir::InstructionType::LoadPtr => todo!("LoadPtr"),
        crate::ir::InstructionType::WritePtr => {
            assert_eq!(inputs.len(), 2);

            let ptr = match inputs[0] {
                Constant::Ptr(p) => p,
                _ => panic!("Expected pointer as first input"),
            };
            let value = constant_to_u64(&inputs[1]);
            let raw_ptr = ptr as *mut u8;
            unsafe {
                match instruction.data_tp {
                    DataType::U8 => *(raw_ptr as *mut u8) = value as u8,
                    DataType::S8 => *(raw_ptr as *mut i8) = value as i8,
                    DataType::U16 => *(raw_ptr as *mut u16) = value as u16,
                    DataType::S16 => *(raw_ptr as *mut i16) = value as i16,
                    DataType::U32 => *(raw_ptr as *mut u32) = value as u32,
                    DataType::S32 => *(raw_ptr as *mut i32) = value as i32,
                    DataType::U64 => *(raw_ptr as *mut u64) = value,
                    DataType::S64 => *(raw_ptr as *mut i64) = value as i64,
                    DataType::F32 => todo!("F32 write"),
                    DataType::F64 => todo!("F64 write"),
                    DataType::Ptr => unimplemented!("Pointer write"),
                    DataType::Flags => unimplemented!("Flags write"),
                }
            }

            return vec![];
        }
    }
}

pub fn interpret_block(block: IRBlock) {
    println!("Interpreting IRBlock");
    block.instructions.iter().for_each(|instruction| {
        println!("{} {:?}", instruction.index, instruction.instruction);
    });

    let mut results: Vec<Vec<Constant>> = vec![];

    block.instructions.iter().for_each(|instruction| {
        let result = evaluate_instr(&block, &mut results, &instruction.instruction);
        println!(
            "Done evaluating instruction: {:?} - Result {:?}",
            instruction, result
        );
        results.push(result);
    })
}
