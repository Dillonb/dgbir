use std::collections::HashMap;

use ordered_float::OrderedFloat;

use crate::ir::{
    BlockReference, CompareType, Constant, DataType, IRFunction, InputSlot, Instruction, InstructionType, OutputSlot,
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
        Constant::F32(v) => MiniConstant::F32(**v),
        Constant::F64(v) => MiniConstant::F64(**v),
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
        (MiniConstant::F32(v), DataType::F32) => Constant::F32(OrderedFloat(*v)),
        (MiniConstant::F32(_), b) => unimplemented!("Cannot convert F32 to {:?}", b),
        (MiniConstant::F64(v), DataType::F64) => Constant::F64(OrderedFloat(*v)),
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
        Constant::RoundingMode(_) => unimplemented!("RoundingMode is not supported"),
    }
}

fn resolve_inputslot(
    input: &InputSlot,
    block_inputs: &HashMap<usize, Vec<Constant>>,
    results: &HashMap<usize, Vec<Constant>>,
) -> Constant {
    match input {
        InputSlot::InstructionOutput {
            instruction_index,
            output_index,
            ..
        } => {
            let res = results[instruction_index][*output_index].clone();
            res
        }
        InputSlot::BlockInput {
            block_index,
            input_index,
            ..
        } => {
            let res = block_inputs[block_index][*input_index].clone();
            res
        }
        InputSlot::Constant(constant) => constant.clone(),
    }
}

fn jump_to(
    target: &BlockReference,
    block_index: &mut usize,
    pc: &mut usize,
    block_inputs: &mut HashMap<usize, Vec<Constant>>,
    results: &mut HashMap<usize, Vec<Constant>>,
) {
    *block_index = target.block_index;
    *pc = 0;
    let mut inputs = vec![];
    for arg in &target.arguments {
        inputs.push(resolve_inputslot(arg, &block_inputs, &results));
    }
    block_inputs.insert(*block_index, inputs);
}

fn evaluate_add(inputs: &Vec<Constant>, outputs: &Vec<OutputSlot>) -> Constant {
    let result = inputs
        .into_iter()
        .map(|val| constant_to_mini_constant(&val))
        .reduce(|acc, val| match (acc, val) {
            // Pure integer addition
            (MiniConstant::U64(a), MiniConstant::U64(b)) => MiniConstant::U64(a.wrapping_add(b)),
            (MiniConstant::U64(a), MiniConstant::S64(b)) => MiniConstant::U64(a.wrapping_add_signed(b)),
            (MiniConstant::S64(a), MiniConstant::U64(b)) => MiniConstant::S64(a.wrapping_add_unsigned(b)),
            (MiniConstant::S64(a), MiniConstant::S64(b)) => MiniConstant::S64(a.wrapping_add(b)),

            // Mixed integers and floats: int result
            (MiniConstant::U64(a), MiniConstant::F32(b)) => MiniConstant::U64(a.wrapping_add_signed(b as i64)),
            (MiniConstant::U64(a), MiniConstant::F64(b)) => MiniConstant::U64(a.wrapping_add_signed(b as i64)),
            (MiniConstant::S64(a), MiniConstant::F32(b)) => MiniConstant::S64(a.wrapping_add(b as i64)),
            (MiniConstant::S64(a), MiniConstant::F64(b)) => MiniConstant::S64(a.wrapping_add(b as i64)),

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

    return mini_constant_to_constant(&result, outputs[0].tp);
}

fn evaluate_load_ptr(_inputs: &Vec<Constant>, _outputs: &Vec<OutputSlot>) -> Constant {
    todo!("LoadPtr");
}

fn evaluate_write_ptr(inputs: &Vec<Constant>) {
    assert_eq!(inputs.len(), 4);

    let ptr = match inputs[0] {
        Constant::Ptr(p) => p,
        _ => panic!("Expected pointer as first input"),
    };
    let offset = match inputs[1] {
        Constant::U64(o) => o as usize,
        _ => panic!("Expected offset as second input"),
    };
    let value = constant_to_u64(&inputs[2]);
    let raw_ptr = (ptr + offset) as *mut u8;
    let tp = match inputs[3] {
        Constant::DataType(tp) => tp,
        _ => panic!("Expected DataType as fourth input"),
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

            DataType::U128 => *raw_ptr.cast() = value as u128,

            DataType::Bool => *raw_ptr.cast() = value != 0,

            // For floats, the constant_to_u64() above will extract the bits, we just need
            // to write the correct number of bytes here.
            DataType::F32 => *raw_ptr.cast() = value as u32,
            DataType::F64 => *raw_ptr.cast() = value as u64,

            DataType::Ptr => *raw_ptr.cast() = value as usize,
        }
    }
}

fn evaluate_compare(inputs: &Vec<Constant>, _outputs: &Vec<OutputSlot>) -> Constant {
    assert_eq!(inputs.len(), 4);
    let d = match inputs[0] {
        Constant::DataType(d) => d,
        _ => panic!("Expected DataType as first input"),
    };

    let a = constant_to_u64(&inputs[1]);
    let c = match inputs[2] {
        Constant::CompareType(c) => c,
        _ => panic!("Expected CompareType as second input"),
    };
    let b = constant_to_u64(&inputs[3]);

    let signed = d.is_signed();

    let result = match (signed, c) {
        (_, CompareType::Equal) => a == b,
        (_, CompareType::NotEqual) => a != b,

        (true, CompareType::LessThan) => todo!("LessThanSigned"),
        (true, CompareType::GreaterThan) => todo!("GreaterThanSigned"),
        (true, CompareType::LessThanOrEqual) => todo!("LessThanOrEqualSigned"),
        (true, CompareType::GreaterThanOrEqual) => todo!("GreaterThanOrEqualSigned"),

        (false, CompareType::LessThan) => a < b,
        (false, CompareType::GreaterThan) => a > b,
        (false, CompareType::LessThanOrEqual) => a <= b,
        (false, CompareType::GreaterThanOrEqual) => a >= b,
    };

    return Constant::Bool(result);
}

fn evaluate_instr(tp: &InstructionType, inputs: &Vec<Constant>, outputs: &Vec<OutputSlot>) -> Vec<Constant> {
    match tp {
        InstructionType::Add => vec![evaluate_add(inputs, outputs)],
        InstructionType::LoadPtr => vec![evaluate_load_ptr(inputs, outputs)],
        InstructionType::WritePtr => {
            evaluate_write_ptr(inputs);
            vec![]
        }
        InstructionType::Compare => vec![evaluate_compare(inputs, outputs)],
        InstructionType::SpillToStack => todo!("SpillToStack in IR interpreter"),
        InstructionType::LoadFromStack => todo!("LoadFromStack in IR interpreter"),
        InstructionType::LeftShift => todo!(),
        InstructionType::RightShift => todo!(),
        InstructionType::Convert => todo!(),
        InstructionType::And => todo!(),
        InstructionType::Or => todo!(),
        InstructionType::Not => todo!(),
        InstructionType::Xor => todo!(),
        InstructionType::Subtract => todo!(),
        InstructionType::Multiply => todo!(),
        InstructionType::Divide => todo!(),
        InstructionType::SquareRoot => todo!(),
        InstructionType::AbsoluteValue => todo!(),
        InstructionType::Negate => todo!(),
        InstructionType::CallFunction => todo!(),
    }
}

pub fn interpret_func(func: &IRFunction, args: Vec<Constant>) -> Option<Constant> {
    func.validate();
    let func = func.func.borrow();
    let mut block_index: usize = 0;
    let mut pc: usize = 0;
    let mut returned = false;

    // block -> input
    let mut block_inputs: HashMap<usize, Vec<Constant>> = HashMap::new();
    // instruction index -> output index
    let mut results: HashMap<usize, Vec<Constant>> = HashMap::new();

    let mut return_value: Option<Constant> = None;

    // Fill in the inputs for the first block
    for arg in args {
        block_inputs.entry(0).or_insert_with(Vec::new).push(arg);
    }

    while !returned {
        let block = &func.blocks[block_index];
        let instruction = &func.instructions[block.instructions[pc]];
        match &instruction.instruction {
            Instruction::Instruction { tp, inputs, outputs } => {
                let const_inputs = inputs
                    .iter()
                    .map(|input| resolve_inputslot(input, &block_inputs, &results))
                    .collect::<Vec<Constant>>();

                results
                    .entry(instruction.index)
                    .insert_entry(evaluate_instr(tp, &const_inputs, outputs));

                pc += 1;
            }
            Instruction::Branch {
                cond,
                if_true,
                if_false,
            } => {
                if let Constant::Bool(cond) = resolve_inputslot(cond, &block_inputs, &results) {
                    if cond {
                        jump_to(if_true, &mut block_index, &mut pc, &mut block_inputs, &mut results);
                    } else {
                        jump_to(if_false, &mut block_index, &mut pc, &mut block_inputs, &mut results);
                    }
                } else {
                    panic!("Expected boolean condition");
                }
            }
            Instruction::Jump { target } => jump_to(target, &mut block_index, &mut pc, &mut block_inputs, &mut results),
            Instruction::Return { value } => {
                return_value = value.map(|v| resolve_inputslot(&v, &block_inputs, &results));
                returned = true;
            }
        }
    }

    return return_value;
}
