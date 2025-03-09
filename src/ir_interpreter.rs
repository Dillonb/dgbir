use crate::ir::{IRBlock, Instruction};

fn evaluate_instr(_block: &IRBlock, instruction: &Instruction) {
        match instruction.tp {
            crate::ir::InstructionType::Add => todo!(),
            crate::ir::InstructionType::LoadPtr => todo!(),
            crate::ir::InstructionType::WritePtr => todo!(),
        }

}

pub fn interpret_block(block: IRBlock) {
    println!("Interpreting IRBlock");
    println!("{:?}", block);

    block.instructions.iter().for_each(|instruction| {
        evaluate_instr(&block, &instruction.instruction);
    })
}
