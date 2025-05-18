use std::{collections::HashMap, mem};

use dynasmrt::AssemblyOffset;
use itertools::Itertools;

#[allow(unused_imports)]
use crate::{compiler_aarch64, compiler_x64, ir::IRFunction};
use crate::{
    compiler_aarch64::Aarch64Compiler,
    disassembler::disassemble,
    ir::{BlockReference, IndexedInstruction, InputSlot, Instruction, InstructionType, OutputSlot},
    register_allocator::{Register, Value},
};

fn compile_instruction<'a, TO, TC: Compiler<'a, TO>>(ops: &mut TO, compiler: &TC, instruction: &IndexedInstruction) {
    let instruction_index = instruction.index;
    let block_index = instruction.block_index;
    match &instruction.instruction {
        Instruction::Instruction { tp, inputs, outputs } => {
            let output_regs = outputs
                .iter()
                .enumerate()
                .map(|(output_index, output)| {
                    compiler
                        .get_allocations()
                        .get(&Value::InstructionOutput {
                            instruction_index,
                            output_index,
                            block_index,
                            data_type: output.tp,
                        })
                        .map(|r| *r)
                })
                .collect::<Vec<_>>();

            match tp {
                InstructionType::Add => compiler.add(ops, inputs, outputs, output_regs),
                InstructionType::Compare => compiler.compare(ops, inputs, output_regs),
                InstructionType::LoadPtr => compiler.load_ptr(ops, inputs, outputs, output_regs),
                InstructionType::WritePtr => compiler.write_ptr(ops, inputs),
                InstructionType::SpillToStack => compiler.spill_to_stack(ops, inputs),
                InstructionType::LoadFromStack => compiler.load_from_stack(ops, inputs, outputs, output_regs),
            }
        }
        Instruction::Branch {
            cond,
            if_true,
            if_false,
        } => compiler.branch(ops, cond, if_true, if_false),
        Instruction::Jump { target } => compiler.call_block(ops, target),
        Instruction::Return { value } => compiler.ret(ops, value),
    }
}

pub trait Compiler<'a, T> {
    /// Creates a new Compiler object and sets up the function for compilation. Should also emit
    /// the prologue.
    fn new(ops: &mut T, func: &'a mut IRFunction) -> Self;
    /// Emit the function epilogue.
    fn epilogue(&self, ops: &mut T);

    /// Called whenever a new block is beginning to be compiled
    fn on_new_block_begin(&self, ops: &mut T, block_index: usize);

    /// Gets the func object this Compiler is compiling
    fn get_func(&self) -> &IRFunction;
    /// Gets all the register allocations for this function
    fn get_allocations(&self) -> &HashMap<Value, Register>;
    /// Gets an offset to the entry point of this function
    fn get_entrypoint(&self) -> AssemblyOffset;

    /// Emit a jump to another block + move all inputs into place
    fn call_block(&self, ops: &mut T, target: &BlockReference);
    /// Conditionally emit a jump to one of two blocks + move all inputs into place
    fn branch(&self, ops: &mut T, cond: &InputSlot, if_true: &BlockReference, if_false: &BlockReference);
    /// Emit a return with an optional value
    fn ret(&self, ops: &mut T, value: &Option<InputSlot>);

    /// Compile an IR add instruction
    fn add(&self, ops: &mut T, inputs: &Vec<InputSlot>, outputs: &Vec<OutputSlot>, output_regs: Vec<Option<Register>>);
    /// Compile an IR compare instruction
    fn compare(&self, ops: &mut T, inputs: &Vec<InputSlot>, output_regs: Vec<Option<Register>>);
    /// Compile an IR load pointer instruction
    fn load_ptr(
        &self,
        ops: &mut T,
        inputs: &Vec<InputSlot>,
        outputs: &Vec<OutputSlot>,
        output_regs: Vec<Option<Register>>,
    );
    /// Compile an IR write pointer instruction
    fn write_ptr(&self, ops: &mut T, inputs: &Vec<InputSlot>);
    /// Compile an IR spill to stack instruction
    fn spill_to_stack(&self, ops: &mut T, inputs: &Vec<InputSlot>);
    /// Compile an IR load from stack instruction
    fn load_from_stack(
        &self,
        ops: &mut T,
        inputs: &Vec<InputSlot>,
        outputs: &Vec<OutputSlot>,
        output_regs: Vec<Option<Register>>,
    );
}

/// Compile an IR function into machine code
pub fn compile(func: &mut IRFunction) {
    #[cfg(target_arch = "aarch64")]
    let mut ops = dynasmrt::aarch64::Assembler::new().unwrap();
    let compiler = Aarch64Compiler::new(&mut ops, func);

    println!("Allocations:");
    compiler
        .get_allocations()
        .iter()
        .sorted_by_key(|(v, _)| **v)
        .for_each(|(k, v)| {
            println!("{} \t {:?}", k, v);
        });

    for (block_index, block) in compiler.get_func().blocks.iter().enumerate() {
        compiler.on_new_block_begin(&mut ops, block_index);
        block
            .instructions
            .iter()
            .map(|i| &compiler.get_func().instructions[*i])
            .for_each(|instruction| compile_instruction::<_, _>(&mut ops, &compiler, instruction))
    }
    compiler.epilogue(&mut ops);

    let code = ops.finalize().unwrap();
    let f: extern "C" fn() = unsafe { mem::transmute(code.ptr(compiler.get_entrypoint())) };

    println!("{}", disassemble(&code, f as u64));

    println!("Running:");
    f();
}
