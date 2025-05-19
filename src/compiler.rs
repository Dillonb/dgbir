use std::{
    collections::{HashMap, HashSet},
    mem,
};

use dynasmrt::AssemblyOffset;

#[allow(unused_imports)]
use crate::{compiler_aarch64, compiler_x64, ir::IRFunction};
use crate::{
    compiler_aarch64::Aarch64Compiler,
    disassembler::disassemble,
    ir::{
        BlockReference, CompareType, Constant, DataType, IndexedInstruction, InputSlot, Instruction, InstructionType,
    },
    register_allocator::{Register, RegisterAllocations, Value},
};

#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
pub enum ConstOrReg {
    U32(u32),
    U64(u64),
    GPR(u32),
}

impl ConstOrReg {
    pub fn to_reg(&self) -> Option<Register> {
        match self {
            ConstOrReg::GPR(r) => Some(Register::GPR(*r as usize)),
            ConstOrReg::U32(_) => None,
            ConstOrReg::U64(_) => None,
        }
    }
}

impl Register {
    pub fn to_const_or_reg(&self) -> ConstOrReg {
        match self {
            Register::GPR(r) => ConstOrReg::GPR(*r as u32),
        }
    }
}

fn compile_instruction<'a, Ops, TC: Compiler<'a, Ops>>(ops: &mut Ops, compiler: &TC, instruction: &IndexedInstruction) {
    let instruction_index = instruction.index;
    let block_index = instruction.block_index;
    match &instruction.instruction {
        Instruction::Instruction { tp, inputs, outputs } => {
            let output_regs = outputs
                .iter()
                .enumerate()
                .map(|(output_index, output)| {
                    compiler.get_allocations().get(&Value::InstructionOutput {
                        instruction_index,
                        output_index,
                        block_index,
                        data_type: output.tp,
                    })
                })
                .collect::<Vec<_>>();

            match tp {
                InstructionType::Add => {
                    let a = compiler.to_imm_or_reg(&inputs[0]);
                    let b = compiler.to_imm_or_reg(&inputs[1]);
                    let tp = outputs[0].tp;
                    let r_out = output_regs[0].unwrap();
                    compiler.add(ops, tp, r_out, a, b);
                }

                InstructionType::Compare => {
                    assert_eq!(inputs.len(), 3);
                    assert_eq!(outputs.len(), 1);

                    // TODO: remove this #allow when the Register enum has more than just GPR in it (this warning will go away)
                    #[allow(irrefutable_let_patterns)]
                    if let Register::GPR(r_out) = output_regs[0].unwrap() {
                        if let InputSlot::Constant(Constant::CompareType(cmp_type)) = inputs[1] {
                            let a = compiler.to_imm_or_reg(&inputs[0]);
                            let b = compiler.to_imm_or_reg(&inputs[2]);
                            compiler.compare(ops, r_out, a, cmp_type, b);
                        } else {
                            panic!("Expected compare type constant as second input, got {:?}", inputs[1]);
                        }
                    } else {
                        panic!("Output register is not a GPR: {:?}", output_regs[0]);
                    }
                }

                InstructionType::LoadPtr => {
                    assert_eq!(inputs.len(), 1);
                    assert_eq!(outputs.len(), 1);
                    let ptr = compiler.to_imm_or_reg(&inputs[0]);
                    let tp = outputs[0].tp;
                    let r_out = output_regs[0].unwrap();

                    compiler.load_ptr(ops, r_out, tp, ptr);
                }
                InstructionType::WritePtr => {
                    assert_eq!(inputs.len(), 3);
                    // ptr, value, type
                    let ptr = compiler.to_imm_or_reg(&inputs[0]);
                    let value = compiler.to_imm_or_reg(&inputs[1]);
                    if let InputSlot::Constant(Constant::DataType(data_type)) = inputs[2] {
                        compiler.write_ptr(ops, ptr, value, data_type);
                    } else {
                        panic!("Expected data type constant as third input, got {:?}", inputs[2]);
                    }
                }
                InstructionType::SpillToStack => {
                    assert_eq!(inputs.len(), 3);
                    if let InputSlot::Constant(Constant::DataType(tp)) = inputs[2] {
                        let to_spill = compiler.to_imm_or_reg(&inputs[0]);
                        let stack_offset = compiler.to_imm_or_reg(&inputs[1]);
                        compiler.spill_to_stack(ops, to_spill, stack_offset, tp);
                    } else {
                        panic!("Expected data type constant as third input, got {:?}", inputs[2]);
                    }
                }
                InstructionType::LoadFromStack => {
                    assert_eq!(inputs.len(), 1);
                    assert_eq!(outputs.len(), 1);
                    let r_out = output_regs[0].unwrap();
                    let stack_offset = compiler.to_imm_or_reg(&inputs[0]);
                    let tp = outputs[0].tp;
                    compiler.load_from_stack(ops, r_out, stack_offset, tp);
                }
            }
        }
        Instruction::Branch {
            cond,
            if_true,
            if_false,
        } => compiler.branch(ops, &compiler.to_imm_or_reg(cond), if_true, if_false),
        Instruction::Jump { target } => compiler.call_block(ops, target),
        Instruction::Return { value } => compiler.ret(ops, value),
    }
}

pub trait Compiler<'a, Ops> {
    // Utility functions, shouldn't be overridden
    fn to_imm_or_reg(&self, s: &InputSlot) -> ConstOrReg {
        match *s {
            InputSlot::InstructionOutput { .. } | InputSlot::BlockInput { .. } => {
                match self.get_allocations().allocations[&s.to_value(self.get_func()).unwrap()] {
                    Register::GPR(r) => ConstOrReg::GPR(r as u32),
                }
            }
            InputSlot::Constant(constant) => match constant {
                Constant::U32(c) => ConstOrReg::U32(c as u32),
                Constant::U8(_) => todo!(),
                Constant::S8(_) => todo!(),
                Constant::U16(_) => todo!(),
                Constant::S16(_) => todo!(),
                Constant::S32(_) => todo!(),
                Constant::U64(c) => ConstOrReg::U64(c as u64),
                Constant::S64(_) => todo!(),
                Constant::F32(_) => todo!(),
                Constant::F64(_) => todo!(),
                Constant::Ptr(c) => ConstOrReg::U64(c as u64),
                Constant::Bool(_) => todo!(),
                Constant::DataType(_) => todo!(),
                Constant::CompareType(_) => todo!(),
            },
            // _ => todo!("Unsupported input slot type: {:?}", s),
        }
    }

    // Functions that must be overridden by the different architecture bacends
    /// Creates a new Compiler object and sets up the function for compilation.
    fn new(ops: &mut Ops, func: &'a mut IRFunction) -> Self;
    /// Emit the function prologue.
    fn prologue(&self, ops: &mut Ops);
    /// Emit the function epilogue.
    fn epilogue(&self, ops: &mut Ops);

    /// Called whenever a new block is beginning to be compiled
    fn on_new_block_begin(&self, ops: &mut Ops, block_index: usize);

    /// Gets the func object this Compiler is compiling
    fn get_func(&self) -> &IRFunction;
    /// Gets all the register allocations for this function
    fn get_allocations(&self) -> &RegisterAllocations;
    /// Gets an offset to the entry point of this function
    fn get_entrypoint(&self) -> AssemblyOffset;

    /// Emit a jump to another block + move all inputs into place
    fn call_block(&self, ops: &mut Ops, target: &BlockReference);
    /// Conditionally emit a jump to one of two blocks + move all inputs into place
    fn branch(&self, ops: &mut Ops, cond: &ConstOrReg, if_true: &BlockReference, if_false: &BlockReference);
    /// Emit a return with an optional value
    fn ret(&self, ops: &mut Ops, value: &Option<InputSlot>);

    /// Compile an IR add instruction
    fn add(&self, ops: &mut Ops, tp: DataType, r_out: Register, a: ConstOrReg, b: ConstOrReg);
    /// Compile an IR compare instruction
    fn compare(&self, ops: &mut Ops, r_out: usize, a: ConstOrReg, cmp_type: CompareType, b: ConstOrReg);
    /// Compile an IR load pointer instruction
    fn load_ptr(&self, ops: &mut Ops, r_out: Register, tp: DataType, ptr: ConstOrReg);
    /// Compile an IR write pointer instruction
    fn write_ptr(&self, ops: &mut Ops, ptr: ConstOrReg, value: ConstOrReg, data_type: DataType);
    /// Compile an IR spill to stack instruction
    fn spill_to_stack(&self, ops: &mut Ops, to_spill: ConstOrReg, stack_offset: ConstOrReg, tp: DataType);
    /// Compile an IR load from stack instruction
    fn load_from_stack(&self, ops: &mut Ops, r_out: Register, stack_offset: ConstOrReg, tp: DataType);
}

/// Moves a set of values into a set of registers. The sets can overlap, but the algorithm assumes
/// each move is unique. That is, {A->C, B->C} is not allowed, but {A->B, B->C, C->A} is allowed.
/// Moves to self (A->A) will be ignored.
/// - `moves` All moves in the format of from -> to
/// - `do_move` Emit a move from the source to the target. When this lambda is called, it is
///            guaranteed to be safe to do the move.
pub fn move_regs_multi(mut moves: HashMap<ConstOrReg, Register>, mut do_move: impl FnMut(ConstOrReg, Register)) {
    let mut pending_move_targets = HashSet::new();
    let mut pending_move_sources = HashSet::new();

    for (from, to) in &moves {
        // assert true: A register can be the target of only one move per run of this algorithm
        assert_eq!(pending_move_targets.insert(*to), true);
        if let ConstOrReg::GPR(r) = *from {
            // assert true: A register can be the source of only one move per run of this algorithm
            assert_eq!(pending_move_sources.insert(Register::GPR(r as usize)), true);
        }
    }

    // Used to reorder the moves in case there's a conflict
    let mut postponed_moves = Vec::new();
    while moves.len() > 0 {
        postponed_moves.push(*moves.keys().next().unwrap());

        while let Some(from) = postponed_moves.pop() {
            let to = moves[&from];
            if Some(to) == from.to_reg() {
                // If it's a move to self, no-op.
                moves.remove(&from);
                pending_move_targets.remove(&to);
                pending_move_sources.remove(&to);
            } else if pending_move_sources.contains(&to) {
                if postponed_moves.contains(&to.to_const_or_reg()) {
                    // What I think I have to do here is:
                    //
                    // Allocate a temporary register for the move target.
                    //
                    // Move the target into the temporary register.
                    //
                    // Remove the move of the target to _its target_ from the move queue and replace
                    // it with a move from the temp reg to the target's target.
                    //
                    // Add this move back to postponed_moves and `continue`
                    panic!("Would overwrite a pending move target - we have a cycle. Need to allocate temp regs to fix this.");
                } else {
                    // We couldn't make this move, so we need to add it back to the list of moves
                    postponed_moves.push(from);
                    // But do the conflicting one first
                    postponed_moves.push(to.to_const_or_reg());
                }
            } else {
                // It is safe to do the move. It's not a self-move, and it doesn't conflict with any other moves.
                do_move(from, to);

                moves.remove(&from);
                from.to_reg().iter().for_each(|r| {
                    pending_move_sources.remove(r);
                });
                pending_move_targets.remove(&to);
            }
        }
    }
}

/// Compile an IR function into machine code
pub fn compile(func: &mut IRFunction) {
    #[cfg(target_arch = "aarch64")]
    let mut ops = dynasmrt::aarch64::Assembler::new().unwrap();
    let compiler = Aarch64Compiler::new(&mut ops, func);

    compiler.prologue(&mut ops);

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
