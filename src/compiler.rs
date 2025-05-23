use std::collections::{HashMap, HashSet};

use dynasmrt::{AssemblyOffset, ExecutableBuffer};
use ordered_float::OrderedFloat;

use crate::abi::get_function_argument_registers;
#[cfg(target_arch = "aarch64")]
use crate::compiler_aarch64;
#[cfg(target_arch = "x86_64")]
use crate::compiler_x64;
use crate::ir::IRFunction;
use crate::{
    ir::{
        BlockReference, CompareType, Constant, DataType, IndexedInstruction, InputSlot, Instruction, InstructionType,
    },
    register_allocator::{Register, RegisterAllocations, Value},
};

#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
pub enum ConstOrReg {
    U32(u32),
    U64(u64),
    F32(OrderedFloat<f32>),
    GPR(u32),
    SIMD(u32),
}

impl ConstOrReg {
    pub fn to_reg(&self) -> Option<Register> {
        match self {
            ConstOrReg::GPR(r) => Some(Register::GPR(*r as usize)),
            ConstOrReg::SIMD(r) => Some(Register::SIMD(*r as usize)),
            ConstOrReg::U32(_) => None,
            ConstOrReg::U64(_) => None,
            ConstOrReg::F32(_) => None,
        }
    }

    pub fn is_same_type_as(&self, other: &Register) -> bool {
        match (self, other) {
            (ConstOrReg::U32(_), Register::GPR(_)) => true,
            (ConstOrReg::U32(_), Register::SIMD(_)) => false,
            (ConstOrReg::U64(_), Register::GPR(_)) => true,
            (ConstOrReg::U64(_), Register::SIMD(_)) => false,
            (ConstOrReg::F32(_), Register::GPR(_)) => false,
            (ConstOrReg::F32(_), Register::SIMD(_)) => true,
            (ConstOrReg::GPR(_), Register::GPR(_)) => true,
            (ConstOrReg::GPR(_), Register::SIMD(_)) => false,
            (ConstOrReg::SIMD(_), Register::GPR(_)) => false,
            (ConstOrReg::SIMD(_), Register::SIMD(_)) => true,
        }
    }
}

impl Register {
    pub fn to_const_or_reg(&self) -> ConstOrReg {
        match self {
            Register::GPR(r) => ConstOrReg::GPR(*r as u32),
            Register::SIMD(r) => ConstOrReg::SIMD(*r as u32),
        }
    }
}

fn expect_constant_data_type(input: &InputSlot) -> DataType {
    if let InputSlot::Constant(Constant::DataType(data_type)) = input {
        *data_type
    } else {
        panic!("Expected data type constant, got {:?}", input);
    }
}

fn expect_constant_cmp_type(input: &InputSlot) -> CompareType {
    if let InputSlot::Constant(Constant::CompareType(cmp_type)) = input {
        *cmp_type
    } else {
        panic!("Expected compare type constant, got {:?}", input);
    }
}

fn expect_constant_u64(input: &InputSlot) -> u64 {
    if let InputSlot::Constant(c) = input {
        match c {
            Constant::U64(value) => *value,
            Constant::U32(value) => *value as u64,
            Constant::U8(value) => *value as u64,
            Constant::S64(value) if *value >= 0 => *value as u64,
            Constant::S16(value) if *value >= 0 => *value as u64,
            Constant::S8(value) if *value >= 0 => *value as u64,
            Constant::Ptr(value) => *value as u64,
            _ => panic!("Expected unsigned, positive, or ptr constant, got {:?}", input),
        }
    } else {
        panic!("Expected u64 constant, got {:?}", input);
    }
}

pub fn expect_gpr(register: Register) -> usize {
    match register {
        Register::GPR(r) => r,
        _ => panic!("Expected GPR, got {:?}", register),
    }
}

fn compile_instruction<'a, Ops, TC: Compiler<'a, Ops>>(
    ops: &mut Ops,
    lp: &mut LiteralPool,
    compiler: &TC,
    instruction: &IndexedInstruction,
) {
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
                    compiler.add(ops, lp, tp, r_out, a, b);
                }
                InstructionType::Compare => {
                    assert_eq!(inputs.len(), 3);
                    assert_eq!(outputs.len(), 1);
                    let r_out = expect_gpr(output_regs[0].unwrap());
                    let a = compiler.to_imm_or_reg(&inputs[0]);
                    let cmp_type = expect_constant_cmp_type(&inputs[1]);
                    let b = compiler.to_imm_or_reg(&inputs[2]);
                    compiler.compare(ops, r_out, a, cmp_type, b);
                }
                InstructionType::LoadPtr => {
                    assert_eq!(inputs.len(), 2);
                    assert_eq!(outputs.len(), 1);
                    let ptr = compiler.to_imm_or_reg(&inputs[0]);
                    let offset = expect_constant_u64(&inputs[1]);
                    let tp = outputs[0].tp;
                    let r_out = output_regs[0].unwrap();
                    compiler.load_ptr(ops, lp, r_out, tp, ptr, offset);
                }
                InstructionType::WritePtr => {
                    assert_eq!(inputs.len(), 4);
                    // ptr, offset, value, type
                    let ptr = compiler.to_imm_or_reg(&inputs[0]);
                    let offset = expect_constant_u64(&inputs[1]);
                    let value = compiler.to_imm_or_reg(&inputs[2]);
                    let data_type = expect_constant_data_type(&inputs[3]);
                    compiler.write_ptr(ops, lp, ptr, offset, value, data_type);
                }
                InstructionType::SpillToStack => {
                    assert_eq!(inputs.len(), 3);
                    let to_spill = compiler.to_imm_or_reg(&inputs[0]);
                    let stack_offset = compiler.to_imm_or_reg(&inputs[1]);
                    let tp = expect_constant_data_type(&inputs[2]);
                    compiler.spill_to_stack(ops, to_spill, stack_offset, tp);
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
        } => compiler.branch(ops, lp, &compiler.to_imm_or_reg(cond), if_true, if_false),
        Instruction::Jump { target } => compiler.call_block(ops, lp, target),
        Instruction::Return { value } => {
            let v = value.map(|v| compiler.to_imm_or_reg(&v));
            compiler.ret(ops, lp, &v);
        }
    }
}

pub struct LiteralPool {
    pub literals: HashMap<Constant, dynasmrt::DynamicLabel>,
}

impl LiteralPool {
    pub fn new() -> Self {
        Self {
            literals: HashMap::new(),
        }
    }
}

pub trait Compiler<'a, Ops> {
    /// Get a new dynamic label.
    fn new_dynamic_label(ops: &mut Ops) -> dynasmrt::DynamicLabel;

    // Utility functions, shouldn't be overridden
    fn to_imm_or_reg(&self, s: &InputSlot) -> ConstOrReg {
        match *s {
            InputSlot::InstructionOutput { .. } | InputSlot::BlockInput { .. } => {
                match self.get_allocations().allocations[&s.to_value(self.get_func()).unwrap()] {
                    Register::GPR(r) => ConstOrReg::GPR(r as u32),
                    Register::SIMD(r) => ConstOrReg::SIMD(r as u32),
                }
            }
            InputSlot::Constant(constant) => match constant {
                Constant::U32(c) => ConstOrReg::U32(c),
                Constant::U8(_) => todo!(),
                Constant::S8(_) => todo!(),
                Constant::U16(_) => todo!(),
                Constant::S16(_) => todo!(),
                Constant::S32(_) => todo!(),
                Constant::U64(c) => ConstOrReg::U64(c),
                Constant::S64(_) => todo!(),
                Constant::F32(c) => ConstOrReg::F32(c),
                Constant::F64(_) => todo!(),
                Constant::Ptr(c) => ConstOrReg::U64(c as u64),
                Constant::Bool(_) => todo!(),
                Constant::DataType(_) => todo!(),
                Constant::CompareType(_) => todo!(),
            },
            // _ => todo!("Unsupported input slot type: {:?}", s),
        }
    }

    /// Moves a set of values into a set of registers. The sets can overlap, but the algorithm assumes
    /// each move is unique. That is, {A->C, B->C} is not allowed, but {A->B, B->C, C->A} is allowed.
    /// Moves to self (A->A) will be ignored.
    /// - `moves` All moves in the format of from -> to
    /// - `do_move` Emit a move from the source to the target. When this lambda is called, it is
    ///            guaranteed to be safe to do the move.
    fn move_regs_multi(&self, ops: &mut Ops, lp: &mut LiteralPool, mut moves: HashMap<ConstOrReg, Register>) {
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
                    // do_move(from, to);
                    self.move_to_reg(ops, lp, from, to);

                    moves.remove(&from);
                    from.to_reg().iter().for_each(|r| {
                        pending_move_sources.remove(r);
                    });
                    pending_move_targets.remove(&to);
                }
            }
        }
    }

    fn call_block(&self, ops: &mut Ops, lp: &mut LiteralPool, target: &BlockReference) {
        let moves = target
            .arguments
            .iter()
            .enumerate()
            .map(|(input_index, arg)| {
                let data_type = self.get_func().blocks[target.block_index].inputs[input_index];

                let in_block_value = Value::BlockInput {
                    block_index: target.block_index,
                    input_index,
                    data_type,
                };

                let block_arg_reg = self.get_allocations().get(&in_block_value).unwrap();
                (self.to_imm_or_reg(&arg), block_arg_reg)
            })
            .collect::<HashMap<_, _>>();

        if moves.len() > 0 {
            self.move_regs_multi(ops, lp, moves);
        }

        // TODO: figure out when we can elide this jump. If it's a jmp instruction to the next block,
        // we definitely can, but it gets more complicated when it's a branch instruction. Maybe the
        // branch instruction should detect if one of the targets is the next block and always put that
        // second. Then we could always elide this jump here.
        // if target.block_index != from_block_index + 1 {
        let target_label = self.get_block_label(target.block_index);
        self.jump_to_dynamic_label(ops, target_label);
        // }
    }

    fn handle_function_arguments(&self, ops: &mut Ops, lp: &mut LiteralPool) {
        // Move all arguments into the correct registers

        let arg_regs = get_function_argument_registers();
        let mut allocated_arg_regs = HashSet::new();
        self.get_func().blocks[0]
            .inputs
            .iter()
            .enumerate()
            .for_each(|(input_index, input)| {
                let block_input = Value::BlockInput {
                    block_index: 0,
                    input_index,
                    data_type: *input,
                };
                let input_reg = self.get_allocations().get(&block_input).unwrap();
                let arg_reg = arg_regs
                    .iter()
                    .find(|r| r.is_same_type_as(&input_reg) && !allocated_arg_regs.contains(r))
                    .unwrap();
                allocated_arg_regs.insert(arg_reg);
                self.move_to_reg(ops, lp, arg_reg.to_const_or_reg(), input_reg);
            });
    }

    fn add_literal(ops: &mut Ops, lp: &mut LiteralPool, literal: Constant) -> dynasmrt::DynamicLabel {
        *lp.literals.entry(literal).or_insert(Self::new_dynamic_label(ops))
    }

    // Functions that must be overridden by the different architecture bacends
    /// Creates a new Compiler object and sets up the function for compilation.
    fn new(ops: &mut Ops, func: &'a mut IRFunction) -> Self;
    /// Emit the function prologue.
    fn prologue(&self, ops: &mut Ops);
    /// Emit the function epilogue.
    fn epilogue(&self, ops: &mut Ops);
    /// Emit a jump to a dynamic label.
    fn jump_to_dynamic_label(&self, ops: &mut Ops, label: dynasmrt::DynamicLabel);
    /// Emit a move from a register or value to a register.
    fn move_to_reg(&self, ops: &mut Ops, lp: &mut LiteralPool, from: ConstOrReg, to: Register);
    /// Emits the literal pool and resolve all the labels. Called at the end of the function.
    fn emit_literal_pool(&self, ops: &mut Ops, lp: LiteralPool);

    /// Called whenever a new block is beginning to be compiled
    fn on_new_block_begin(&self, ops: &mut Ops, block_index: usize);

    /// Gets the func object this Compiler is compiling
    fn get_func(&self) -> &IRFunction;
    /// Gets all the register allocations for this function
    fn get_allocations(&self) -> &RegisterAllocations;
    /// Gets an offset to the entry point of this function
    fn get_entrypoint(&self) -> AssemblyOffset;
    /// Gets a label for a block index
    fn get_block_label(&self, block_index: usize) -> dynasmrt::DynamicLabel;

    /// Conditionally emit a jump to one of two blocks + move all inputs into place
    fn branch(&self, ops: &mut Ops, lp: &mut LiteralPool, cond: &ConstOrReg, if_true: &BlockReference, if_false: &BlockReference);
    /// Emit a return with an optional value
    fn ret(&self, ops: &mut Ops, lp: &mut LiteralPool, value: &Option<ConstOrReg>);

    /// Compile an IR add instruction
    fn add(&self, ops: &mut Ops, lp: &mut LiteralPool, tp: DataType, r_out: Register, a: ConstOrReg, b: ConstOrReg);
    /// Compile an IR compare instruction
    fn compare(&self, ops: &mut Ops, r_out: usize, a: ConstOrReg, cmp_type: CompareType, b: ConstOrReg);
    /// Compile an IR load pointer instruction
    fn load_ptr(&self, ops: &mut Ops, lp: &mut LiteralPool, r_out: Register, tp: DataType, ptr: ConstOrReg, offset: u64);
    /// Compile an IR write pointer instruction
    fn write_ptr(&self, ops: &mut Ops, lp: &mut LiteralPool, ptr: ConstOrReg, offset: u64, value: ConstOrReg, data_type: DataType);
    /// Compile an IR spill to stack instruction
    fn spill_to_stack(&self, ops: &mut Ops, to_spill: ConstOrReg, stack_offset: ConstOrReg, tp: DataType);
    /// Compile an IR load from stack instruction
    fn load_from_stack(&self, ops: &mut Ops, r_out: Register, stack_offset: ConstOrReg, tp: DataType);
}

pub struct CompiledFunction {
    pub entrypoint: AssemblyOffset,
    pub code: ExecutableBuffer,
}

impl CompiledFunction {
    pub fn ptr_entrypoint(&self) -> *const u8 {
        self.code.ptr(self.entrypoint)
    }
}

/// Compile an IR function into machine code
pub fn compile(func: &mut IRFunction) -> CompiledFunction {
    #[cfg(target_arch = "x86_64")]
    let mut ops = dynasmrt::x64::Assembler::new().unwrap();
    #[cfg(target_arch = "aarch64")]
    let mut ops = dynasmrt::aarch64::Assembler::new().unwrap();

    #[cfg(target_arch = "aarch64")]
    let compiler = compiler_aarch64::Aarch64Compiler::new(&mut ops, func);
    #[cfg(target_arch = "x86_64")]
    let compiler = compiler_x64::X64Compiler::new(&mut ops, func);

    let mut lp = LiteralPool::new();

    compiler.prologue(&mut ops);
    compiler.handle_function_arguments(&mut ops, &mut lp);

    for (block_index, block) in compiler.get_func().blocks.iter().enumerate() {
        compiler.on_new_block_begin(&mut ops, block_index);
        block
            .instructions
            .iter()
            .map(|i| &compiler.get_func().instructions[*i])
            .for_each(|instruction| compile_instruction::<_, _>(&mut ops, &mut lp, &compiler, instruction))
    }
    compiler.epilogue(&mut ops);
    compiler.emit_literal_pool(&mut ops, lp);

    return CompiledFunction {
        entrypoint: compiler.get_entrypoint(),
        code: ops.finalize().unwrap(),
    };
}
