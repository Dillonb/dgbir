use std::collections::{BTreeMap, HashSet};

#[cfg(target_arch = "aarch64")]
use dynasmrt::aarch64::Aarch64Relocation;
use dynasmrt::relocations::Relocation;
#[cfg(target_arch = "x86_64")]
use dynasmrt::x64::X64Relocation;
use dynasmrt::{AssemblyOffset, DynamicLabel, DynasmApi, DynasmLabelApi, ExecutableBuffer};
use ordered_float::OrderedFloat;

use crate::abi::get_function_argument_registers;
#[cfg(target_arch = "aarch64")]
use crate::compiler_aarch64;
#[cfg(target_arch = "x86_64")]
use crate::compiler_x64;
use crate::ir::{IRFunction, IRFunctionInternal};
use crate::{
    ir::{
        BlockReference, CompareType, Constant, DataType, IndexedInstruction, InputSlot, Instruction, InstructionType,
    },
    register_allocator::{Register, RegisterAllocations, Value},
};

// A custom trait that acts as a generic assembler interface. Works with both VecAssembler and
// Assembler.
pub trait GenericAssembler<R: Relocation>: DynasmApi + DynasmLabelApi<Relocation = R> {
    type R;
    fn new_dynamic_label(&mut self) -> DynamicLabel;
}

#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq, PartialOrd, Ord)]
pub enum ConstOrReg {
    U16(u16),
    S16(i16),
    U32(u32),
    S32(i32),
    U64(u64),
    F32(OrderedFloat<f32>),
    GPR(u32),
    SIMD(u32),
    S64(i64),
}

impl ConstOrReg {
    pub fn to_reg(&self) -> Option<Register> {
        match self {
            ConstOrReg::GPR(r) => Some(Register::GPR(*r as usize)),
            ConstOrReg::SIMD(r) => Some(Register::SIMD(*r as usize)),
            ConstOrReg::U16(_) => None,
            ConstOrReg::S16(_) => None,
            ConstOrReg::U32(_) => None,
            ConstOrReg::S32(_) => None,
            ConstOrReg::U64(_) => None,
            ConstOrReg::S64(_) => None,
            ConstOrReg::F32(_) => None,
        }
    }

    pub fn to_u64_const(&self) -> Option<u64> {
        match self {
            ConstOrReg::U16(c) => Some(*c as u64),
            ConstOrReg::S16(c) => Some(*c as u64),
            ConstOrReg::U32(c) => Some(*c as u64),
            ConstOrReg::S32(c) => Some(*c as u64),
            ConstOrReg::U64(c) => Some(*c),
            ConstOrReg::S64(c) => Some(*c as u64),
            ConstOrReg::F32(_) => None,
            ConstOrReg::GPR(_) => None,
            ConstOrReg::SIMD(_) => None,
        }
    }

    pub fn is_const(&self) -> bool {
        match self {
            ConstOrReg::U16(_) => true,
            ConstOrReg::S16(_) => true,
            ConstOrReg::U32(_) => true,
            ConstOrReg::S32(_) => true,
            ConstOrReg::U64(_) => true,
            ConstOrReg::S64(_) => true,
            ConstOrReg::F32(_) => true,
            ConstOrReg::GPR(_) => false,
            ConstOrReg::SIMD(_) => false,
        }
    }

    pub fn to_s64_const(&self) -> Option<i64> {
        match self {
            ConstOrReg::U16(c) => (*c).try_into().ok(),
            ConstOrReg::S16(c) => (*c).try_into().ok(),
            ConstOrReg::U32(c) => (*c).try_into().ok(),
            ConstOrReg::S32(c) => (*c).try_into().ok(),
            ConstOrReg::U64(c) => (*c).try_into().ok(),
            ConstOrReg::S64(c) => Some(*c),
            ConstOrReg::F32(_) => None,
            ConstOrReg::GPR(_) => None,
            ConstOrReg::SIMD(_) => None,
        }
    }

    // Can this type go into this register?
    pub fn is_same_type_as(&self, other: &Register) -> bool {
        match (self, other) {
            (ConstOrReg::GPR(_), Register::GPR(_)) => true,
            (ConstOrReg::GPR(_), Register::SIMD(_)) => false,

            (ConstOrReg::U16(_), Register::GPR(_)) => true,
            (ConstOrReg::U16(_), Register::SIMD(_)) => false,

            (ConstOrReg::S16(_), Register::GPR(_)) => true,
            (ConstOrReg::S16(_), Register::SIMD(_)) => false,

            (ConstOrReg::U32(_), Register::GPR(_)) => true,
            (ConstOrReg::U32(_), Register::SIMD(_)) => false,

            (ConstOrReg::S32(_), Register::GPR(_)) => true,
            (ConstOrReg::S32(_), Register::SIMD(_)) => false,

            (ConstOrReg::U64(_), Register::GPR(_)) => true,
            (ConstOrReg::U64(_), Register::SIMD(_)) => false,

            (ConstOrReg::S64(_), Register::GPR(_)) => true,
            (ConstOrReg::S64(_), Register::SIMD(_)) => false,

            (ConstOrReg::F32(_), Register::GPR(_)) => false,
            (ConstOrReg::F32(_), Register::SIMD(_)) => true,

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

fn compile_instruction<'a, R: Relocation, Ops: GenericAssembler<R>, TC: Compiler<'a, R, Ops>>(
    ops: &mut Ops,
    lp: &mut LiteralPool,
    compiler: &TC,
    instruction_index_in_block: usize,
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
                    output_regs[0].iter().for_each(|r_out| {
                        compiler.add(ops, lp, tp, *r_out, a, b);
                    });
                }
                InstructionType::Compare => {
                    assert_eq!(inputs.len(), 3);
                    assert_eq!(outputs.len(), 1);
                    let a = compiler.to_imm_or_reg(&inputs[0]);
                    let cmp_type = expect_constant_cmp_type(&inputs[1]);
                    let b = compiler.to_imm_or_reg(&inputs[2]);
                    output_regs[0].iter().for_each(|r_out| {
                        compiler.compare(ops, lp, expect_gpr(*r_out), a, cmp_type, b);
                    });
                }
                InstructionType::LoadPtr => {
                    assert_eq!(inputs.len(), 2);
                    assert_eq!(outputs.len(), 1);
                    let ptr = compiler.to_imm_or_reg(&inputs[0]);
                    let offset = expect_constant_u64(&inputs[1]);
                    let tp = outputs[0].tp;
                    output_regs[0].iter().for_each(|r_out| {
                        compiler.load_ptr(ops, lp, *r_out, tp, ptr, offset);
                    });
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
                    let stack_offset = compiler.to_imm_or_reg(&inputs[0]);
                    let tp = outputs[0].tp;
                    output_regs[0].iter().for_each(|r_out| {
                        compiler.load_from_stack(ops, *r_out, stack_offset, tp);
                    });
                }
                InstructionType::LeftShift => {
                    assert_eq!(inputs.len(), 2);
                    assert_eq!(outputs.len(), 1);
                    let n = compiler.to_imm_or_reg(&inputs[0]);
                    let amount = compiler.to_imm_or_reg(&inputs[1]);
                    let tp = outputs[0].tp;
                    output_regs[0].iter().for_each(|r_out| {
                        compiler.left_shift(ops, expect_gpr(*r_out), n, amount, tp);
                    });
                }
                InstructionType::RightShift => {
                    assert_eq!(inputs.len(), 2);
                    assert_eq!(outputs.len(), 1);
                    let n = compiler.to_imm_or_reg(&inputs[0]);
                    let amount = compiler.to_imm_or_reg(&inputs[1]);
                    let tp = outputs[0].tp;
                    output_regs[0].iter().for_each(|r_out| {
                        compiler.right_shift(ops, expect_gpr(*r_out), n, amount, tp);
                    });
                }
                InstructionType::Convert => {
                    assert_eq!(outputs.len(), 1);
                    assert_eq!(inputs.len() > 0, true); // need at least one input
                    let input = compiler.to_imm_or_reg(&inputs[0]);

                    let from_tp = if inputs.len() == 2 {
                        // If we have two arguments, the second argument is the data type to
                        // convert from.
                        expect_constant_data_type(&inputs[1])
                    } else if inputs.len() == 1 {
                        // Otherwise, use the input's own type
                        inputs[0].tp()
                    } else {
                        panic!("Expected 1 or 2 inputs for convert instruction, got {}", inputs.len());
                    };

                    let to_tp = outputs[0].tp;
                    output_regs[0].iter().for_each(|r_out| {
                        compiler.convert(ops, lp, *r_out, input, from_tp, to_tp);
                    })
                }
                InstructionType::And => {
                    assert_eq!(inputs.len(), 2);
                    assert_eq!(outputs.len(), 1);
                    let a = compiler.to_imm_or_reg(&inputs[0]);
                    let b = compiler.to_imm_or_reg(&inputs[1]);
                    let tp = outputs[0].tp;
                    output_regs[0].iter().for_each(|r_out| {
                        compiler.and(ops, lp, tp, *r_out, a, b);
                    });
                }
                InstructionType::Or => {
                    assert_eq!(inputs.len(), 2);
                    assert_eq!(outputs.len(), 1);
                    let a = compiler.to_imm_or_reg(&inputs[0]);
                    let b = compiler.to_imm_or_reg(&inputs[1]);
                    let tp = outputs[0].tp;
                    output_regs[0].iter().for_each(|r_out| {
                        compiler.or(ops, lp, tp, *r_out, a, b);
                    });
                }
                InstructionType::Not => {
                    assert_eq!(inputs.len(), 1);
                    assert_eq!(outputs.len(), 1);
                    let a = compiler.to_imm_or_reg(&inputs[0]);
                    let tp = outputs[0].tp;
                    output_regs[0].iter().for_each(|r_out| {
                        compiler.not(ops, lp, tp, *r_out, a);
                    });
                }
                InstructionType::Xor => {
                    assert_eq!(inputs.len(), 2);
                    assert_eq!(outputs.len(), 1);
                    let a = compiler.to_imm_or_reg(&inputs[0]);
                    let b = compiler.to_imm_or_reg(&inputs[1]);
                    let tp = outputs[0].tp;
                    output_regs[0].iter().for_each(|r_out| {
                        compiler.xor(ops, lp, tp, *r_out, a, b);
                    });
                }
                InstructionType::Subtract => {
                    assert_eq!(inputs.len(), 2);
                    assert_eq!(outputs.len(), 1);
                    let minuend = compiler.to_imm_or_reg(&inputs[0]);
                    let subtrahend = compiler.to_imm_or_reg(&inputs[1]);
                    let tp = outputs[0].tp;
                    output_regs[0].iter().for_each(|r_out| {
                        compiler.subtract(ops, lp, tp, *r_out, minuend, subtrahend);
                    });
                }
                InstructionType::Multiply => {
                    assert_eq!(inputs.len(), 3);
                    assert_eq!(outputs.len() == 1 || outputs.len() == 2, true);
                    let a = compiler.to_imm_or_reg(&inputs[0]);
                    let b = compiler.to_imm_or_reg(&inputs[1]);
                    let arg_tp = expect_constant_data_type(&inputs[2]);
                    let result_tp = outputs[0].tp;
                    compiler.multiply(ops, lp, result_tp, arg_tp, output_regs, a, b);
                }
                InstructionType::Divide => {
                    assert_eq!(inputs.len(), 2);
                    assert_eq!(outputs.len(), 2);
                    let dividend = compiler.to_imm_or_reg(&inputs[0]);
                    let divisor = compiler.to_imm_or_reg(&inputs[1]);
                    let r_quotient = output_regs[0];
                    let r_remainder = output_regs[1];
                    let tp = outputs[0].tp;
                    compiler.divide(ops, lp, tp, r_quotient, r_remainder, dividend, divisor);
                }
                InstructionType::SquareRoot => {
                    assert_eq!(inputs.len(), 1);
                    assert_eq!(outputs.len(), 1);
                    let value = compiler.to_imm_or_reg(&inputs[0]);
                    let tp = outputs[0].tp;
                    output_regs[0].iter().for_each(|r_out| {
                        compiler.square_root(ops, lp, tp, *r_out, value);
                    });
                }
                InstructionType::AbsoluteValue => {
                    assert_eq!(inputs.len(), 1);
                    assert_eq!(outputs.len(), 1);
                    let value = compiler.to_imm_or_reg(&inputs[0]);
                    let tp = outputs[0].tp;
                    output_regs[0].iter().for_each(|r_out| {
                        compiler.absolute_value(ops, lp, tp, *r_out, value);
                    });
                }
                InstructionType::Negate => {
                    assert_eq!(inputs.len(), 1);
                    assert_eq!(outputs.len(), 1);
                    let value = compiler.to_imm_or_reg(&inputs[0]);
                    let tp = outputs[0].tp;
                    output_regs[0].iter().for_each(|r_out| {
                        compiler.negate(ops, lp, tp, *r_out, value);
                    });
                }
                InstructionType::CallFunction => {
                    assert_eq!(inputs.len() >= 1, true);
                    assert_eq!(outputs.len() <= 1, true);
                    let address = compiler.to_imm_or_reg(&inputs[0]);
                    let args = inputs[1..]
                        .iter()
                        .map(|i| compiler.to_imm_or_reg(i))
                        .collect::<Vec<_>>();
                    let r_out = output_regs.get(0).map(|r| *r).flatten();
                    let allocations = compiler.get_allocations();

                    let active_volatile_regs = allocations
                        .lifetimes
                        .get_active_at_index(&compiler.get_func(), block_index, instruction_index_in_block)
                        .iter()
                        .map(|v| allocations.get(v).unwrap())
                        .filter(|r| r.is_volatile())
                        .collect::<Vec<_>>();

                    compiler.call_function(ops, lp, address, active_volatile_regs, r_out, args);
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
    pub literals: BTreeMap<Constant, dynasmrt::DynamicLabel>,
}

impl LiteralPool {
    pub fn new() -> Self {
        Self {
            literals: BTreeMap::new(),
        }
    }
}

pub trait Compiler<'a, R: Relocation, Ops: GenericAssembler<R>> {
    /// Get a new dynamic label.
    fn new_dynamic_label(ops: &mut Ops) -> dynasmrt::DynamicLabel;

    // Utility functions, shouldn't be overridden
    fn to_imm_or_reg(&self, s: &InputSlot) -> ConstOrReg {
        match *s {
            InputSlot::InstructionOutput { .. } | InputSlot::BlockInput { .. } => {
                match self.get_allocations().allocations[&s.to_value(&self.get_func()).unwrap()] {
                    Register::GPR(r) => ConstOrReg::GPR(r as u32),
                    Register::SIMD(r) => ConstOrReg::SIMD(r as u32),
                }
            }
            InputSlot::Constant(constant) => match constant {
                Constant::U8(_) => todo!(),
                Constant::S8(_) => todo!(),
                Constant::U16(c) => ConstOrReg::U16(c),
                Constant::S16(c) => ConstOrReg::S16(c),
                Constant::U32(c) => ConstOrReg::U32(c),
                Constant::S32(c) => ConstOrReg::S32(c),
                Constant::U64(c) => ConstOrReg::U64(c),
                Constant::S64(c) => ConstOrReg::S64(c),
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
    fn move_regs_multi(&self, ops: &mut Ops, lp: &mut LiteralPool, mut moves: BTreeMap<ConstOrReg, Register>) {
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
            .collect::<BTreeMap<_, _>>();

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
    fn new(ops: &mut Ops, func: &'a mut IRFunctionInternal) -> Self;
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
    fn get_func(&self) -> &IRFunctionInternal;
    /// Gets all the register allocations for this function
    fn get_allocations(&self) -> &RegisterAllocations;
    /// Gets an offset to the entry point of this function
    fn get_entrypoint(&self) -> AssemblyOffset;
    /// Gets a label for a block index
    fn get_block_label(&self, block_index: usize) -> dynasmrt::DynamicLabel;

    /// Conditionally emit a jump to one of two blocks + move all inputs into place
    fn branch(
        &self,
        ops: &mut Ops,
        lp: &mut LiteralPool,
        cond: &ConstOrReg,
        if_true: &BlockReference,
        if_false: &BlockReference,
    );
    /// Emit a return with an optional value
    fn ret(&self, ops: &mut Ops, lp: &mut LiteralPool, value: &Option<ConstOrReg>);

    /// Compile an IR add instruction
    fn add(&self, ops: &mut Ops, lp: &mut LiteralPool, tp: DataType, r_out: Register, a: ConstOrReg, b: ConstOrReg);
    /// Compile an IR compare instruction
    fn compare(
        &self,
        ops: &mut Ops,
        lp: &mut LiteralPool,
        r_out: usize,
        a: ConstOrReg,
        cmp_type: CompareType,
        b: ConstOrReg,
    );
    /// Compile an IR load pointer instruction
    fn load_ptr(
        &self,
        ops: &mut Ops,
        lp: &mut LiteralPool,
        r_out: Register,
        tp: DataType,
        ptr: ConstOrReg,
        offset: u64,
    );
    /// Compile an IR write pointer instruction
    fn write_ptr(
        &self,
        ops: &mut Ops,
        lp: &mut LiteralPool,
        ptr: ConstOrReg,
        offset: u64,
        value: ConstOrReg,
        data_type: DataType,
    );
    /// Compile an IR spill to stack instruction
    fn spill_to_stack(&self, ops: &mut Ops, to_spill: ConstOrReg, stack_offset: ConstOrReg, tp: DataType);
    /// Compile an IR load from stack instruction
    fn load_from_stack(&self, ops: &mut Ops, r_out: Register, stack_offset: ConstOrReg, tp: DataType);
    /// Compile an IR left shift instruction
    fn left_shift(&self, ops: &mut Ops, r_out: usize, n: ConstOrReg, amount: ConstOrReg, tp: DataType);
    /// Compile an IR right shift instruction
    fn right_shift(&self, ops: &mut Ops, r_out: usize, n: ConstOrReg, amount: ConstOrReg, tp: DataType);
    /// Compile an IR convert instruction
    fn convert(
        &self,
        ops: &mut Ops,
        lp: &mut LiteralPool,
        r_out: Register,
        input: ConstOrReg,
        from_tp: DataType,
        to_tp: DataType,
    );
    /// Compile an IR bitwise AND instruction
    fn and(&self, ops: &mut Ops, lp: &mut LiteralPool, tp: DataType, r_out: Register, a: ConstOrReg, b: ConstOrReg);
    /// Compile an IR bitwise OR instruction
    fn or(&self, ops: &mut Ops, lp: &mut LiteralPool, tp: DataType, r_out: Register, a: ConstOrReg, b: ConstOrReg);
    /// Compile an IR bitwise NOT instruction
    fn not(&self, ops: &mut Ops, lp: &mut LiteralPool, tp: DataType, r_out: Register, a: ConstOrReg);
    /// Compile an IR bitwise XOR instruction
    fn xor(&self, ops: &mut Ops, lp: &mut LiteralPool, tp: DataType, r_out: Register, a: ConstOrReg, b: ConstOrReg);
    /// Compile an IR subtract instruction
    fn subtract(
        &self,
        ops: &mut Ops,
        lp: &mut LiteralPool,
        tp: DataType,
        r_out: Register,
        minuend: ConstOrReg,
        subtrahend: ConstOrReg,
    );
    /// Compile an IR multiply instruction
    fn multiply(
        &self,
        ops: &mut Ops,
        lp: &mut LiteralPool,
        result_tp: DataType,
        arg_tp: DataType,
        output_regs: Vec<Option<Register>>,
        a: ConstOrReg,
        b: ConstOrReg,
    );
    /// Compile an IR divide instruction
    fn divide(
        &self,
        ops: &mut Ops,
        lp: &mut LiteralPool,
        tp: DataType,
        r_quotient: Option<Register>,
        r_remainder: Option<Register>,
        dividend: ConstOrReg,
        divisor: ConstOrReg,
    );
    /// Compile an IR square root instruction
    fn square_root(&self, ops: &mut Ops, lp: &mut LiteralPool, tp: DataType, r_out: Register, value: ConstOrReg);
    /// Compile an IR absolute value instruction
    fn absolute_value(&self, ops: &mut Ops, lp: &mut LiteralPool, tp: DataType, r_out: Register, value: ConstOrReg);
    /// Compile an IR negate instruction
    fn negate(&self, ops: &mut Ops, lp: &mut LiteralPool, tp: DataType, r_out: Register, value: ConstOrReg);
    /// Compile an IR call instruction
    fn call_function(
        &self,
        ops: &mut Ops,
        lp: &mut LiteralPool,
        address: ConstOrReg,
        active_volatile_regs: Vec<Register>,
        r_out: Option<Register>,
        args: Vec<ConstOrReg>,
    );
}

pub struct CompiledFunction {
    pub entrypoint: AssemblyOffset,
    pub code: ExecutableBuffer,
    // TODO: maybe only include this for debug builds?
    pub allocations: RegisterAllocations,
}

impl CompiledFunction {
    pub fn ptr_entrypoint(&self) -> *const u8 {
        self.code.ptr(self.entrypoint)
    }
}

fn compile_common<'a, R: Relocation, Ops: GenericAssembler<R>, C: Compiler<'a, R, Ops>>(ops: &mut Ops, compiler: &C) {
    let mut lp = LiteralPool::new();

    compiler.prologue(ops);
    compiler.handle_function_arguments(ops, &mut lp);

    for (block_index, block) in compiler.get_func().blocks.iter().enumerate() {
        compiler.on_new_block_begin(ops, block_index);
        block
            .instructions
            .iter()
            .map(|i_in_block| (i_in_block, &compiler.get_func().instructions[*i_in_block]))
            .for_each(|(i_in_block, instruction)| {
                compile_instruction(ops, &mut lp, compiler, *i_in_block, instruction);
            })
    }
    compiler.epilogue(ops);
    compiler.emit_literal_pool(ops, lp);
}

pub fn compile_vec(func: &IRFunction, baseaddr: usize) -> Vec<u8> {
    func.validate();
    #[cfg(target_arch = "x86_64")]
    let mut ops = dynasmrt::VecAssembler::<X64Relocation>::new(baseaddr);
    // TODO fix above for x86_64
    #[cfg(target_arch = "aarch64")]
    let mut ops = dynasmrt::VecAssembler::<Aarch64Relocation>::new(baseaddr);

    let mut func = func.func.borrow_mut();

    #[cfg(target_arch = "aarch64")]
    let compiler = compiler_aarch64::Aarch64Compiler::new(&mut ops, &mut func);
    #[cfg(target_arch = "x86_64")]
    let compiler = compiler_x64::X64Compiler::new(&mut ops, &mut func);

    compile_common(&mut ops, &compiler);

    return ops.finalize().unwrap();
}

/// Compile an IR function into machine code
pub fn compile(func: &IRFunction) -> CompiledFunction {
    func.validate();
    #[cfg(target_arch = "x86_64")]
    let mut ops = dynasmrt::x64::Assembler::new().unwrap();
    #[cfg(target_arch = "aarch64")]
    let mut ops = dynasmrt::aarch64::Assembler::new().unwrap();

    let mut func = func.func.borrow_mut();

    #[cfg(target_arch = "aarch64")]
    let compiler = compiler_aarch64::Aarch64Compiler::new(&mut ops, &mut func);
    #[cfg(target_arch = "x86_64")]
    let compiler = compiler_x64::X64Compiler::new(&mut ops, &mut func);

    compile_common(&mut ops, &compiler);

    return CompiledFunction {
        entrypoint: compiler.get_entrypoint(),
        code: ops.finalize().unwrap(),
        // TODO: maybe only include this for debug builds?
        allocations: compiler.get_allocations().clone(),
    };
}
