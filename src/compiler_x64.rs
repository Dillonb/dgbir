use crate::{
    compiler::{Compiler, ConstOrReg},
    ir::{BlockReference, CompareType, DataType, IRFunction, InputSlot},
    reg_pool::RegPool,
    register_allocator::{Register, RegisterAllocations},
};
use dynasmrt::x64::X64Relocation;

type Ops = dynasmrt::Assembler<X64Relocation>;

pub struct X64Compiler<'a> {
    scratch_regs: RegPool,
    func: &'a IRFunction,
    allocations: RegisterAllocations,
    callee_saved: Vec<(Register, usize)>,
    entrypoint: dynasmrt::AssemblyOffset,
    block_labels: Vec<dynasmrt::DynamicLabel>,
}

impl<'a> Compiler<'a, Ops> for X64Compiler<'a> {
    fn new(ops: &mut Ops, func: &'a mut IRFunction) -> Self {
        todo!()
    }

    fn prologue(&self, ops: &mut Ops) {
        todo!()
    }

    fn epilogue(&self, ops: &mut Ops) {
        todo!()
    }

    fn on_new_block_begin(&self, ops: &mut Ops, block_index: usize) {
        todo!()
    }

    fn get_func(&self) -> &IRFunction {
        todo!()
    }

    fn get_allocations(&self) -> &RegisterAllocations {
        todo!()
    }

    fn get_entrypoint(&self) -> dynasmrt::AssemblyOffset {
        todo!()
    }

    fn call_block(&self, ops: &mut Ops, target: &BlockReference) {
        todo!()
    }

    fn branch(&self, ops: &mut Ops, cond: &ConstOrReg, if_true: &BlockReference, if_false: &BlockReference) {
        todo!()
    }

    fn ret(&self, ops: &mut Ops, value: &Option<InputSlot>) {
        todo!()
    }

    fn add(&self, ops: &mut Ops, tp: DataType, r_out: Register, a: ConstOrReg, b: ConstOrReg) {
        todo!()
    }

    fn compare(&self, ops: &mut Ops, r_out: usize, a: ConstOrReg, cmp_type: CompareType, b: ConstOrReg) {
        todo!()
    }

    fn load_ptr(&self, ops: &mut Ops, r_out: Register, tp: DataType, ptr: ConstOrReg) {
        todo!()
    }

    fn write_ptr(&self, ops: &mut Ops, ptr: ConstOrReg, value: ConstOrReg, data_type: DataType) {
        todo!()
    }

    fn spill_to_stack(&self, ops: &mut Ops, to_spill: ConstOrReg, stack_offset: ConstOrReg, tp: DataType) {
        todo!()
    }

    fn load_from_stack(&self, ops: &mut Ops, r_out: Register, stack_offset: ConstOrReg, tp: DataType) {
        todo!()
    }
}
