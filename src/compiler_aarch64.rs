use std::mem;

use crate::{disassembler::disassemble, ir::IRFunction, register_allocator::alloc_for};
use dynasmrt::{dynasm, DynasmApi};

pub fn compile(func: &mut IRFunction) {
    let register_mappings = alloc_for(func);

    let mut ops = dynasmrt::aarch64::Assembler::new().unwrap();

    let add_one = ops.offset();
    dynasm!(ops
        ; .arch aarch64
        ; add x0, x0, 1
        ; ret
    );

    let code = ops.finalize().unwrap();
    let hello_fn: extern "C" fn(x: u64) -> u64 = unsafe { mem::transmute(code.ptr(add_one)) };

    println!("{}", disassemble(&code, hello_fn as u64));

    println!("1+1 = {}", hello_fn(1));
}
