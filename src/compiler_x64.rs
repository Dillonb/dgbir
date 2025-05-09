use std::mem;

use crate::{disassembler::disassemble, ir::IRFunction};
use dynasmrt::{dynasm, DynasmApi};

pub fn compile(_func: IRFunction) {
    let mut ops = dynasmrt::x64::Assembler::new().unwrap();

    let add_one = ops.offset();
    dynasm!(ops
        ; .arch x64
        ; mov rax, rdi
        ; add rax, 1
        ; ret
    );

    let code = ops.finalize().unwrap();
    let hello_fn: extern "C" fn(x: u64) -> u64 = unsafe { mem::transmute(code.ptr(add_one)) };

    println!("{}", disassemble(&code, hello_fn as u64));

    println!("1+1 = {}", hello_fn(1));
}
