pub mod compiler;
#[cfg(target_arch = "aarch64")]
pub mod compiler_aarch64;

#[cfg(target_arch = "x86_64")]
pub mod compiler_x64;

pub mod abi;
pub mod disassembler;
pub mod ir;
pub mod ir_interpreter;
pub mod parser;
pub mod reg_pool;
pub mod register_allocator;
