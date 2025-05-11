#[allow(unused_imports)]
use crate::{compiler_aarch64, compiler_x64, ir::IRFunction};

pub fn compile(func: &mut IRFunction) {
    #[cfg(target_arch = "x86_64")]
    compiler_x64::compile(func);

    #[cfg(target_arch = "aarch64")]
    compiler_aarch64::compile(func);
}
