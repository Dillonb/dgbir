use crate::{compiler_aarch64, ir::IRFunction};

pub fn compile(func: IRFunction) {
    compiler_aarch64::compile(func);
}
