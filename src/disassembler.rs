use capstone::{arch::BuildsCapstone, Capstone};

use crate::compiler::{CompiledFunction, CompiledFunctionDebugInfo, CompiledFunctionVec};

#[cfg(target_arch = "aarch64")]
fn get_capstone() -> Capstone {
    Capstone::new()
        .arm64()
        .mode(capstone::arch::arm64::ArchMode::Arm)
        .build()
        .unwrap()
}

#[cfg(target_arch = "x86_64")]
fn get_capstone() -> Capstone {
    Capstone::new()
        .x86()
        .mode(capstone::arch::x86::ArchMode::Mode64)
        .build()
        .unwrap()
}

fn disassemble_internal(code: &[u8], addr: u64, debug_info: &CompiledFunctionDebugInfo) -> String {
    // Disassemble the code

    let cs = get_capstone();

    let insns = cs.disasm_all(code, addr).unwrap();

    insns
        .iter()
        .map(|insn| {
            let offset = insn.address() - addr;
            let comment = debug_info.comments_at_offset(offset as usize).map(|comments| {
                comments
                    .iter()
                    .fold(String::new(), |acc, c| {
                        acc  + "\n// " + c
                    })
            });
            let disasm = format!("0x{:x}:\t{}\t{}", insn.address(), insn.mnemonic().unwrap(), insn.op_str().unwrap());

            if let Some(comment) = comment {
                format!("\n{}\n{}", comment, disasm)
            } else {
                disasm
            }
        })
        .collect::<Vec<String>>()
        .join("\n")
}

pub fn disassemble_vec_function(func: &CompiledFunctionVec) -> String {
    disassemble_internal(&func.code, func.ptr_entrypoint() as u64, &func.debug_info)
}

pub fn disassemble_function(func: &CompiledFunction) -> String {
    disassemble_internal(&func.code, func.ptr_entrypoint() as u64, &func.debug_info)
}

pub fn disassemble(code: &[u8], addr: u64) -> String {
    disassemble_internal(code, addr, &CompiledFunctionDebugInfo::new())
}
