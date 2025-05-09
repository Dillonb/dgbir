use capstone::{arch::BuildsCapstone, Capstone};

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

pub fn disassemble(code: &[u8], addr: u64) -> String {
    // Disassemble the code

    let cs = get_capstone();

    let insns = cs.disasm_all(code, addr).unwrap();

    // for insn in insns.iter() {
    insns
        .iter()
        .map(|insn| {
            format!(
                "0x{:x}:\t{}\t{}",
                insn.address(),
                insn.mnemonic().unwrap(),
                insn.op_str().unwrap()
            )
        })
        .collect::<Vec<String>>()
        .join("\n")
}
