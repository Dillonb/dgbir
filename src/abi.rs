use crate::register_allocator::Register;

#[allow(dead_code)]
#[cfg(target_arch = "x86_64")]
mod reg_constants {
    use crate::register_allocator::Register;

    // X64
    pub const RAX: Register = Register::GPR(0);
    pub const RCX: Register = Register::GPR(1);
    pub const RDX: Register = Register::GPR(2);
    pub const RBX: Register = Register::GPR(3);
    pub const RSP: Register = Register::GPR(4);
    pub const RBP: Register = Register::GPR(5);
    pub const RSI: Register = Register::GPR(6);
    pub const RDI: Register = Register::GPR(7);
    pub const R8: Register = Register::GPR(8);
    pub const R9: Register = Register::GPR(9);
    pub const R10: Register = Register::GPR(10);
    pub const R11: Register = Register::GPR(11);
    pub const R12: Register = Register::GPR(12);
    pub const R13: Register = Register::GPR(13);
    pub const R14: Register = Register::GPR(14);
    pub const R15: Register = Register::GPR(15);

    pub const XMM0: Register = Register::SIMD(0);
    pub const XMM1: Register = Register::SIMD(1);
    pub const XMM2: Register = Register::SIMD(2);
    pub const XMM3: Register = Register::SIMD(3);
    pub const XMM4: Register = Register::SIMD(4);
    pub const XMM5: Register = Register::SIMD(5);
    pub const XMM6: Register = Register::SIMD(6);
    pub const XMM7: Register = Register::SIMD(7);
    pub const XMM8: Register = Register::SIMD(8);
    pub const XMM9: Register = Register::SIMD(9);
    pub const XMM10: Register = Register::SIMD(10);
    pub const XMM11: Register = Register::SIMD(11);
    pub const XMM12: Register = Register::SIMD(12);
    pub const XMM13: Register = Register::SIMD(13);
    pub const XMM14: Register = Register::SIMD(14);
    pub const XMM15: Register = Register::SIMD(15);
}

#[allow(dead_code)]
#[cfg(target_arch = "aarch64")]
mod reg_constants {
    use crate::register_allocator::Register;

    // AArch64
    pub const X0: Register = Register::GPR(0);
    pub const X1: Register = Register::GPR(1);
    pub const X2: Register = Register::GPR(2);
    pub const X3: Register = Register::GPR(3);
    pub const X4: Register = Register::GPR(4);
    pub const X5: Register = Register::GPR(5);
    pub const X6: Register = Register::GPR(6);
    pub const X7: Register = Register::GPR(7);
    pub const X8: Register = Register::GPR(8);
    pub const X9: Register = Register::GPR(9);
    pub const X10: Register = Register::GPR(10);
    pub const X11: Register = Register::GPR(11);
    pub const X12: Register = Register::GPR(12);
    pub const X13: Register = Register::GPR(13);
    pub const X14: Register = Register::GPR(14);
    pub const X15: Register = Register::GPR(15);
    pub const X16: Register = Register::GPR(16);
    pub const X17: Register = Register::GPR(17);
    pub const X18: Register = Register::GPR(18);
    pub const X19: Register = Register::GPR(19);
    pub const X20: Register = Register::GPR(20);
    pub const X21: Register = Register::GPR(21);
    pub const X22: Register = Register::GPR(22);
    pub const X23: Register = Register::GPR(23);
    pub const X24: Register = Register::GPR(24);
    pub const X25: Register = Register::GPR(25);
    pub const X26: Register = Register::GPR(26);
    pub const X27: Register = Register::GPR(27);
    pub const X28: Register = Register::GPR(28);
    pub const X29: Register = Register::GPR(29);
    pub const X30: Register = Register::GPR(30);
    pub const SP: Register = Register::GPR(31);

    pub const V0: Register = Register::SIMD(0);
    pub const V1: Register = Register::SIMD(1);
    pub const V2: Register = Register::SIMD(2);
    pub const V3: Register = Register::SIMD(3);
    pub const V4: Register = Register::SIMD(4);
    pub const V5: Register = Register::SIMD(5);
    pub const V6: Register = Register::SIMD(6);
    pub const V7: Register = Register::SIMD(7);
    pub const V8: Register = Register::SIMD(8);
    pub const V9: Register = Register::SIMD(9);
    pub const V10: Register = Register::SIMD(10);
    pub const V11: Register = Register::SIMD(11);
    pub const V12: Register = Register::SIMD(12);
    pub const V13: Register = Register::SIMD(13);
    pub const V14: Register = Register::SIMD(14);
    pub const V15: Register = Register::SIMD(15);
    pub const V16: Register = Register::SIMD(16);
    pub const V17: Register = Register::SIMD(17);
    pub const V18: Register = Register::SIMD(18);
    pub const V19: Register = Register::SIMD(19);
    pub const V20: Register = Register::SIMD(20);
    pub const V21: Register = Register::SIMD(21);
    pub const V22: Register = Register::SIMD(22);
    pub const V23: Register = Register::SIMD(23);
    pub const V24: Register = Register::SIMD(24);
    pub const V25: Register = Register::SIMD(25);
    pub const V26: Register = Register::SIMD(26);
    pub const V27: Register = Register::SIMD(27);
    pub const V28: Register = Register::SIMD(28);
    pub const V29: Register = Register::SIMD(29);
    pub const V30: Register = Register::SIMD(30);
    pub const V31: Register = Register::SIMD(31);
}

pub fn get_registers() -> Vec<Register> {
    use reg_constants::*;
    // Callee-saved registers
    #[cfg(target_arch = "aarch64")]
    {
        vec![
            X19, X20, X21, X22, X23, X24, X25, X26, X27, X28, V8, V9, V10, V11, V12, V13, V14, V15,
        ]
    }

    // For x64, it matters whether we're on Linux or Windows
    #[cfg(target_arch = "x86_64")]
    {
        // rbx, rsp, rbp, r12, r13, r14, r15
        // but we can't use rbp and rsp - so we just have to use rbx, r12, r13, r14, r15
        #[cfg(target_os = "linux")]
        {
            vec![
                RBX, R12, R13, R14, R15,
                // These aren't callee-saved, but none are on this ABI. So, we make do.
                // Use the regs not also used for function arguments.
                XMM8, XMM9, XMM10, XMM11, XMM12, XMM13, XMM14, XMM15,
            ]
        }
        #[cfg(target_os = "windows")]
        {
            todo!("Preserved regs on x64 Windows")
        }
    }
}

pub fn get_scratch_registers() -> Vec<Register> {
    use reg_constants::*;
    #[cfg(target_arch = "aarch64")]
    {
        vec![
            X9, X10, X11, X12, X13, X14, X15, V16, V17, V18, V19, V20, V21, V22, V23, V24, V25, V26, V27, V28, V29,
            V30, V31,
        ]
    }

    // For x64, it matters whether we're on Linux or Windows
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(target_os = "linux")]
        {
            vec![
                RAX, RDI, RSI, RDX, RCX, R8, R9, R10, R11, XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7,
            ]
        }
        #[cfg(target_os = "windows")]
        {
            todo!("Scratch regs on x64 Windows")
        }
    }
}

pub fn get_function_argument_registers() -> Vec<Register> {
    use reg_constants::*;
    #[cfg(target_arch = "aarch64")]
    {
        vec![X0, X1, X2, X3, X4, X5, X6, X7, V0, V1, V2, V3, V4, V5, V6, V7]
    }
    // For x64, it matters whether we're on Linux or Windows
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(target_os = "linux")]
        {
            vec![
                RDI, RSI, RDX, RCX, R8, R9, XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7,
            ]
        }
        #[cfg(target_os = "windows")]
        {
            vec![RCX, RDX, R8, R9] // TODO SIMD registers
        }
    }
}

pub fn get_return_value_registers() -> Vec<Register> {
    use reg_constants::*;
    #[cfg(target_arch = "aarch64")]
    {
        // Technically, it's x0-x7,v0-v7, but we only support returning one value
        return vec![X0, V0];
    }
    #[cfg(target_arch = "x86_64")]
    {
        vec![RAX, XMM0]
    }
}
