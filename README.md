# dgbir

dgbir is a simple, SSA intermediate language for JIT compilers. It is built for emulators, but designed to be general purpose. It is designed to be architecture-agnostic, but assumes you are targeting a modern 64 bit host.

Currently, x86_64 and ARM64 are supported, but the language is designed to be easily extensible to other architectures. RISC-V64 support is planned.
