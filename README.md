dgbir is a simple, SSA intermediate language for JIT compilers. It is built for emulators, but designed to be general purpose. Workload specific features are part of extensions. It is designed to be architecture-agnostic, but assumes you are targeting a modern 64 bit host with a flags register.

# Basic Syntax

A basic instruction consists of a value assignment with a type, and an operation:
```
v2 : u32  = operation(v1)
```

Operations can also return multiple values of different types:
```
v2 : u32, v3 : s64  = operation(v1)
```

Multiple returned values can be accessed explicitly by their name
```
.result(v3 : u32), .flags(v4 : flags) = add(v1, v2)
```

Not all operations return a result.

# Type system
## Integer types
### Signed
s8, s16, s32, s64
### Unsigned
u8, u16, u32, u64
## Float types
f32, f64
## Vector types
128 bit: v + integer or float type.
256 bit: dv + integer or float type.
512 bit: qv + integer or float type.

examples:
- vs8 = vector of 16 s8 values
- vs16 = vector of 8 s16 values
- dvs8 = vector of 32 s8 values
## Other types
ptr
	64 bit host pointer
flags
	host flag register value
