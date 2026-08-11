## Running tests

For testing on this machine's native architecture, just run `cargo test`.

You can use Nix to run tests for other architectures. For example, to run the ARM64 tests from an x86_64 machine, you can run:

```bash
nix run .#apps.aarch64-linux.cargo-tests
```

The Windows tests can be run from an x86_64 Linux machine. This cross compiles to `x86_64-pc-windows-gnu`, which uses the same ABI as MSVC, and runs the tests under Wine:

```bash
nix run .#apps.x86_64-linux.windows-tests
```

Wine keeps its state in `target/wine`, so `cargo clean` removes it.

