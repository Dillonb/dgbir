## Running tests

For testing on this machine's native architecture, just run `cargo test`.

You can use Nix to run tests for other architectures. For example, to run the ARM64 tests from an x86_64 machine, you can run:

```bash
nix run .#apps.aarch64-linux.cargo-tests
```

