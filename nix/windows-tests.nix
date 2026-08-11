{ inputs, ... }:
let
  pkgs = inputs.nixpkgs.legacyPackages.x86_64-linux;
  inherit (pkgs.pkgsCross) mingwW64;
  cc = mingwW64.stdenv.cc;
  pthreads = mingwW64.windows.mingw_w64_pthreads;
  # nixpkgs' native rustc has no windows-gnu std.
  windowsRustc = mingwW64.buildPackages.rustc;
in
{
  # Wine runs x86_64 PE binaries natively, so this only works on an x86_64 host.
  flake.apps.x86_64-linux.windows-tests = {
    type = "app";
    meta.description = "Cross-compile the dgbir test suite for Windows and run it under Wine";
    program = pkgs.lib.getExe (
      pkgs.writeShellApplication {
        name = "windows-tests";
        runtimeInputs = with pkgs; [
          cargo
          cargo-nextest
        ];
        text = ''
          export RUSTC="${windowsRustc}/bin/rustc"
          export CARGO_BUILD_TARGET="x86_64-pc-windows-gnu"
          export CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER="${cc}/bin/${cc.targetPrefix}gcc"
          export CARGO_TARGET_X86_64_PC_WINDOWS_GNU_RUSTFLAGS="-L native=${pthreads}/lib"
          # The wine64 package installs a single `wine` binary.
          export CARGO_TARGET_X86_64_PC_WINDOWS_GNU_RUNNER="${pkgs.wine64}/bin/wine"
          export CC_x86_64_pc_windows_gnu="${cc}/bin/${cc.targetPrefix}gcc"
          export AR_x86_64_pc_windows_gnu="${cc.bintools}/bin/${cc.targetPrefix}ar"
          export CARGO_TARGET_DIR="''${CARGO_TARGET_DIR:-target/nix-windows}"

          export WINEPATH="${pthreads}/bin"
          # Wine requires an absolute path.
          export WINEPREFIX="''${WINEPREFIX:-$PWD/target/wine}"
          export WINEDEBUG="''${WINEDEBUG:--all}"

          exec cargo nextest run "$@"
        '';
      }
    );
  };
}
