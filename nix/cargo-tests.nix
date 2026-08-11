{
  perSystem =
    { pkgs, system, ... }:
    {
      apps.cargo-tests = {
        type = "app";
        meta.description = "Run the dgbir test suite with cargo-nextest";
        program = pkgs.lib.getExe (
          pkgs.writeShellApplication {
            name = "cargo-tests";
            runtimeInputs = with pkgs; [
              cargo
              rustc
              cargo-nextest
              stdenv.cc
            ];
            text = ''
              export CARGO_TARGET_DIR="''${CARGO_TARGET_DIR:-target/nix-${system}}"
              exec cargo nextest run "$@"
            '';
          }
        );
      };
    };
}
