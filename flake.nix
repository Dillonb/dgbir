{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs =
    { self, nixpkgs }:
    let
      supportedSystems = [
        "x86_64-linux"
        "x86_64-darwin"
        "aarch64-linux"
        "aarch64-darwin"
      ];
      forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
      pkgs = forAllSystems (system: nixpkgs.legacyPackages.${system});
    in
    {
      apps = forAllSystems (system: {
        cargo-tests = {
          type = "app";
          meta.description = "Run the dgbir test suite with cargo-nextest";
          program = nixpkgs.lib.getExe (
            pkgs.${system}.writeShellApplication {
              name = "cargo-tests";
              runtimeInputs = with pkgs.${system}; [
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
      });

      devShells = forAllSystems (system: {
        default = pkgs.${system}.mkShell {
          buildInputs = with pkgs.${system}; [
            cargo
            cargo-outdated
            rustc
            rust-analyzer
            rustfmt
            lldb
            bacon
            cargo-nextest
          ];
        };
      });
    };
}
