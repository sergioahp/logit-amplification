{
  description = "Development environment with torch and transformers";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        
        pythonPackages = pkgs.python3.withPackages (ps: with ps; [
          # Core ML packages
          torch
          transformers
          accelerate
          jax
          jaxtyping
          datasets
          zstandard
          peft
          matplotlib
          openai
          
          # Common dependencies
          numpy
          
          # Development tools
          pip
          pytest
        ]);
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pythonPackages
          ];
        };
      }
    );
}
