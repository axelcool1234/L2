# Install
I use [Nix](https://nixos.org) along with [uv2nix](https://github.com/pyproject-nix/uv2nix) to manage the dependencies.
Install Nix using the Determinate Systems installer which automatically sets up flakes and supports [easy uninstallation](https://github.com/DeterminateSystems/nix-installer#uninstalling):
```bash
curl -fsSL https://install.determinate.systems/nix | sh -s -- install
```

# Run the Compiler/Interpreter
`nix run "github:axelcool1234/L2" [input_file] -- [flags go here]`
## Help
`nix run "github:axelcool1234/L2" -- -h`

# Build the Compiler/Interpreter
`nix build "github:axelcool1234/L2"`
## Run the Built Compiler/Interpreter
`./result/bin/l2 [input_file] [flags go here]`
## Help
`./result/bin/l2 -h`

# Enter the Nix Shell 
One of these two commands will suffice. Typically you only need to do this for development purposes.
- `nix develop "github:axelcool1234/L2#uv2nix"` (preferred)
- `nix develop "github:axelcool1234/L2#impure"`

# Building and Running Without Nix
Nearly all of the dependencies are specified in `pyproject.toml` and `uv.lock`.
If you decide to use `uv` or some other Python package manager, you'll need to make sure that you have the following available:
- Clang
- mlir-opt
- GNU's GMP library

Make sure to install these with your system package manager. Then, you'll need to call:
`clang -c src/dialects/bignum_runtime.c -o bignum_runtime.o`

`export BIGNUM_RUNTIME_PATH=bignum_runtime.o` 
