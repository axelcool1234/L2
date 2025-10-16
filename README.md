# Install
I use [Nix](https://nixos.org) to manage the dependencies.
Install Nix using the Determinate Systems installer which automatically sets up flakes and supports [easy uninstallation](https://github.com/DeterminateSystems/nix-installer#uninstalling):
```bash
curl -fsSL https://install.determinate.systems/nix | sh -s -- install
```

# Enter the Nix Shell
`nix develop "github:axelcool1234/L2#impure"`

# Run the Program
`nix run "github:axelcool1234/L2"`
