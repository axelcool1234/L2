{
  description = "LoopLang";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      uv2nix,
      pyproject-nix,
      pyproject-build-systems,
      ...
    }:

    flake-utils.lib.eachDefaultSystem (
      system:
      let
        inherit (nixpkgs) lib;
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
        python = pkgs.python313; # Your desired Python version

        # 1. Load Project Workspace (parses pyproject.toml, uv.lock)
        # Load a uv workspace from a workspace root.
        # Uv2nix treats all uv projects as workspace projects.
        #
        # This function is your entry point. It inspects your workspaceRoot
        # (your project directory) and parses pyproject.toml for project metadata
        # and uv.lock for the exact dependency versions and hashes. The resulting
        # workspace object is a treasure trove of information.
        workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };

        # 2. Generate Nix Overlay from uv.lock (via workspace)
        # Create package overlay from workspace.
        #
        # The workspace object has a handy method, mkPyprojectOverlay.
        # This is where the core uv2nix magic happens. It takes the parsed
        # lock file data and generates a Nix overlay. This overlay tells
        # Nix, "For every Python package requested, if it was in my uv.lock,
        # build that specific version."
        overlay = workspace.mkPyprojectOverlay {
          # Prefer prebuilt binary wheels as a package source.
          # Sdists are less likely to "just work" because of the metadata missing from uv.lock.
          # Binary wheels are more likely to, but may still require overrides for library dependencies.
          sourcePreference = "wheel"; # or sourcePreference = "sdist";
          # Optionally customise PEP 508 environment
          # environ = {
          #   platform_release = "5.10.65";
          # };
        };

        # 3. Placeholder for Your Custom Package Overrides
        # A place for you to fix any stubborn packages that need manual intervention to build in Nix.
        #
        # Extend generated overlay with build fixups
        #
        # Uv2nix can only work with what it has, and uv.lock is missing essential metadata to perform some builds.
        # This is an additional overlay implementing build fixups.
        # See:
        # - https://pyproject-nix.github.io/uv2nix/FAQ.html
        pyprojectOverrides = final: prev: {
          # Implement build fixups here.
          # Note that uv2nix is _not_ using Nixpkgs buildPythonPackage.
          # It's using https://pyproject-nix.github.io/pyproject.nix/build.html
        };

        # 4. Construct the Final Python Package Set
        # Construct package set
        #
        # We start with a base Python package set from pyproject-nix
        # (which provides fundamental Python building capabilities
        # within Nix). Then, we apply a series of overlays.
        pythonSet =
          # Use base package set from pyproject.nix builders
          (pkgs.callPackage pyproject-nix.build.packages {
            inherit python;
          }).overrideScope
            (
              lib.composeManyExtensions [
                # Provides Nix packages for common Python build tools
                # (like Setuptools, Wheel, Hatch). These are needed if
                # uv2nix has to build any of your dependencies from source.
                pyproject-build-systems.overlays.default

                # This is our generated overlay. It ensures that any package
                # also listed in your uv.lock will be resolved to the locked version.
                overlay

                # A place for you to fix any stubborn packages that need manual
                # intervention to build in Nix.
                pyprojectOverrides
              ]
            );

        # After the overlays, your own project (as defined by [project.name] in pyproject.toml)
        # becomes a package within pythonSet. We fetch it (e.g., pythonSet."my-app-name") to get
        # its Nix-ified pname (package name) and version.
        projectNameInToml = "l2"; # MUST match [project.name] in pyproject.toml!
        projectAsNixPkg = pythonSet.${projectNameInToml};

        # 5. Create the Python Runtime Environment
        # Using mkVirtualEnv, we create an environment (virtualenv) that includes the direct dependencies
        # of your project (listed in workspace.deps.default, which comes from pyproject.toml). Because
        # these dependencies are resolved from pythonSet, they respect the versions in your uv.lock.
        virtualenv = pythonSet.mkVirtualEnv (projectAsNixPkg.pname + "-env") workspace.deps.default; # Uses deps from pyproject.toml [project.dependencies]
      in
      {
        # Nix Package for the application
        packages.default = pkgs.stdenv.mkDerivation {
          pname = projectAsNixPkg.pname;
          version = projectAsNixPkg.version;
          src = ./.;

          nativeBuildInputs = [ pkgs.makeWrapper ];
          buildInputs = [
            virtualenv
            pkgs.clang
            pkgs.llvmPackages.mlir
            pkgs.gmp
          ];

          dontUnpack = true;

          buildPhase = ''
            mkdir -p $out/bin
            mkdir -p $out/lib
            mkdir -p $out/tests

            cp -r $src/tests/l2 $out/tests/
            clang -c $src/src/dialects/bignum_runtime.c -o $out/lib/bignum_runtime.o -I${pkgs.gmp.dev}/include

            makeWrapper ${virtualenv}/bin/${projectAsNixPkg.pname} $out/bin/${projectAsNixPkg.pname} \
              --set PATH "${virtualenv}/bin:${pkgs.clang}/bin:${pkgs.llvmPackages.mlir}/bin:$PATH" \
              --set LDFLAGS "-L${pkgs.gmp.out}/lib" \
              --set BIGNUM_RUNTIME_PATH "$out/lib/bignum_runtime.o"
          '';
        };
        packages.${projectAsNixPkg.pname} = self.packages.${system}.default;

        # Nix Package for testing the application
        packages.test =
          # let
          # Construct a virtual environment with only the test dependency-group enabled for testing.
          # testVirtualenv = pythonSet.mkVirtualEnv (projectAsNixPkg.pname + "-test-env") {
          #   l2 = [ "test" ];
          # };
          # in
          pkgs.stdenv.mkDerivation {
            pname = "${projectAsNixPkg.pname}-test";
            version = projectAsNixPkg.version;
            src = ./.;

            buildInputs = [
              self.packages.${system}.default
              pkgs.clang
              pkgs.llvmPackages.mlir

              # With these dependencies, testVirtualenv does not need to be used and sourced
              pkgs.lit
              pkgs.filecheck
            ];

            dontUnpack = true;
            # Because this package is running tests, and not actually building the main package
            # the build phase is running the tests.
            #
            # In this particular example we also output a HTML coverage report, which is used as the build output.
            # NOTE: The following commented out buildPhase uses testVirtualenv, which I have disabled.
            # buildPhase = ''
            #   source ${testVirtualenv}/bin/activate
            #   export PATH="${self.packages.${system}.default}"/bin:$PATH
            #   ${testVirtualenv}/bin/lit -v \
            #     ${self.packages.${system}.default}/tests/l2
            # '';
            buildPhase = ''
              lit -v $src/tests/*
            '';

            installPhase = ''
              mkdir -p $out
              echo 'Tests passed!' > $out/result
            '';
          };

        packages.ruff-check = pkgs.stdenv.mkDerivation {
          pname = "${projectAsNixPkg.pname}-ruff-check";
          version = projectAsNixPkg.version;
          src = ./.;

          buildInputs = [
            pkgs.ruff
          ];

          dontUnpack = true;
          buildPhase = ''
            ruff check $src
          '';

          installPhase = ''
            mkdir -p $out
            echo "Ruff check passed!" > $out/result
          '';
        };

        packages.ruff-format = pkgs.stdenv.mkDerivation {
          pname = "${projectAsNixPkg.pname}-ruff-format";
          version = projectAsNixPkg.version;
          src = ./.;

          buildInputs = [
            pkgs.ruff
          ];

          dontUnpack = true;
          buildPhase = ''
            ruff format --check $src
          '';

          installPhase = ''
            mkdir -p $out
            echo "Ruff format passed!" > $out/result
          '';
        };

        # Pyrefly project integrity check
        packages.pyrefly-check =
          let
            devVirtualenv = pythonSet.mkVirtualEnv (projectAsNixPkg.pname + "-test-env") {
              l2 = [ "dev" ];
            };
          in
          pkgs.stdenv.mkDerivation {
            pname = "${projectAsNixPkg.pname}-pyrefly-check";
            version = projectAsNixPkg.version;
            src = ./.;

            buildInputs = [
              devVirtualenv
            ];

            dontUnpack = true;
            buildPhase = ''
              pyrefly check $src
            '';

            installPhase = ''
              mkdir -p $out
              echo "Pyrefly check passed!" > $out/result
            '';
          };

        # App for `nix run`
        apps.default = {
          type = "app";
          program = "${self.packages.${system}.default}/bin/${projectAsNixPkg.pname}";
        };
        apps.${projectAsNixPkg.pname} = self.apps.${system}.default;

        apps.test = {
          type = "app";
          program = "${pkgs.writeShellScriptBin "run-tests" ''
            cat ${self.packages.${system}.test}/result
          ''}/bin/run-tests"; # This just echoes that it succeeded. It won't even get this far if the tests fail in the buildPhase.
        };
        apps.ruff-check = {
          type = "app";
          program = "${pkgs.writeShellScriptBin "ruff-check" ''
            cat ${self.packages.${system}.ruff-check}/result
          ''}/bin/ruff-check";
        };

        apps.ruff-format = {
          type = "app";
          program = "${pkgs.writeShellScriptBin "ruff-format" ''
            cat ${self.packages.${system}.ruff-format}/result
          ''}/bin/ruff-format";
        };

        apps.pyrefly-check = {
          type = "app";
          program = "${pkgs.writeShellScriptBin "pyrefly-check" ''
            cat ${self.packages.${system}.pyrefly-check}/result
          ''}/bin/pyrefly-check";
        };

        # `nix flake check`
        checks = {
          test = self.packages.${system}.test;
          ruff-check = self.packages.${system}.ruff-check;
          ruff-format = self.packages.${system}.ruff-format;
          pyrefly-check = self.packages.${system}.pyrefly-check;
        };

        # There are two different modes of development:
        # - Impurely using uv to manage virtual environments
        # - Pure development using uv2nix to manage virtual environments
        devShells = {
          # It is of course perfectly OK to keep using an impure virtualenv workflow and only use uv2nix to build packages.
          # This devShell simply adds Python and undoes the dependency leakage done by Nixpkgs Python infrastructure.
          impure = pkgs.mkShell {
            packages = [
              virtualenv
              pkgs.uv
              pkgs.clang-tools

              pkgs.clang
              pkgs.llvmPackages.mlir
              pkgs.gmp
            ];
            env = {
              # Prevent uv from managing Python downloads
              UV_PYTHON_DOWNLOADS = "never";
              # Force uv to use nixpkgs Python interpreter
              UV_PYTHON = python.interpreter;
            }
            // lib.optionalAttrs pkgs.stdenv.isLinux {
              # Python libraries often load native shared objects using dlopen(3).
              # Setting LD_LIBRARY_PATH makes the dynamic library loader aware of libraries without using RPATH for lookup.
              LD_LIBRARY_PATH = lib.makeLibraryPath pkgs.pythonManylinuxPackages.manylinux1;
            };
            shellHook = ''
              unset PYTHONPATH
            '';
          };

          # This devShell uses uv2nix to construct a virtual environment purely from Nix, using the same dependency specification as the application.
          # The notable difference is that we also apply another overlay here enabling editable mode ( https://setuptools.pypa.io/en/latest/userguide/development_mode.html ).
          #
          # This means that any changes done to your local files do not require a rebuild.
          #
          # Note: Editable package support is still unstable and subject to change.
          uv2nix =
            let
              # Create an overlay enabling editable mode for all local dependencies.
              editableOverlay = workspace.mkEditablePyprojectOverlay {
                # Use environment variable
                root = "$REPO_ROOT";
                # Optional: Only enable editable for these packages
                # members = [ "loop-lang" ];
              };

              # Override previous set with our overrideable overlay.
              editablePythonSet = pythonSet.overrideScope (
                lib.composeManyExtensions [
                  editableOverlay

                  # Apply fixups for building an editable package of your workspace packages
                  (final: prev: {
                    loop-lang = prev.loop-lang.overrideAttrs (old: {
                      # It's a good idea to filter the sources going into an editable build
                      # so the editable package doesn't have to be rebuilt on every change.
                      src = lib.fileset.toSource {
                        root = old.src;
                        fileset = lib.fileset.unions [
                          (old.src + "/pyproject.toml")
                          (old.src + "/README.md")

                          (old.src + "/src/l2/__init__.py")
                          (old.src + "/src/l2/__main__.py")
                          (old.src + "/src/l2/core.py")

                          (old.src + "/src/dialects/__init__.py")
                          (old.src + "/src/dialects/bigint.py")
                          (old.src + "/src/dialects/bignum.py")
                          (old.src + "/src/dialects/bignum_to_llvm.py")
                          (old.src + "/src/dialects/bignum_runtime.c")

                          (old.src + "/src/transforms/__init__.py")
                          (old.src + "/src/transforms/convert_scf_to_cf.py")

                          (old.src + "/src/ic3/__init__.py")
                        ];
                      };

                      # Hatchling (our build system) has a dependency on the `editables` package when building editables.
                      #
                      # In normal Python flows this dependency is dynamically handled, and doesn't need to be explicitly declared.
                      # This behaviour is documented in PEP-660.
                      #
                      # With Nix the dependency needs to be explicitly declared.
                      nativeBuildInputs =
                        old.nativeBuildInputs
                        ++ final.resolveBuildSystem {
                          editables = [ ];
                        };
                    });

                  })
                ]
              );

              # Build virtual environment, with local packages being editable.
              #
              # Enable all optional dependencies for development.
              virtualenv = editablePythonSet.mkVirtualEnv (projectAsNixPkg.pname + "-dev-env") workspace.deps.all;
            in
            pkgs.mkShell {
              packages = [
                virtualenv
                pkgs.uv
                pkgs.clang-tools

                pkgs.clang
                pkgs.llvmPackages.mlir
                pkgs.gmp
              ];

              env = {
                # Don't create venv using uv
                UV_NO_SYNC = "1";

                # Force uv to use nixpkgs Python interpreter
                UV_PYTHON = python.interpreter;

                # Prevent uv from downloading managed Python's
                UV_PYTHON_DOWNLOADS = "never";
              };

              shellHook = ''
                # Undo dependency propagation by nixpkgs.
                unset PYTHONPATH

                # Get repository root using git. This is expanded at runtime by the editable `.pth` machinery.
                export REPO_ROOT=$(git rev-parse --show-toplevel)
              '';

            };
        };
      }
    );
}
