# src/l2/__main__.py

"""
LoopLang CLI entrypoint.
"""

from xdsl.dialects.builtin import ModuleOp
from pathlib import Path
from lark import Lark
import tempfile
import os

from l2 import IRGen, grammar

import subprocess
import argparse


def run_cmd(cmd, input_bytes=None) -> bytes:
    try:
        result = subprocess.run(
            cmd,
            input=input_bytes,
            capture_output=True,
            check=True,
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {' '.join(cmd)}")
        if e.stdout:
            print("--- stdout ---")
            print(e.stdout.decode())
        if e.stderr:
            print("--- stderr ---")
            print(e.stderr.decode())
        raise


def compile_loop_lang(
    input: Path,
    output: Path | None,
    emit: str | None = None,
    run: bool = False,
) -> None:
    def emit_check(step: str, code: ModuleOp | bytes) -> bool:
        if emit == step:
            if isinstance(code, bytes):
                text = code.decode()
            else:  # isinstance(code, ModuleOp)
                text = str(code)
            if output is None:
                print(text)
            else:
                output.write_text(text)
            return True
        return False

    l2_code = input.read_text()

    # Source -> AST
    parser = Lark(grammar, start="program", parser="lalr")
    tree = parser.parse(l2_code)

    # AST -> MLIR
    generator = IRGen()
    generator.visit(tree)
    try:
        generator.module.verify()
    except Exception:
        print("module verification error")
        raise

    # Emit MLIR and exit if specified
    if emit_check("mlir", generator.module):
        return

    # MLIR -> LLVM
    llvm_bytes = run_cmd(
        ["xdsl-opt", "--frontend", "mlir", "-p", "printf-to-llvm"],
        input_bytes=str(generator.module).encode(),
    )
    llvm_bytes = run_cmd(
        [
            "mlir-opt",
            "--convert-scf-to-cf",
            "--convert-cf-to-llvm",
            "--convert-arith-to-llvm",
            "--convert-func-to-llvm",
        ],
        input_bytes=llvm_bytes,
    )
    llvm_bytes = run_cmd(
        ["mlir-translate", "--mlir-to-llvmir"],
        input_bytes=llvm_bytes,
    )

    # Emit LLVM and exit if specified
    if emit_check("llvm", llvm_bytes):
        return

    # LLVM -> bin
    if run:
        # Run program immediately after compilation
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            run_cmd(["clang", "-x", "ir", "-", "-o", tmp.name], input_bytes=llvm_bytes)
        run_cmd([tmp.name])
        os.remove(tmp.name)
    else:
        if output is None:
            output = Path("a.out")
        run_cmd(["clang", "-x", "ir", "-", "-o", str(output)], input_bytes=llvm_bytes)


def main() -> None:
    parser = argparse.ArgumentParser(description="LoopLang (L2) Compiler")
    parser.add_argument("input_file", type=Path, help="LoopLang source file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output executable name",
    )
    parser.add_argument(
        "--emit",
        choices=["mlir", "llvm"],
        default=None,
        help="Emit intermediate representation and stop",
    )
    parser.add_argument(
        "-r",
        "--run",
        action="store_true",
        help="Run the program immediately after compilation and discard the resulting binary",
    )
    args = parser.parse_args()

    compile_loop_lang(args.input_file, args.output, args.emit, args.run)


if __name__ == "__main__":
    main()
