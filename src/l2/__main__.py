# src/l2/__main__.py

"""
LoopLang CLI entrypoint.
"""

from pathlib import Path
from lark import Lark
from xdsl.printer import Printer

from l2 import L2Interpreter, grammar

import subprocess
import argparse


def run_cmd(cmd, input_bytes=None):
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


def compile_loop_lang(input: Path, output: Path | None = Path("a.out")):
    l2_code = input.read_text()

    # Parse
    parser = Lark(grammar, start="program", parser="lalr")
    tree = parser.parse(l2_code)

    # AST -> MLIR
    interpreter = L2Interpreter()
    interpreter.visit(tree)
    try:
        interpreter.module.verify()
    except Exception:
        print("module verification error")
        raise

    # MLIR -> LLVM
    llvm_str = run_cmd(
        ["xdsl-opt", "--frontend", "mlir", "-p", "printf-to-llvm"],
        input_bytes=str(interpreter.module).encode(),
    )
    llvm_str = run_cmd(
        [
            "mlir-opt",
            "--convert-scf-to-cf",
            "--convert-cf-to-llvm",
            "--convert-arith-to-llvm",
            "--convert-func-to-llvm",
        ],
        input_bytes=llvm_str,
    )
    llvm_str = run_cmd(
        ["mlir-translate", "--mlir-to-llvmir"],
        input_bytes=llvm_str,
    )

    # LLVM -> bin
    run_cmd(["clang", "-x", "ir", "-", "-o", str(output)], input_bytes=llvm_str)


def main() -> None:
    parser = argparse.ArgumentParser(description="LoopLang (L2) Compiler")
    parser.add_argument("input_file", type=Path, help="LoopLang source file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("a.out"),
        help="Output executable name",
    )
    args = parser.parse_args()

    compile_loop_lang(args.input_file, args.output)


if __name__ == "__main__":
    main()
