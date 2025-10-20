# src/l2/__main__.py

"""
LoopLang CLI entrypoint.
"""

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

from lark import Lark
from xdsl.context import Context
from xdsl.dialects import arith, func, printf, scf
from xdsl.dialects.builtin import Builtin, ModuleOp

from dialects import LowerBigNumToLLVM
from l2 import IRGen, grammar, insert_bignum_decls, precedence


def context() -> Context:
    ctx = Context()
    ctx.load_dialect(arith.Arith)
    ctx.load_dialect(Builtin)
    ctx.load_dialect(func.Func)
    ctx.load_dialect(printf.Printf)
    ctx.load_dialect(scf.Scf)
    return ctx


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
    debug: bool = False,
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
    generator = IRGen(debug)
    generator.visit(tree)
    module = generator.module
    insert_bignum_decls(module)
    try:
        module.verify()
    except Exception:
        print("module verification error")
        raise

    # Emit MLIR and exit if specified
    if emit_check("mlir", module):
        return

    # Lower custom dialects first
    ctx = context()
    LowerBigNumToLLVM().apply(ctx, module)

    # MLIR -> LLVM
    llvm_bytes = run_cmd(
        ["xdsl-opt", "--frontend", "mlir", "-p", "printf-to-llvm"],
        input_bytes=str(module).encode(),
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
    # Write LLVM IR to temporary file
    with tempfile.NamedTemporaryFile(suffix=".ll", delete=False) as tmp_ir:
        tmp_ir.write(llvm_bytes)
        tmp_ir_path = Path(tmp_ir.name)

    # Compile runtime.c to object file
    runtime_obj = tmp_ir_path.with_suffix(".o")

    cflags = os.getenv("CFLAGS", "").split()
    ldflags = os.getenv("LDFLAGS", "").split()
    run_cmd(
        [
            "clang",
            "-c",
            "src/dialects/bignum_runtime.c",
            "-o",
            str(runtime_obj),
            *cflags,
        ]
    )

    # Link LLVM IR + runtime object into final executable
    if output is None:
        if run:
            output_path = tmp_ir_path.with_suffix("")
        else:
            output_path = Path("a.out")
    else:
        output_path = output

    run_cmd(
        [
            "clang",
            str(tmp_ir_path),
            str(runtime_obj),
            "-lgmp",
            "-o",
            str(output_path),
            *ldflags,
        ]
    )

    # Cleanup temp files
    tmp_ir_path.unlink()
    runtime_obj.unlink()

    # Optionally run the binary
    if run:
        try:
            subprocess.run([str(output_path)])
        except subprocess.CalledProcessError as e:
            print("Run failed")
            if e.stdout:
                print("--- stdout ---")
                print(e.stdout.decode())
            if e.stderr:
                print("--- stderr ---")
                print(e.stderr.decode())
            raise
        finally:
            output_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="LoopLang (L2) Compiler")
    parser.add_argument("input_file", type=Path, nargs="?", help="LoopLang source file")
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
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Parser outputs information on each node visit",
    )
    parser.add_argument(
        "-g",
        "--grammar",
        action="store_true",
        help="Display grammar and precedence rules for L2",
    )
    args = parser.parse_args()

    if args.grammar:
        print(f"--- L2 Precedence Rules ---\n{precedence}")
        print(f"--- L2 Grammar ---\n{grammar}")
    else:
        if args.input_file is None:
            parser.error(
                "the following arguments are required: input_file (unless -g/--grammar is used)"
            )
        else:
            compile_loop_lang(
                args.input_file, args.output, args.emit, args.run, args.debug
            )


if __name__ == "__main__":
    main()
