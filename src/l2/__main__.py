# src/l2/__main__.py

"""
LoopLang CLI entrypoint.
"""

import argparse
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Type, TypeVar

from lark import Lark
from xdsl import interpreters
from xdsl.context import Context
from xdsl.dialects import arith, builtin, cf, func, llvm, printf, scf
from xdsl.dialects.builtin import ModuleOp
from xdsl.interpreter import Interpreter, PythonValues, impl, register_impls
from xdsl.interpreters.arith import ArithFunctions, _int_bitwidth
from xdsl.transforms.mlir_opt import MLIROptPass
from xdsl.utils.hints import isa

from dialects import LowerBigNumToLLVM, bigint
from l2 import (
    IRGenCompiler,
    IRGenInterpreter,
    grammar,
    precedence,
)

IRGen = TypeVar("IRGen", IRGenCompiler, IRGenInterpreter)


@register_impls
class ArithFunctions(ArithFunctions):
    """
    Extends xdsl.interpreters.arith.ArithFunctions.
    """

    @impl(arith.ExtUIOp)
    def run_extui(
        self, interpreter: Interpreter, op: arith.ExtUIOp, args: PythonValues
    ):
        assert len(args) == 1
        assert isa(op.input.type, builtin.IntegerType)
        assert isa(op.result.type, builtin.IntegerType)

        value = args[0]
        input_width = _int_bitwidth(interpreter, op.input.type)
        mask = (1 << input_width) - 1
        return (value & mask,)


def context() -> Context:
    ctx = Context()
    ctx.load_dialect(bigint.BigInt)
    ctx.load_dialect(arith.Arith)
    ctx.load_dialect(builtin.Builtin)
    ctx.load_dialect(cf.Cf)
    ctx.load_dialect(func.Func)
    ctx.load_dialect(printf.Printf)
    ctx.load_dialect(scf.Scf)
    ctx.load_dialect(llvm.LLVM)
    return ctx


def register_implementations(interpreter: Interpreter, ctx: Context):
    interpreter.register_implementations(bigint.BigIntFunctions())
    interpreter.register_implementations(ArithFunctions())
    interpreter.register_implementations(interpreters.builtin.BuiltinFunctions())
    interpreter.register_implementations(interpreters.cf.CfFunctions())
    interpreter.register_implementations(interpreters.func.FuncFunctions())
    interpreter.register_implementations(interpreters.printf.PrintfFunctions())
    interpreter.register_implementations(interpreters.scf.ScfFunctions())


def emit_check(
    emit: str | None, step: str, code: ModuleOp | bytes, output: Path | None
) -> bool:
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


def source_to_mlir(input: Path, debug: bool, irgen: Type[IRGen]) -> ModuleOp:
    l2_code = input.read_text()

    # Source -> AST
    parser = Lark(grammar, start="program", parser="lalr")
    tree = parser.parse(l2_code)

    # AST -> MLIR
    generator = irgen(debug)
    module: ModuleOp = generator.visit(tree)
    try:
        module.verify()
    except Exception:
        print("module verification error")
        raise

    return module


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
    # Source -> AST -> MLIR
    module = source_to_mlir(input, debug, IRGenCompiler)

    # Emit MLIR and exit if specified
    if emit_check(emit, "mlir", module, output):
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
    if emit_check(emit, "llvm", llvm_bytes, output):
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


def interpret_loop_lang(
    input: Path,
    output: Path | None,
    emit: str | None = None,
    run: bool = False,
    debug: bool = False,
) -> None:
    # Source -> AST -> MLIR
    module = source_to_mlir(input, debug, IRGenInterpreter)

    # Emit MLIR and exit if specified
    if emit_check(emit, "mlir", module, output):
        return

    # Load and register dialects
    interpreter = Interpreter(module)
    ctx = context()
    register_implementations(interpreter, ctx)

    # MLIR -> MLIR (lowerings to make it interpretable)
    MLIROptPass(
        arguments=("--convert-scf-to-cf", "--allow-unregistered-dialect")
    ).apply(ctx, module)

    # MLIR -> Interpretation
    assert module.body.blocks[0].first_op is not None
    interpreter.call_op(module.body.blocks[0].first_op)


def main() -> None:
    parser = argparse.ArgumentParser(description="LoopLang (L2) Compiler")
    parser.add_argument("input_file", type=Path, nargs="?", help="LoopLang source file")
    parser.add_argument(
        "-c",
        "--compile",
        action="store_true",
        help="Compile to a binary instead of interpreting",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="If compiling, this specifies the output executable name. Otherwise, this does nothing",
    )
    parser.add_argument(
        "-r",
        "--run",
        action="store_true",
        help="If compiling, run the program immediately after compilation and discard the resulting binary. Else, this flag does nothing",
    )

    parser.add_argument(
        "--emit",
        choices=["mlir", "llvm"],
        default=None,
        help="Emit intermediate representation and stop",
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
        return

    if args.input_file is None:
        parser.error(
            "the following arguments are required: input_file (unless -g/--grammar is used)"
        )
    else:
        if args.compile:
            compile_loop_lang(
                args.input_file, args.output, args.emit, args.run, args.debug
            )
        else:
            interpret_loop_lang(
                args.input_file, args.output, args.emit, args.run, args.debug
            )


if __name__ == "__main__":
    main()
