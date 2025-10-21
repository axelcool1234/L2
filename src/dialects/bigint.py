from typing import cast

from xdsl.dialects import bigint
from xdsl.dialects.bigint import *
from xdsl.dialects.builtin import IntegerAttr
from xdsl.interpreter import (
    Interpreter,
    InterpreterFunctions,
    PythonValues,
    impl,
    register_impls,
)
from xdsl.ir.core import Dialect, Operation, SSAValue
from xdsl.irdl.operations import (
    IRDLOperation,
    attr_def,
    irdl_op_definition,
    operand_def,
    result_def,
)


@irdl_op_definition
class ConstantOp(IRDLOperation):
    name = "bigint.constant"

    value = attr_def(IntegerAttr)
    result = result_def(bigint)

    def __init__(self, value: IntegerAttr):
        super().__init__(
            operands=[],
            result_types=[bigint],
            attributes={"value": value},
        )


@irdl_op_definition
class PrintOp(IRDLOperation):
    """Print a `bigint`."""

    name = "bigint.print"

    val = operand_def(bigint)
    assembly_format = "$val attr-dict"

    def __init__(self, value: Operation | SSAValue):
        super().__init__(operands=[value], result_types=[])


@irdl_op_definition
class PrintlnOp(IRDLOperation):
    """Println a `bigint`."""

    name = "bigint.println"

    val = operand_def(bigint)
    assembly_format = "$val attr-dict"

    def __init__(self, value: Operation | SSAValue):
        super().__init__(operands=[value], result_types=[])


BigInt = Dialect(
    BigInt.name,
    list(BigInt.operations) + [ConstantOp, PrintOp, PrintlnOp],
    list(BigInt.attributes),
    BigInt._interfaces,
)


@register_impls
class BigIntFunctions(InterpreterFunctions):
    @impl(ConstantOp)
    def run_constant(
        self, interpreter: Interpreter, op: ConstantOp, args: PythonValues
    ) -> PythonValues:
        interpreter.interpreter_assert(
            isinstance(op.value, IntegerAttr),
            f"bigint.constant not implemented for {type(op.value)}",
        )
        value = cast(IntegerAttr, op.value)
        return (value.value.data,)

    @impl(AddOp)
    def run_add(self, interpreter: Interpreter, op: AddOp, args: PythonValues):
        return (args[0] + args[1],)

    @impl(EqOp)
    def run_eq(self, interpreter: Interpreter, op: EqOp, args: PythonValues):
        return (args[0] == args[1],)

    @impl(NeqOp)
    def run_neq(self, interpreter: Interpreter, op: NeqOp, args: PythonValues):
        return (args[0] != args[1],)

    @impl(GtOp)
    def run_gt(self, interpreter: Interpreter, op: GtOp, args: PythonValues):
        return (args[0] > args[1],)

    @impl(GteOp)
    def run_gte(self, interpreter: Interpreter, op: GteOp, args: PythonValues):
        return (args[0] >= args[1],)

    @impl(LtOp)
    def run_lt(self, interpreter: Interpreter, op: LtOp, args: PythonValues):
        return (args[0] < args[1],)

    @impl(LteOp)
    def run_lte(self, interpreter: Interpreter, op: LteOp, args: PythonValues):
        return (args[0] <= args[1],)

    @impl(PrintOp)
    def run_print(self, interpreter: Interpreter, op: PrintOp, args: PythonValues):
        print(args[0], file=interpreter.file, end="")
        return ()

    @impl(PrintlnOp)
    def run_println(self, interpreter: Interpreter, op: PrintlnOp, args: PythonValues):
        print(args[0], file=interpreter.file)
        return ()
