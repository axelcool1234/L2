# ruff: noqa: F405
from xdsl.ir.core import Attribute
from xdsl.dialects.builtin import VectorType
from typing import Sequence
from xdsl.irdl.operations import var_operand_def
from typing import cast

from xdsl.dialects.bigint import *  # noqa: F403
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

    assembly_format = "$value attr-dict `:` type($result)"

    def __init__(self, value: IntegerAttr):
        super().__init__(
            operands=[],
            result_types=[bigint],
            attributes={"value": value},
        )


@irdl_op_definition
class FromElementsOp(IRDLOperation):
    """Create a vector of bigints."""

    name = "bigint.from_elements"
    elements = var_operand_def(bigint)
    result = result_def()

    assembly_format = "$elements attr-dict `:` type($result)"

    def __init__(self, elements: Sequence[SSAValue | Operation]):
        super().__init__(
            operands=[elements],
            result_types=[VectorType(BigIntegerType(), [len(elements)])],
        )


@irdl_op_definition
class ExtractOp(IRDLOperation):
    """Extract an element out of a bigint vector."""

    name = "bigint.extract"

    vector = operand_def(VectorType)

    result = result_def(bigint)

    assembly_format = "$vector attr-dict `:` type($result) `from` type($vector)"

    def __init__(
        self,
        vector: SSAValue,
        positions: Sequence[SSAValue | Operation],
        result_type: Attribute,
    ):
        super().__init__(
            operands=[vector],
            result_types=[result_type],
        )


# @irdl_op_definition
# class InsertOp(IRDLOperation):
#     """Insert an element into a bigint vector at the specified index."""

#     name = "bigint.insert"

#     source = operand_def(VectorType)
#     dest = operand_def(bigint)
#     result = result_def(VectorType)

#     traits = traits_def(Pure())

#     assembly_format = (
#         "$source `,` $dest custom<DynamicIndexList>($dynamic_position, $static_position)"
#         "attr-dict `:` type($source) `into` type($dest)"
#     )

#     custom_directives = (DynamicIndexList,)

#     def __init__(
#         self,
#         source: SSAValue,
#         dest: SSAValue,
#         positions: Sequence[SSAValue | int],
#         result_type: Attribute | None = None,
#     ):
#         static_positions, dynamic_positions = split_dynamic_index_list(
#             positions, InsertOp.DYNAMIC_INDEX
#         )

#         if result_type is None:
#             result_type = dest.type

#         super().__init__(
#             operands=[source, dest, dynamic_positions],
#             result_types=[result_type],
#             properties={
#                 "static_position": DenseArrayBase.from_list(i64, static_positions)
#             },
#         )


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
