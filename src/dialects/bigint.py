# src/dialects/bigint.py

# ruff: noqa: F405
from xdsl.irdl.attributes import base
from typing import ClassVar
from xdsl.irdl.constraints import VarConstraint
from xdsl.irdl.operations import opt_operand_def
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
    """Create a vector of `bigint`s."""

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
    """Extract an element out of a `bigint` vector."""

    name = "bigint.extractelement"

    vector = operand_def(VectorType)
    position = opt_operand_def(bigint)
    result = result_def(Attribute)
    traits = traits_def(Pure())

    assembly_format = (
        "$vector `[` $position `]` attr-dict `:` type($result) `from` type($vector)"
    )

    def __init__(
        self,
        vector: SSAValue | Operation,
        position: SSAValue | Operation | None = None,
    ):
        vector = SSAValue.get(vector, type=VectorType)
        assert isinstance(vector.type, VectorType)

        super().__init__(
            operands=[vector, position],
            result_types=[vector.type.element_type],
        )


@irdl_op_definition
class InsertOp(IRDLOperation):
    """Insert an element into a `bigint` vector at the specified index."""

    name = "bigint.insert"

    source = operand_def(bigint)  # value to store
    dest = operand_def(VectorType)
    position = operand_def(bigint)
    result = result_def()
    traits = traits_def(Pure())

    assembly_format = "$source `,` $dest `[` $position `]` attr-dict `:` type($source) `into` type($dest) `as` type($result)"

    def __init__(
        self,
        source: SSAValue,
        dest: SSAValue,
        position: SSAValue,
    ):
        super().__init__(
            operands=[source, dest, position],
            result_types=[dest.type],
        )


@irdl_op_definition
class PrintOp(IRDLOperation):
    """Print a `bigint` or a vector of `bigint`s."""

    name = "bigint.print"

    _T: ClassVar = VarConstraint("T", base(BigIntegerType))
    _V: ClassVar = VarConstraint("V", VectorType.constr(_T))

    val = operand_def(VectorType.constr(_T) | _T)
    assembly_format = "$val attr-dict `:` type($val)"

    def __init__(self, value: Operation | SSAValue):
        super().__init__(operands=[value], result_types=[])


@irdl_op_definition
class PrintlnOp(IRDLOperation):
    """Println a `bigint` or a vector of `bigint`s."""

    name = "bigint.println"

    _T: ClassVar = VarConstraint("T", base(BigIntegerType))
    _V: ClassVar = VarConstraint("V", VectorType.constr(_T))

    val = operand_def(VectorType.constr(_T) | _T)
    assembly_format = "$val attr-dict `:` type($val)"

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

    @impl(InsertOp)
    def run_insert(self, interpreter: Interpreter, op: InsertOp, args: PythonValues):
        vec = args[1]
        vec[args[2]] = args[0]
        return (vec,)

    @impl(ExtractOp)
    def run_extract(self, interpreter: Interpreter, op: ExtractOp, args: PythonValues):
        return (args[0][args[1]],)

    @impl(FromElementsOp)
    def run_fromelements(
        self, interpreter: Interpreter, op: FromElementsOp, args: PythonValues
    ):
        return (list(args),)

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
