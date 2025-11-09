# src/dialects/bignum.py

from typing import ClassVar
from xdsl.irdl.constraints import VarConstraint
from xdsl.irdl.attributes import base
from xdsl.ir.core import Attribute
from xdsl.irdl.operations import opt_operand_def
from xdsl.dialects.builtin import VectorType
from typing import Sequence
from xdsl.irdl.operations import var_operand_def
import abc

from xdsl.dialects.builtin import (
    IntegerAttr,
    i1,
)
from xdsl.ir import Dialect, Operation
from xdsl.ir.core import ParametrizedAttribute, SSAValue, TypeAttribute
from xdsl.irdl import irdl_op_definition
from xdsl.irdl.attributes import irdl_attr_definition
from xdsl.irdl.operations import (
    IRDLOperation,
    attr_def,
    operand_def,
    result_def,
    traits_def,
)
from xdsl.traits import Commutative, Pure, SameOperandsAndResultType


@irdl_attr_definition
class BigNumType(ParametrizedAttribute, TypeAttribute):
    """Type for unlimited precision numbers, with Gnu MultiPrecision Library semantics."""

    name = "bignum.bignum"


bignum = BigNumType()


@irdl_op_definition
class ConstantOp(IRDLOperation):
    name = "bignum.constant"

    value = attr_def(IntegerAttr)
    result = result_def(bignum)

    def __init__(self, value: IntegerAttr):
        super().__init__(
            operands=[],
            result_types=[bignum],
            attributes={"value": value},
        )


@irdl_op_definition
class FromElementsOp(IRDLOperation):
    """Create a vector of `bignums`."""

    name = "bignum.from_elements"
    elements = var_operand_def(bignum)
    result = result_def()

    assembly_format = "$elements attr-dict `:` type($result)"

    def __init__(self, elements: Sequence[SSAValue | Operation]):
        super().__init__(
            operands=[elements],
            result_types=[VectorType(BigNumType(), [len(elements)])],
        )


@irdl_op_definition
class ExtractOp(IRDLOperation):
    """Extract an element out of a `bignum` vector."""

    name = "bignum.extractelement"

    vector = operand_def(VectorType)
    position = opt_operand_def(bignum)
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
    """Insert an element into a `bignum` vector at the specified index."""

    name = "bignum.insert"

    source = operand_def(bignum)  # value to store
    dest = operand_def(VectorType)
    position = operand_def(bignum)
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


class BinaryOperation(IRDLOperation, abc.ABC):
    """Binary operation where all operands and results are `bignum`s."""

    lhs = operand_def(bignum)
    rhs = operand_def(bignum)
    result = result_def(bignum)

    assembly_format = "$lhs `,` $rhs attr-dict `:` type($result)"

    def __init__(
        self,
        operand1: Operation | SSAValue,
        operand2: Operation | SSAValue,
    ):
        super().__init__(operands=[operand1, operand2], result_types=[bignum])


@irdl_op_definition
class AddOp(BinaryOperation):
    """Add two `bignum`s."""

    name = "bignum.add"

    traits = traits_def(
        Pure(),
        Commutative(),
        SameOperandsAndResultType(),
    )


class ComparisonOperation(IRDLOperation, abc.ABC):
    """Binary operation comparing two `bignum`s and returning a boolean."""

    lhs = operand_def(bignum)
    rhs = operand_def(bignum)
    result = result_def(i1)

    assembly_format = "$lhs `,` $rhs attr-dict `:` type($result)"

    def __init__(
        self,
        operand1: Operation | SSAValue,
        operand2: Operation | SSAValue,
    ):
        super().__init__(operands=[operand1, operand2], result_types=[i1])


@irdl_op_definition
class EqOp(ComparisonOperation):
    """Check equality of two `bignum`s."""

    name = "bignum.eq"

    traits = traits_def(
        Pure(),
        Commutative(),
    )


@irdl_op_definition
class NeqOp(ComparisonOperation):
    """Check inequality of two `bignum`s."""

    name = "bignum.neq"

    traits = traits_def(
        Pure(),
        Commutative(),
    )


@irdl_op_definition
class GtOp(ComparisonOperation):
    """Check if one `bignum` is greater than another."""

    name = "bignum.gt"

    traits = traits_def(
        Pure(),
    )


@irdl_op_definition
class GteOp(ComparisonOperation):
    """Check if one `bignum` is greater than or equal to another."""

    name = "bignum.gte"

    traits = traits_def(
        Pure(),
    )


@irdl_op_definition
class LtOp(ComparisonOperation):
    """Check if one `bignum` is less than another."""

    name = "bignum.lt"

    traits = traits_def(
        Pure(),
    )


@irdl_op_definition
class LteOp(ComparisonOperation):
    """Check if one `bignum` is less than or equal to another."""

    name = "bignum.lte"

    traits = traits_def(
        Pure(),
    )


@irdl_op_definition
class PrintOp(IRDLOperation):
    """Print a `bignum`."""

    name = "bignum.print"

    _T: ClassVar = VarConstraint("T", base(BigNumType))
    _V: ClassVar = VarConstraint("V", VectorType.constr(_T))

    val = operand_def(VectorType.constr(_T) | _T)
    assembly_format = "$val attr-dict `:` type($val)"

    def __init__(self, value: Operation | SSAValue):
        super().__init__(operands=[value], result_types=[])


@irdl_op_definition
class PrintlnOp(IRDLOperation):
    """Println a `bignum`."""

    name = "bignum.println"

    _T: ClassVar = VarConstraint("T", base(BigNumType))
    _V: ClassVar = VarConstraint("V", VectorType.constr(_T))

    val = operand_def(VectorType.constr(_T) | _T)
    assembly_format = "$val attr-dict `:` type($val)"

    def __init__(self, value: Operation | SSAValue):
        super().__init__(operands=[value], result_types=[])


@irdl_op_definition
class FreeOp(IRDLOperation):
    """Free a `bignum`."""

    name = "bignum.free"

    val = operand_def(bignum)
    assembly_format = "$val attr-dict"

    def __init__(self, value: Operation | SSAValue):
        super().__init__(operands=[value], result_types=[])


BigNum = Dialect(
    "bignum",
    [
        AddOp,
        EqOp,
        NeqOp,
        GtOp,
        GteOp,
        LtOp,
        LteOp,
        ConstantOp,
        PrintOp,
        PrintlnOp,
        FreeOp,
        InsertOp,
        ExtractOp,
        FromElementsOp,
    ],
    [
        BigNumType,
    ],
)
