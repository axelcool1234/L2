# src/dialects/noop.py

from xdsl.ir.core import Dialect
from xdsl.dialects.builtin import StringAttr
from xdsl.irdl.operations import IRDLOperation
from xdsl.irdl.operations import irdl_op_definition


@irdl_op_definition
class NoOperation(IRDLOperation):
    name = "noop.noop"

    def __init__(self, value: StringAttr):
        super().__init__(
            operands=[],
            result_types=[],
            attributes={"info": value},
        )


NoOp = Dialect(
    "noop",
    [NoOperation],
    [],
)
