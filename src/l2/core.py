from lark import Transformer
from xdsl.builder import Builder
from xdsl.dialects.arith import AddiOp, ConstantOp
from xdsl.dialects.builtin import (
    IntegerAttr,
    ModuleOp,
    i1,
    i32,
)
from xdsl.dialects.comb import AndOp, OrOp
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.ir import Block, Region
from xdsl.ir.core import Attribute, SSAValue
from xdsl.rewriter import InsertPoint
from xdsl.utils.scoped_dict import ScopedDict

grammar = r"""
?program: stmt+

?stmt: lvar "=" expr              -> assign_stmt
     | "while" expr "{" stmt* "}" -> loop_stmt

?expr: expr "+" expr   -> add_expr
     | expr "&&" expr  -> and_expr
     | expr "||" expr  -> or_expr
     | rvar
     | INT
     | BOOL
BOOL: "@T" | "@F"

lvar: VAR
rvar: VAR

%import common.INT
%import common.CNAME -> VAR
%import common.WS
%ignore WS
"""


class L2Transformer(Transformer):
    """
    Implementation of a simple MLIR emission from the L2 AST.
    """

    module: ModuleOp
    """
    A "module" matches an L2 source file: containing a list with only one function.
    """

    module_builder: Builder
    func_builder: Builder
    entry_block: Block
    func_region: Region
    func: FuncOp
    """
    The builder is a helper class to create IR inside a function. The builder
    is stateful, in particular it keeps an "insertion point": this is where
    the next operations will be introduced.
    """

    symbol_table: ScopedDict[str, SSAValue] | None = None
    """
    The symbol table maps a variable name to a value in the current scope.
    Entering a function creates a new scope, and the function arguments are
    added to the mapping. When the processing of a function is terminated, the
    scope is destroyed and the mappings created in this scope are dropped.
    """

    def __init__(self):
        # Create empty module. Function containing all of the user's transformed L2 instructions will be placed in here at the end (in `program`).
        self.module = ModuleOp([])
        self.module_builder = Builder(InsertPoint.at_end(self.module.body.blocks[0]))

        # Create one singular function where every transformed L2 instruction will be inserted into.
        self.entry_block = Block()
        self.func_region = Region(self.entry_block)
        self.func = FuncOp.from_region("main", [], [], self.func_region)
        self.func_builder = Builder(InsertPoint.at_end(self.entry_block))

        # Create an empty symbol table for variable name to SSA value mappings
        self.symbol_table = ScopedDict()

    def program(self, node) -> FuncOp:
        self.func_builder.insert(ReturnOp())
        return self.module_builder.insert(self.func)

    def assign_stmt(self, node):
        assert self.symbol_table is not None
        var_name = str(node[0].children[0])
        expr_op = node[1]
        self.symbol_table[var_name] = expr_op.result
        return expr_op

    def loop_stmt(self, node):
        pass

    def add_expr(self, node) -> AddiOp:
        return self.func_builder.insert(AddiOp(node[0], node[1]))

    def or_expr(self, node) -> OrOp:
        return self.func_builder.insert(OrOp([node[0], node[1]], i1))

    def and_expr(self, node) -> AndOp:
        return self.func_builder.insert(AndOp([node[0], node[1]], i1))

    def INT(self, node):
        return self.func_builder.insert(ConstantOp(IntegerAttr(int(node.value), i32)))

    def BOOL(self, node) -> ConstantOp:
        if node[1] == "T":
            return self.func_builder.insert(ConstantOp(IntegerAttr(1, i1)))
        else:  # node[1] == "F":
            return self.func_builder.insert(ConstantOp(IntegerAttr(0, i1)))

    def rvar(self, node) -> SSAValue[Attribute]:
        assert self.symbol_table is not None
        var_name = str(node[0])
        try:
            return self.symbol_table[var_name]
        except Exception as e:
            raise Exception(f"error: Unknown variable `{var_name}`") from e
