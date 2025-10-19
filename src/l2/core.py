from xdsl.ir.core import BlockArgument
from xdsl.dialects.scf import YieldOp
from xdsl.dialects.scf import ConditionOp
from lark.tree import Tree
from typing import Union
from xdsl.ir.core import OpResult
from typing import List
from lark.lexer import Token
from lark.visitors import (
    Interpreter,
    visit_children_decor,
)
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
from xdsl.dialects.scf import WhileOp
from xdsl.ir import Block, Region
from xdsl.ir.core import Attribute, SSAValue
from xdsl.rewriter import InsertPoint
from xdsl.utils.scoped_dict import ScopedDict

Op = Union[AddiOp, OrOp, AndOp, ConstantOp]
Use = Union[Op, OpResult, BlockArgument]
Node = List[Union[Token, Use]]

grammar = r"""
?program: stmt+

?stmt: lvar "=" expr              -> assign_stmt
     | "while" expr "{" stmt* "}" -> loop_stmt

?expr: expr "+" expr   -> add_expr
     | expr "&&" expr  -> and_expr
     | expr "||" expr  -> or_expr
     | rvar
     | INT             -> int_expr
     | BOOL            -> bool_expr
BOOL: "@T" | "@F"

lvar: VAR
rvar: VAR

%import common.INT
%import common.CNAME -> VAR
%import common.WS
%ignore WS
"""


class L2Interpreter(Interpreter):
    """
    Implementation of a simple MLIR emission from the L2 AST.
    """

    module: ModuleOp
    """
    A "module" matches an L2 source file: containing a list with only one function.
    """

    module_builder: Builder
    func_builder: Builder
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
        entry_block = Block()
        func_region = Region(entry_block)
        self.func = FuncOp.from_region("main", [], [], func_region)
        self.func_builder = Builder(InsertPoint.at_end(entry_block))

        # Create an empty symbol table for variable name to SSA value mappings
        self.symbol_table = ScopedDict()

        # DEBUG MODE
        self.debug = True

    def _dbg(self, name: str, node):
        if self.debug:
            print(f"\n[DEBUG] Visiting {name}:")
            try:
                print(f"  node = {node}")
                print(f"  len(node) = {len(node)}")
            except Exception:
                print("  node missing")
            finally:
                print("  -----------------------")

            try:
                print(f"  node[0] = {node[0]}")
            except Exception:
                print("  node[0] missing")
            finally:
                print("  -----------------------")

            try:
                print(f"  node[1] = {node[1]}")
            except Exception:
                print("  node[1] missing")
            finally:
                print("  -----------------------")

    @visit_children_decor  # pyrefly: ignore
    def program(self, node: Node) -> FuncOp:
        self._dbg("program", node)
        self.func_builder.insert(ReturnOp())
        return self.module_builder.insert(self.func)

    def loop_stmt(self, node: Tree):
        """
        node[0] = condition expression (should evaluate to i1)
        node[1:] = body statements
        """
        self._dbg("loop_stmt", node)
        assert self.symbol_table is not None

        loop_vars = []
        for stmt in node.children[1:]:
            if (
                stmt.data == "assign_stmt"
            ):  # Tree('assign_stmt', [Tree(Token('RULE', 'lvar'), [Token('VAR', 'y')]), ... ])
                loop_vars.append(stmt.children[0].children[0])

        # --- Prepare initial SSAValues ---
        # ERROR: What about variables that are initialized per iteration?
        # Fixing this would involve having the condition for the loop be done in the outer scope rather than the before region
        initial_ssas = [self.symbol_table[var] for var in loop_vars]
        ssa_types = [ssa_val.type for ssa_val in initial_ssas]

        # Create before block, before block args, and before region
        before_block = Block()
        before_args = [
            before_block.insert_arg(arg_type=ssa_type, index=index)
            for index, ssa_type in enumerate(ssa_types)
        ]
        before_region = Region(before_block)

        # Create after block, after block args, and after region
        after_block = Block()
        after_args = [
            after_block.insert_arg(arg_type=ssa_type, index=index)
            for index, ssa_type in enumerate(ssa_types)
        ]
        after_region = Region(after_block)

        # Build the WhileOp node
        loop_op = WhileOp(
            arguments=initial_ssas,
            result_types=ssa_types,
            before_region=before_region,
            after_region=after_region,
        )

        # Save current builder/scope
        old_builder = self.func_builder
        old_symbols = self.symbol_table

        # --- BEFORE REGION ---
        self.func_builder = Builder(InsertPoint.at_end(before_block))
        self.symbol_table = ScopedDict(old_symbols)

        # Evaluate the condition inside the 'before' region (scf.while runs this first)
        # NOTE: We do not use do-while loops, so nothing needs to be done in the before region beyond the condition.
        cond_value_inner = self.visit(node.children[0])
        assert isinstance(cond_value_inner, Use), "Loop condition must yield a value"
        self.func_builder.insert(ConditionOp(cond_value_inner, *before_args))

        # --- AFTER REGION ---
        self.func_builder = Builder(InsertPoint.at_end(after_block))
        self.symbol_table = ScopedDict(old_symbols)

        # Map loop-carried variable names to the after block args
        for name, ssa in zip(loop_vars, after_args):
            self.symbol_table[name] = ssa

        # Visit body statements
        for stmt in node.children[1:]:
            self.visit(stmt)

        # Terminate the loop body region with scf.yield (no yielded values)
        self.func_builder.insert(
            YieldOp(*[self.symbol_table[var] for var in loop_vars])
        )

        # Restore builder/scope
        self.func_builder = old_builder
        self.symbol_table = old_symbols

        # Insert loop op in parent region
        return self.func_builder.insert(loop_op)

    @visit_children_decor  # pyrefly: ignore
    def assign_stmt(self, node: Node) -> Use:
        """
        node[0] = [Token('VAR', var_name)]
        node[1] = AddiOp | OrOp | AndOp | ConstantOp | OpResult
        """
        self._dbg("assign_stmt", node)
        assert self.symbol_table is not None
        assert len(node) == 2
        assert isinstance(node[0], Token)
        assert isinstance(node[1], Use)

        var_name = str(node[0])
        if isinstance(node[1], OpResult):
            self.symbol_table[var_name] = node[1]
        else:
            assert isinstance(node[1], Op)
            self.symbol_table[var_name] = node[1].result
        return node[1]

    @visit_children_decor  # pyrefly: ignore
    def add_expr(self, node: List[Use]) -> AddiOp:
        self._dbg("add_expr", node)
        assert len(node) == 2
        assert isinstance(node[0], Use)
        assert isinstance(node[1], Use)

        return self.func_builder.insert(AddiOp(node[0], node[1]))

    @visit_children_decor  # pyrefly: ignore
    def or_expr(self, node: List[Use]) -> OrOp:
        self._dbg("or_expr", node)
        assert len(node) == 2
        assert isinstance(node[0], Use)
        assert isinstance(node[1], Use)

        return self.func_builder.insert(OrOp([node[0], node[1]], i1))

    @visit_children_decor  # pyrefly: ignore
    def and_expr(self, node: List[Use]) -> AndOp:
        self._dbg("and_expr", node)
        assert len(node) == 2
        assert isinstance(node[0], Use)
        assert isinstance(node[1], Use)

        return self.func_builder.insert(AndOp([node[0], node[1]], i1))

    @visit_children_decor  # pyrefly: ignore
    def int_expr(self, node: List[Token]) -> ConstantOp:
        """
        node[0] = [Token('INT', '[0-9]')]
        """
        self._dbg("int_expr", node)
        assert len(node) == 1
        assert isinstance(node[0], Token)

        return self.func_builder.insert(ConstantOp(IntegerAttr(int(node[0]), i32)))

    @visit_children_decor  # pyrefly: ignore
    def bool_expr(self, node: List[Token]) -> ConstantOp:
        """
        node[0] = [Token('BOOL', '@T' | '@F')]
        """
        self._dbg("bool_expr", node)
        assert len(node) == 1
        assert isinstance(node[0], Token)

        if node[0] == "@T":
            return self.func_builder.insert(ConstantOp(IntegerAttr(1, i1)))
        else:  # node[1] == "@F":
            return self.func_builder.insert(ConstantOp(IntegerAttr(0, i1)))

    @visit_children_decor  # pyrefly: ignore
    def rvar(self, node: List[Token]) -> SSAValue[Attribute]:
        """
        node[0] = [Token('VAR', rvar)]
        """
        self._dbg("rvar", node)
        assert len(node) == 1
        assert self.symbol_table is not None
        assert isinstance(node[0], Token)

        var_name = str(node[0])
        try:
            return self.symbol_table[var_name]
        except Exception as e:
            raise Exception(f"error: Unknown variable `{var_name}`") from e

    @visit_children_decor  # pyrefly: ignore
    def lvar(self, node: List[Token]) -> Token:
        """
        node[0] = [Token('VAR', lvar)]
        """
        self._dbg("lvar", node)
        assert len(node) == 1
        assert isinstance(node[0], Token)

        return node[0]  # unwrap
