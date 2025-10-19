from typing import List, Union

from lark.lexer import Token
from lark.tree import Tree
from lark.visitors import (
    Interpreter,
    visit_children_decor,
)
from xdsl.builder import Builder
from xdsl.dialects.arith import AddiOp, CmpiOp, ConstantOp
from xdsl.dialects.builtin import (
    IntegerAttr,
    ModuleOp,
    i1,
    i32,
)
from xdsl.dialects.comb import AndOp, OrOp, XorOp
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.dialects.printf import PrintFormatOp
from xdsl.dialects.scf import ConditionOp, WhileOp, YieldOp
from xdsl.ir import Block, Region
from xdsl.ir.core import Attribute, BlockArgument, OpResult, SSAValue
from xdsl.rewriter import InsertPoint
from xdsl.utils.scoped_dict import ScopedDict

Op = Union[
    AndOp,
    OrOp,
    XorOp,
    ConstantOp,
    AddiOp,
    CmpiOp,
]
Use = Union[Op, OpResult, BlockArgument]
Node = List[Union[Token, Use]]

# Precedence levels (lowest -> highest):
# 1. Boolean or  (||)
# 2. Boolean and (&&)
# 3. Comparisons (<, >, <=, >=, ==, !=)
# 4. Addition (+)
# 5. Boolean negation (!)
# 6. Parentheses / literals / variables

grammar = r"""
?program: stmt+

?stmt: lvar "=" expr               -> assign_stmt
     | "while" expr "{" stmt* "}"  -> loop_stmt
     | "%print" expr               -> print_stmt
     | "%println" expr             -> println_stmt

?expr: or_expr

# Boolean or has the lowest precedence
?or_expr: and_expr
        | or_expr "||" and_expr    -> or_expr

# Boolean and has higher precedence than boolean or
?and_expr: cmp_expr
         | and_expr "&&" cmp_expr  -> and_expr

# Comparisons
?cmp_expr: add_expr
         | cmp_expr "<"  add_expr  -> ult_expr
         | cmp_expr ">"  add_expr  -> ugt_expr
         | cmp_expr "<=" add_expr  -> ule_expr
         | cmp_expr ">=" add_expr  -> uge_expr
         | cmp_expr "==" add_expr  -> eq_expr
         | cmp_expr "!=" add_expr  -> ne_expr

# Addition
?add_expr: unary_expr
         | add_expr "+" unary_expr -> add_expr

# Boolean negation
?unary_expr: "!" unary_expr        -> negate_expr
           | atom

# Atoms have the highest precedence
?atom: INT                         -> int_expr
     | BOOL                        -> bool_expr
     | rvar
     | "(" expr ")"                -> paren_expr

BOOL: "%T" | "%F"

lvar: VAR
rvar: VAR

%import common.INT
%import common.CNAME -> VAR
%import common.WS
%ignore WS
"""


class IRGen(Interpreter):
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

    def __init__(self, debug=False):
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

        # Debug mode
        self.debug = debug

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

    def _assert_binary(self, node: List[Use]):
        assert len(node) == 2
        assert isinstance(node[0], Use)
        assert isinstance(node[1], Use)

    def _assert_token(self, node: List[Token]):
        assert len(node) == 1
        assert isinstance(node[0], Token)

    def _binary_op(self, op_type, node: List[Use], ast_name):
        self._dbg(f"{ast_name}", node)
        self._assert_binary(node)

        return self.func_builder.insert(op_type(node[0], node[1]))

    def _binary_logical_op(self, op_type, node: List[Use], ast_name):
        self._dbg(f"{ast_name}", node)
        self._assert_binary(node)

        return self.func_builder.insert(op_type([node[0], node[1]], i1))

    def _cmp_op(self, mnemonic, node: List[Use]) -> CmpiOp:
        self._dbg(f"{mnemonic}_expr", node)
        self._assert_binary(node)

        return self.func_builder.insert(CmpiOp(node[0], node[1], mnemonic))

    def _print_op(self, node: List[Use], fmt: str) -> PrintFormatOp:
        assert self.symbol_table is not None
        assert len(node) == 1
        assert isinstance(node[0], Use)

        return self.func_builder.insert(PrintFormatOp(fmt, node[0]))

    @visit_children_decor  # pyrefly: ignore
    def program(self, node: Node) -> FuncOp:
        self._dbg("program", node)
        self.func_builder.insert(ReturnOp())
        return self.module_builder.insert(self.func)

    def loop_stmt(self, node: Tree) -> WhileOp:
        """
        node[0] = condition expression
        node[1:] = body statements
        """
        self._dbg("loop_stmt", node)
        assert self.symbol_table is not None

        loop_carried_vars = []
        for stmt in node.children[1:]:
            if (
                stmt.data == "assign_stmt"
            ):  # Tree('assign_stmt', [Tree(Token('RULE', 'lvar'), [Token('VAR', 'y')]), ... ])
                # Only worry about variables that existed before the loop
                var = stmt.children[0].children[0]
                assert isinstance(var, Token)
                if var in self.symbol_table:
                    loop_carried_vars.append(var)

        # Initial SSAs
        initial_ssas = [self.symbol_table[var] for var in loop_carried_vars]
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

        # Map loop-carried variable names to the before block args
        for name, ssa in zip(loop_carried_vars, before_args):
            self.symbol_table[name] = ssa

        # Evaluate the condition inside the 'before' region (scf.while runs this first)
        # NOTE: We do not use do-while loops, so nothing needs to be done in the before region beyond the condition.
        cond_value_inner = self.visit(node.children[0])
        assert isinstance(cond_value_inner, Use)
        self.func_builder.insert(ConditionOp(cond_value_inner, *before_args))

        # --- AFTER REGION ---
        self.func_builder = Builder(InsertPoint.at_end(after_block))
        self.symbol_table = ScopedDict(old_symbols)

        # Map loop-carried variable names to the after block args
        for name, ssa in zip(loop_carried_vars, after_args):
            self.symbol_table[name] = ssa

        # Visit body statements
        for stmt in node.children[1:]:
            self.visit(stmt)

        # Terminate the loop body region with scf.yield
        self.func_builder.insert(
            YieldOp(*[self.symbol_table[var] for var in loop_carried_vars])
        )

        # Restore builder/scope
        self.func_builder = old_builder
        self.symbol_table = old_symbols

        # Map loop-carried variable names to the final SSA values
        for name, ssa in zip(loop_carried_vars, loop_op.res):
            self.symbol_table[name] = ssa

        # Insert loop op in parent region
        return self.func_builder.insert(loop_op)

    @visit_children_decor  # pyrefly: ignore
    def assign_stmt(self, node: Node) -> Use:
        """
        node[0] = [Token('VAR', var_name)]
        node[1] = Use
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
    def print_stmt(self, node: List[Use]) -> PrintFormatOp:
        self._dbg("print_stmt", node)
        return self._print_op(node, "{}")

    @visit_children_decor  # pyrefly: ignore
    def println_stmt(self, node: List[Use]) -> PrintFormatOp:
        self._dbg("println_stmt", node)
        return self._print_op(node, "{}\n")

    @visit_children_decor  # pyrefly: ignore
    def negate_expr(self, node: List[Use]) -> XorOp:
        true_const = self.func_builder.insert(ConstantOp(IntegerAttr(1, i1)))
        return self._binary_logical_op(XorOp, [node[0], true_const], "negate_expr")

    @visit_children_decor  # pyrefly: ignore
    def and_expr(self, node: List[Use]) -> AndOp:
        return self._binary_logical_op(AndOp, node, "and_expr")

    @visit_children_decor  # pyrefly: ignore
    def or_expr(self, node: List[Use]) -> OrOp:
        return self._binary_logical_op(OrOp, node, "or_expr")

    @visit_children_decor  # pyrefly: ignore
    def add_expr(self, node: List[Use]) -> AddiOp:
        return self._binary_op(AddiOp, node, "add_expr")

    @visit_children_decor  # pyrefly: ignore
    def ult_expr(self, node: List[Use]) -> CmpiOp:
        return self._cmp_op("ult", node)

    @visit_children_decor  # pyrefly: ignore
    def ugt_expr(self, node: List[Use]) -> CmpiOp:
        return self._cmp_op("ugt", node)

    @visit_children_decor  # pyrefly: ignore
    def uge_expr(self, node: List[Use]) -> CmpiOp:
        return self._cmp_op("uge", node)

    @visit_children_decor  # pyrefly: ignore
    def ule_expr(self, node: List[Use]) -> CmpiOp:
        return self._cmp_op("ule", node)

    @visit_children_decor  # pyrefly: ignore
    def eq_expr(self, node: List[Use]) -> CmpiOp:
        return self._cmp_op("eq", node)

    @visit_children_decor  # pyrefly: ignore
    def ne_expr(self, node: List[Use]) -> CmpiOp:
        return self._cmp_op("ne", node)

    @visit_children_decor  # pyrefly: ignore
    def paren_expr(self, node: List[Use]) -> Use:
        self._dbg("paren_expr", node)
        assert len(node) == 1
        return node[0]

    @visit_children_decor  # pyrefly: ignore
    def int_expr(self, node: List[Token]) -> ConstantOp:
        """
        node[0] = [Token('INT', '[0-9]')]
        """
        self._dbg("int_expr", node)
        self._assert_token(node)

        return self.func_builder.insert(ConstantOp(IntegerAttr(int(node[0]), i32)))

    @visit_children_decor  # pyrefly: ignore
    def bool_expr(self, node: List[Token]) -> ConstantOp:
        """
        node[0] = [Token('BOOL', '%T' | '%F')]
        """
        self._dbg("bool_expr", node)
        self._assert_token(node)

        if node[0] == "%T":
            return self.func_builder.insert(ConstantOp(IntegerAttr(1, i1)))
        else:  # node[1] == "%F":
            return self.func_builder.insert(ConstantOp(IntegerAttr(0, i1)))

    @visit_children_decor  # pyrefly: ignore
    def rvar(self, node: List[Token]) -> SSAValue[Attribute]:
        """
        node[0] = [Token('VAR', rvar)]
        """
        self._dbg("rvar", node)
        self._assert_token(node)
        assert self.symbol_table is not None

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
