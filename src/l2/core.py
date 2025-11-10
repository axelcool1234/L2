# src/l2/core.py

from typing import List, Union

from lark.lexer import Token
from lark.tree import Tree
from lark.visitors import (
    Interpreter as LarkInterpreter,
)
from lark.visitors import (
    visit_children_decor,
)
from xdsl.builder import Builder
from xdsl.dialects import arith, builtin, func, llvm, printf, scf
from xdsl.dialects.builtin import IntegerAttr
from xdsl.ir import Block, Region
from xdsl.ir.core import (
    Attribute,
    BlockArgument,
    OpResult,
    SSAValue,
)
from xdsl.rewriter import InsertPoint
from xdsl.utils.scoped_dict import ScopedDict

from dialects import bigint, bignum

Op = Union[
    arith.ConstantOp,
    arith.AndIOp,
    arith.OrIOp,
    arith.XOrIOp,
    bignum.ConstantOp,
    bignum.AddOp,
    bignum.EqOp,
    bignum.NeqOp,
    bignum.GtOp,
    bignum.GteOp,
    bignum.LtOp,
    bignum.LteOp,
    bignum.FromElementsOp,
    bignum.InsertOp,
    bignum.ExtractOp,
    bigint.ConstantOp,
    bigint.AddOp,
    bigint.EqOp,
    bigint.NeqOp,
    bigint.GtOp,
    bigint.GteOp,
    bigint.LtOp,
    bigint.LteOp,
    bigint.FromElementsOp,
    bigint.InsertOp,
    bigint.ExtractOp,
]
Use = Union[Op, OpResult, BlockArgument]
Node = List[Union[Token, Use]]
Print = Union[
    printf.PrintFormatOp,
    bignum.PrintOp,
    bignum.PrintlnOp,
    bigint.PrintOp,
    bigint.PrintlnOp,
]

precedence = r"""
Precedence levels (lowest -> highest):
1. Boolean or  (||)
2. Boolean and (&&)
3. Comparisons (<, >, <=, >=, ==, !=)
4. Addition (+)
5. Boolean negation (!)
6. Parentheses / literals / variables
"""

grammar = r"""
?program: stmt+

?stmt: lvar "=" expr               -> assign_stmt
     | lvar "[" expr "]" "=" expr  -> array_assign_stmt
     | "%while" expr "{" stmt* "}" -> loop_stmt
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
           | "%len" atom           -> array_len_expr
           | atom

# Atoms have the highest precedence
?atom: INT                         -> int_expr
     | BOOL                        -> bool_expr
     | rvar "[" expr "]"           -> array_load_expr
     | rvar
     | "(" expr ")"                -> paren_expr
     | "[" expr ("," expr)* "]" -> array_literal

BOOL: "%T" | "%F"

lvar: VAR
rvar: VAR

COMMENT: ";" /[^\n]/*
%ignore COMMENT

%import common.INT
%import common.CNAME -> VAR
%import common.WS
%ignore WS
"""


class IRGen(LarkInterpreter):
    """
    Implementation of a simple MLIR emission from the L2 AST.
    """

    _builder: Builder
    _func: func.FuncOp
    """
    The builder is a helper class to create IR inside a function. The builder
    is stateful, in particular it keeps an "insertion point": this is where
    the next operations will be introduced.
    """

    _symbol_table: ScopedDict[str, SSAValue] | None = None
    """
    The symbol table maps a variable name to a value in the current scope.
    Entering a function creates a new scope, and the function arguments are
    added to the mapping. When the processing of a function is terminated, the
    scope is destroyed and the mappings created in this scope are dropped.
    """

    def __init__(self, debug=False):
        # Create one singular function where every transformed L2 instruction will be inserted into.
        entry_block = Block()
        func_region = Region(entry_block)
        self._func = func.FuncOp.from_region("main", [], [], func_region)
        self._builder = Builder(InsertPoint.at_end(entry_block))

        # Create an empty symbol table for variable name to SSA value mappings
        self._symbol_table = ScopedDict()

        # Debug mode
        self._debug = debug

    def _dbg(self, name: str, node):
        if self._debug:
            print(f"\n[DEBUG] Visiting {name}:")
            try:
                print(f"  node = {node}")
                print("  -----------------------")
                print(f"  len(node) = {len(node)}")
                print("  -----------------------")

                for i in range(len(node)):
                    try:
                        print(f"  node[{i}] = {node[i]}")
                    except Exception as e:
                        print(f"  node[{i}] missing: {e}")
                    print("  -----------------------")
            except TypeError:
                print("  node doesn't support indexing")
                print("  -----------------------")
            except Exception as e:
                print(f"  Error accessing node: {e}")
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

        return self._builder.insert(op_type(node[0], node[1]))

    def _cmp_op(self, op_type, node: List[Use], ast_name):
        self._dbg(f"{ast_name}", node)
        self._assert_binary(node)

        return self._builder.insert(op_type(node[0], node[1]))

    def _print_op(self, node: List[Use], fmt: str) -> Print:
        assert self._symbol_table is not None
        assert len(node) == 1
        assert isinstance(node[0], Use)

        val = node[0]

        # If the value is an Op, get its result SSAValue
        if isinstance(val, Op):
            val = val.result

        # Handle boolean values
        if val.type == builtin.i1:
            val = self._builder.insert(arith.ExtUIOp(val, builtin.i32))
            return self._builder.insert(printf.PrintFormatOp(fmt, val))

        # Handle bigints
        elif isinstance(val.type, bigint.BigIntegerType):
            # Call the Python print function
            if fmt == "{}":
                return self._builder.insert(bigint.PrintOp(val))
            else:  # fmt == "{}\n"
                return self._builder.insert(bigint.PrintlnOp(val))

        # Handle bignums
        elif isinstance(val.type, bignum.BigNumType):
            # Call the GMP print function
            if fmt == "{}":
                return self._builder.insert(bignum.PrintOp(val))
            else:  # fmt == "{}\n"
                return self._builder.insert(bignum.PrintlnOp(val))

        # Fallback: normal printing
        else:
            return self._builder.insert(printf.PrintFormatOp(fmt, val))

    @visit_children_decor  # pyrefly: ignore
    def assign_stmt(self, node: Node) -> Use:
        """
        node[0] = [Token('VAR', var_name)]
        node[1] = Use
        """
        self._dbg("assign_stmt", node)
        assert self._symbol_table is not None
        assert len(node) == 2
        assert isinstance(node[0], Token)
        assert isinstance(node[1], Use)

        var_name = str(node[0])
        if isinstance(node[1], (OpResult, BlockArgument)):
            ssa_val = node[1]
            self._symbol_table[var_name] = ssa_val
        else:
            assert isinstance(node[1], Op)
            ssa_val = node[1].result
            self._symbol_table[var_name] = ssa_val

        # TODO: Not sure how robust/correct this is...
        # Try to attach a lightweight name metadata to the defining operation.
        owner_op = ssa_val.owner

        # Skip annotating block arguments (owner can be a Block).
        if isinstance(owner_op, Block):
            return node[1]

        # Only set the attribute if it's not already present. This prevents a later
        # assignment that reuses the same SSA from clobbering the original, more
        # descriptive name.
        if "l2.var_name" not in owner_op.attributes:
            owner_op.attributes["l2.var_name"] = builtin.StringAttr(var_name)

        return node[1]

    def loop_stmt(self, node: Tree) -> scf.WhileOp:
        """
        node[0] = condition expression
        node[1:] = body statements
        """
        self._dbg("loop_stmt", node)
        assert self._symbol_table is not None

        loop_carried_vars = set()
        for stmt in node.children[1:]:
            if (
                stmt.data == "assign_stmt" or stmt.data == "array_assign_stmt"
            ):  # Tree('assign_stmt', [Tree(Token('RULE', 'lvar'), [Token('VAR', 'y')]), ... ])
                # Only worry about variables that existed before the loop
                var = stmt.children[0].children[0]
                assert isinstance(var, Token)
                if var in self._symbol_table:
                    loop_carried_vars.add(var)

        # Initial SSAs
        initial_ssas = [self._symbol_table[var] for var in loop_carried_vars]
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
        loop_op = scf.WhileOp(
            arguments=initial_ssas,
            result_types=ssa_types,
            before_region=before_region,
            after_region=after_region,
        )

        # Save current builder/scope
        old_builder = self._builder
        old_symbols = self._symbol_table

        # --- BEFORE REGION ---
        self._builder = Builder(InsertPoint.at_end(before_block))
        self._symbol_table = ScopedDict(old_symbols)

        # Map loop-carried variable names to the before block args
        for name, ssa in zip(loop_carried_vars, before_args):
            self._symbol_table[name] = ssa

        # Evaluate the condition inside the 'before' region (scf.while runs this first)
        # NOTE: We do not use do-while loops, so nothing needs to be done in the before region beyond the condition.
        cond_value_inner = self.visit(node.children[0])
        assert isinstance(cond_value_inner, Use)
        self._builder.insert(scf.ConditionOp(cond_value_inner, *before_args))

        # --- AFTER REGION ---
        self._builder = Builder(InsertPoint.at_end(after_block))
        self._symbol_table = ScopedDict(old_symbols)

        # Map loop-carried variable names to the after block args
        for name, ssa in zip(loop_carried_vars, after_args):
            self._symbol_table[name] = ssa

        # Visit body statements
        for stmt in node.children[1:]:
            self.visit(stmt)

        # Terminate the loop body region with scf.yield
        self._builder.insert(
            scf.YieldOp(*[self._symbol_table[var] for var in loop_carried_vars])
        )

        # Restore builder/scope
        self._builder = old_builder
        self._symbol_table = old_symbols

        # Map loop-carried variable names to the final SSA values
        for name, ssa in zip(loop_carried_vars, loop_op.res):
            self._symbol_table[name] = ssa

        # Insert loop op in parent region
        return self._builder.insert(loop_op)

    @visit_children_decor  # pyrefly: ignore
    def print_stmt(self, node: List[Use]) -> Print:
        self._dbg("print_stmt", node)
        return self._print_op(node, "{}")

    @visit_children_decor  # pyrefly: ignore
    def println_stmt(self, node: List[Use]) -> Print:
        self._dbg("println_stmt", node)
        return self._print_op(node, "{}\n")

    @visit_children_decor  # pyrefly: ignore
    def and_expr(self, node: List[Use]) -> arith.AndIOp:
        return self._binary_op(arith.AndIOp, node, "and_expr")

    @visit_children_decor  # pyrefly: ignore
    def or_expr(self, node: List[Use]) -> arith.OrIOp:
        return self._binary_op(arith.OrIOp, node, "or_expr")

    @visit_children_decor  # pyrefly: ignore
    def negate_expr(self, node: List[Use]) -> arith.XOrIOp:
        true_const = self._builder.insert(
            arith.ConstantOp(builtin.IntegerAttr(1, builtin.i1))
        )
        return self._binary_op(arith.XOrIOp, [node[0], true_const], "negate_expr")

    @visit_children_decor  # pyrefly: ignore
    def rvar(self, node: List[Token]) -> SSAValue[Attribute]:
        """
        node[0] = [Token('VAR', rvar)]
        """
        self._dbg("rvar", node)
        self._assert_token(node)
        assert self._symbol_table is not None

        var_name = str(node[0])
        try:
            return self._symbol_table[var_name]
        except Exception as e:
            raise Exception(f"error: Unknown variable `{var_name}`") from e

    @visit_children_decor  # pyrefly: ignore
    def lvar(self, node: List[Token]) -> Token:
        """
        node[0] = [Token('VAR', lvar)]
        """
        self._dbg("lvar", node)
        self._assert_token(node)

        return node[0]  # unwrap

    @visit_children_decor  # pyrefly: ignore
    def bool_expr(self, node: List[Token]) -> arith.ConstantOp:
        """
        node[0] = [Token('BOOL', '%T' | '%F')]
        """
        self._dbg("bool_expr", node)
        self._assert_token(node)

        if node[0] == "%T":
            return self._builder.insert(
                arith.ConstantOp(builtin.IntegerAttr(1, builtin.i1))
            )
        else:  # node[1] == "%F":
            return self._builder.insert(
                arith.ConstantOp(builtin.IntegerAttr(0, builtin.i1))
            )

    @visit_children_decor  # pyrefly: ignore
    def paren_expr(self, node: List[Use]) -> Use:
        self._dbg("paren_expr", node)
        assert len(node) == 1
        return node[0]


class IRGenInterpreter(IRGen):
    @visit_children_decor  # pyrefly: ignore
    def program(self, node: Node):
        self._dbg("program", node)
        self._builder.insert(func.ReturnOp())
        module = builtin.ModuleOp([])
        builder = Builder(InsertPoint.at_end(module.body.blocks[0]))
        builder.insert(self._func)
        return module

    @visit_children_decor  # pyrefly: ignore
    def array_assign_stmt(self, node):
        """
        node[0] = Token('VAR', var_name), type(symbol_table[var_name]) == VectorType
        node[1] = Use
        node[2] = Use
        """
        self._dbg("array_assign_stmt", node)
        assert self._symbol_table is not None
        assert isinstance(node[0], Token)
        var_name = str(node[0])
        vector = self._symbol_table[var_name]
        assert isinstance(vector.type, builtin.VectorType)
        res = self._builder.insert(bigint.InsertOp(node[2], vector, node[1]))
        self._symbol_table[var_name] = res.result
        return res

    @visit_children_decor  # pyrefly: ignore
    def array_load_expr(self, node: List[Use]):
        """
        node[0] = vector<#x!bigint.bigint>
        node[1] = Use
        """
        self._dbg("array_load_expr", node)
        assert isinstance(node[0], OpResult) or isinstance(node[0], BlockArgument)

        return self._builder.insert(bigint.ExtractOp(node[0], node[1]))

    @visit_children_decor  # pyrefly: ignore
    def array_len_expr(self, node):
        """
        node[0] = vector<#x!bigint.bigint>
        """
        self._dbg("array_len_expr", node)
        assert isinstance(node[0], OpResult) or isinstance(node[0], BlockArgument)
        assert isinstance(node[0].type, builtin.VectorType)

        return self._builder.insert(
            bigint.ConstantOp(IntegerAttr(node[0].type.get_shape()[0], builtin.i32))
        )

    @visit_children_decor  # pyrefly: ignore
    def array_literal(self, node: List[Use]):
        """
        node = [Use]
        """
        self._dbg("array_literal", node)
        return self._builder.insert(bigint.FromElementsOp(node))

    @visit_children_decor  # pyrefly: ignore
    def add_expr(self, node: List[Use]) -> bigint.AddOp:
        return self._binary_op(bigint.AddOp, node, "add_expr")

    @visit_children_decor  # pyrefly: ignore
    def ult_expr(self, node: List[Use]):
        return self._cmp_op(bigint.LtOp, node, "lt_expr")

    @visit_children_decor  # pyrefly: ignore
    def ugt_expr(self, node: List[Use]):
        return self._cmp_op(bigint.GtOp, node, "gt_expr")

    @visit_children_decor  # pyrefly: ignore
    def uge_expr(self, node: List[Use]):
        return self._cmp_op(bigint.GteOp, node, "ge_expr")

    @visit_children_decor  # pyrefly: ignore
    def ule_expr(self, node: List[Use]):
        return self._cmp_op(bigint.LteOp, node, "le_expr")

    @visit_children_decor  # pyrefly: ignore
    def eq_expr(self, node: List[Use]):
        return self._cmp_op(bigint.EqOp, node, "eq_expr")

    @visit_children_decor  # pyrefly: ignore
    def ne_expr(self, node: List[Use]):
        return self._cmp_op(bigint.NeqOp, node, "ne_expr")

    @visit_children_decor  # pyrefly: ignore
    def int_expr(self, node: List[Token]):
        """
        node[0] = Token('INT', '[0-9]+')
        """
        self._dbg("int_expr", node)
        self._assert_token(node)

        value = int(node[0])
        return self._builder.insert(bigint.ConstantOp(IntegerAttr(value, builtin.i32)))


class IRGenCompiler(IRGen):
    def _insert_bignum_decls(self, module: builtin.ModuleOp):
        builder = Builder(InsertPoint.at_start(module.body.blocks[0]))

        # Helper to insert a function
        def insert_func(name: str, inputs: list, output):
            builder.insert(
                llvm.FuncOp(
                    sym_name=name,
                    function_type=llvm.LLVMFunctionType(inputs=inputs, output=output),
                    linkage=llvm.LinkageAttr("external"),
                    cconv=llvm.CallingConventionAttr("ccc"),
                    visibility=0,
                )
            )

        # Standard functions
        ptr = llvm.LLVMPointerType.opaque()
        insert_func("l2_bignum_from_i32", [builtin.i32], ptr)
        insert_func("l2_bignum_add", [ptr, ptr], ptr)
        insert_func("l2_bignum_print", [ptr], llvm.LLVMVoidType())
        insert_func("l2_bignum_println", [ptr], llvm.LLVMVoidType())
        insert_func("l2_bignum_free", [ptr], llvm.LLVMVoidType())

        # Comparison functions returning i1
        comparisons = ["lt", "lte", "gt", "gte", "eq", "neq"]
        for cmp in comparisons:
            insert_func(f"l2_bignum_{cmp}", [ptr, ptr], builtin.i1)

    @visit_children_decor  # pyrefly: ignore
    def program(self, node: Node):
        self._dbg("program", node)
        self._builder.insert(func.ReturnOp())
        module = builtin.ModuleOp([])
        builder = Builder(InsertPoint.at_end(module.body.blocks[0]))
        builder.insert(self._func)
        self._insert_bignum_decls(module)
        return module

    @visit_children_decor  # pyrefly: ignore
    def array_assign_stmt(self, node):
        """
        node[0] = Token('VAR', var_name), type(symbol_table[var_name]) == VectorType
        node[1] = Use
        node[2] = Use
        """
        self._dbg("array_assign_stmt", node)
        assert self._symbol_table is not None
        assert isinstance(node[0], Token)
        var_name = str(node[0])
        vector = self._symbol_table[var_name]
        assert isinstance(vector.type, builtin.VectorType)
        res = self._builder.insert(bignum.InsertOp(node[2], vector, node[1]))
        self._symbol_table[var_name] = res.result
        return res

    @visit_children_decor  # pyrefly: ignore
    def array_load_expr(self, node: List[Use]):
        """
        node[0] = vector<#x!bigint.bigint>
        node[1] = Use
        """
        self._dbg("array_load_expr", node)
        assert isinstance(node[0], OpResult) or isinstance(node[0], BlockArgument)

        return self._builder.insert(bignum.ExtractOp(node[0], node[1]))

    @visit_children_decor  # pyrefly: ignore
    def array_len_expr(self, node):
        """
        node[0] = vector<#x!bigint.bigint>
        """
        self._dbg("array_len_expr", node)
        assert isinstance(node[0], OpResult) or isinstance(node[0], BlockArgument)
        assert isinstance(node[0].type, builtin.VectorType)

        return self._builder.insert(
            bignum.ConstantOp(IntegerAttr(node[0].type.get_shape()[0], builtin.i32))
        )

    @visit_children_decor  # pyrefly: ignore
    def array_literal(self, node: List[Use]):
        """
        node = [Use]
        """
        self._dbg("array_literal", node)
        return self._builder.insert(bignum.FromElementsOp(node))

    @visit_children_decor  # pyrefly: ignore
    def add_expr(self, node: List[Use]) -> bignum.AddOp:
        return self._binary_op(bignum.AddOp, node, "add_expr")

    @visit_children_decor  # pyrefly: ignore
    def ult_expr(self, node: List[Use]):
        return self._cmp_op(bignum.LtOp, node, "lt_expr")

    @visit_children_decor  # pyrefly: ignore
    def ugt_expr(self, node: List[Use]):
        return self._cmp_op(bignum.GtOp, node, "gt_expr")

    @visit_children_decor  # pyrefly: ignore
    def uge_expr(self, node: List[Use]):
        return self._cmp_op(bignum.GteOp, node, "ge_expr")

    @visit_children_decor  # pyrefly: ignore
    def ule_expr(self, node: List[Use]):
        return self._cmp_op(bignum.LteOp, node, "le_expr")

    @visit_children_decor  # pyrefly: ignore
    def eq_expr(self, node: List[Use]):
        return self._cmp_op(bignum.EqOp, node, "eq_expr")

    @visit_children_decor  # pyrefly: ignore
    def ne_expr(self, node: List[Use]):
        return self._cmp_op(bignum.NeqOp, node, "ne_expr")

    @visit_children_decor  # pyrefly: ignore
    def int_expr(self, node: List[Token]):
        """
        node[0] = Token('INT', '[0-9]+')
        """
        self._dbg("int_expr", node)
        self._assert_token(node)

        value = int(node[0])
        return self._builder.insert(
            bignum.ConstantOp(builtin.IntegerAttr(value, builtin.i32))
        )
