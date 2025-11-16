# src/l2/extractor.py

from typing import Dict
from typing import List
from lark.lexer import Token
from xdsl.context import Context
from transforms.noop import LowerNoOp
from lark.lark import Lark
from xdsl.dialects import arith, builtin, scf
from xdsl.dialects.builtin import StringAttr
from xdsl.ir.core import Block, Operation, SSAValue
from z3 import z3

from dialects import bigint, noop

from lark import Transformer


assert_grammar = r"""
?program: expr

?expr: or_expr

# Boolean OR (lowest precedence)
?or_expr: and_expr
        | or_expr "||" and_expr      -> or_expr

# Boolean AND
?and_expr: cmp_expr
         | and_expr "&&" cmp_expr    -> and_expr

# Comparisons
?cmp_expr: add_expr
         | cmp_expr "<"  add_expr    -> ult_expr
         | cmp_expr ">"  add_expr    -> ugt_expr
         | cmp_expr "<=" add_expr    -> ule_expr
         | cmp_expr ">=" add_expr    -> uge_expr
         | cmp_expr "==" add_expr    -> eq_expr
         | cmp_expr "!=" add_expr    -> ne_expr

# Addition / Subtraction
?add_expr: mul_expr
         | add_expr "+" mul_expr     -> add_expr
         | add_expr "-" mul_expr     -> sub_expr

# Multiplication / Modulo
?mul_expr: unary_expr
         | mul_expr "*" unary_expr   -> mul_expr
         | mul_expr "%" unary_expr   -> mod_expr

# Unary
?unary_expr: "!" unary_expr          -> negate_expr
           | atom

# Atoms
?atom: INT                           -> int_expr
     | VAR                           -> var_expr
     | "(" expr ")"                  -> paren_expr

%import common.INT
%import common.CNAME -> VAR
%import common.WS
%ignore WS
"""


class AssertToZ3(Transformer):
    """
    Transform the assert statements into a z3 BoolRef.

    - All vars are Ints.
    - %T / %F are BoolVal(True/False).
    - + and % are integer operators.
    - Comparisons return Bool expressions.
    - &&, ||, ! mapped to And, Or, Not.
    """

    def __init__(self):
        self.vars: Dict[str, z3.ArithRef] = dict()  # z3 Ints

    def paren_expr(self, node):
        return node[0]

    def or_expr(self, node: List[z3.BoolRef]):
        return z3.Or(node[0], node[1])

    def and_expr(self, node: List[z3.BoolRef]):
        return z3.And(node[0], node[1])

    def add_expr(self, node: List[z3.ArithRef]):
        return node[0] + node[1]

    def sub_expr(self, node: List[z3.ArithRef]):
        return node[0] - node[1]

    def mod_expr(self, node: List[z3.ArithRef]):
        return node[0] % node[1]

    def mul_expr(self, node: List[z3.ArithRef]):
        return node[0] * node[1]

    def negate_expr(self, node: List[z3.ArithRef]):
        return z3.Not(node[0])

    def ult_expr(self, node: List[z3.ArithRef]):
        return node[0] < node[1]

    def ugt_expr(self, node: List[z3.ArithRef]):
        return node[0] > node[1]

    def ule_expr(self, node: List[z3.ArithRef]):
        return node[0] <= node[1]

    def uge_expr(self, node: List[z3.ArithRef]):
        return node[0] >= node[1]

    def eq_expr(self, node: List[z3.ArithRef]):
        return node[0] == node[1]

    def ne_expr(self, node: List[z3.ArithRef]):
        return node[0] != node[1]

    def var_expr(self, node: List[Token]):
        """
        node[0] = Token('VAR', var_name)
        """
        assert isinstance(node[0], Token)
        var_name = node[0].value
        assert isinstance(var_name, str)

        if var_name in self.vars:
            return self.vars[var_name]
        self.vars[var_name] = z3.Int(var_name)
        return self.vars[var_name]

    def int_expr(self, node: List[Token]):
        """
        node[0] = Token('INT', int)
        """
        assert isinstance(node[0], Token)
        integer = int(node[0].value)
        return z3.IntVal(integer)


class TransitionExtractor:
    """
    Extracts initiation condition I(x), transition relation T(x, x'),
    and safety property P(x) from LoopLang MLIR.
    """

    # FIXME: This currently does not support arrays
    # TODO: Figure out a way to use a provided symbol table
    # for better variable names.

    def __init__(self, module: builtin.ModuleOp):
        self.module = module
        self.var_to_z3: dict[
            str, z3.ArithRef
        ] = {}  # Maps variable names to Z3 variables
        self.ssa_to_z3: dict[
            SSAValue, z3.ArithRef
        ] = {}  # Maps ssa values to Z3 expressions

    def extract_from_while(self, while_op: scf.WhileOp):
        """
        Extracts variable names, I(x) T(x, x'), and P(x) from a
        while loop.

        Note: Property P is NOT generated here - it comes from
        %assert statements in the program.
        """
        # TODO: Explore Avy for potential generation of property P
        # as well. https://arieg.bitbucket.io/pdf/avy.pdf

        # Extract variable names and assign them Z3 variables for
        # current and next state
        var_names = self._extract_var_names(while_op)
        for var_name in var_names:
            self.var_to_z3[var_name] = z3.Int(var_name)
            self.var_to_z3[f"{var_name}'"] = z3.Int(f"{var_name}'")

        # Map before_region block args to current-state variables
        for var_name, block_arg in zip(var_names, while_op.before_region.block.args):
            self.ssa_to_z3[block_arg] = self.var_to_z3[var_name]

        # Map after_region block args to current-state variables
        for var_name, block_arg in zip(var_names, while_op.after_region.block.args):
            self.ssa_to_z3[block_arg] = self.var_to_z3[var_name]

        # Extract I(x), T(x, x'), and P(x)
        property = self._extract_property(while_op, var_names)
        initial = self._extract_initial(while_op, var_names)
        transition = self._extract_transition(while_op, var_names)

        assert isinstance(initial, z3.BoolRef)
        assert isinstance(transition, z3.BoolRef)
        # assert isinstance(property, z3.BoolRef)
        return var_names, initial, transition, property

    def _extract_var_names(self, while_op: scf.WhileOp):
        """Get names of loop-carried variables"""
        var_names = []

        for ssa_value in while_op.arguments:
            var_names.append(self._var_name(ssa_value))

        return var_names

    def _extract_initial(self, while_op: scf.WhileOp, var_names):
        """
        Extract initial condition I(x).

        For each loop-carried variable, create constraint:
            var_name == initial_value
        """
        constraints = []

        # Initial values entering the loop
        for var_name, initial_ssa in zip(var_names, while_op.arguments):
            var = self.var_to_z3[var_name]
            ssa = self._get_or_compute_z3_expr(initial_ssa)
            constraints.append(var == ssa)

        # Loop entry condition
        # The before_region ends with a scf.ConditionOp that determines loop entry
        condition_op = while_op.before_region.block.last_op
        if isinstance(condition_op, scf.ConditionOp):
            condition_expr = self._get_or_compute_z3_expr(condition_op.condition)
            constraints.append(condition_expr)

        return z3.And(constraints) if constraints else z3.BoolVal(True)

    def _extract_transition(self, while_op: scf.WhileOp, var_names):
        """
        Extract transition relation T(x, x').

        Map after_region block args -> current state (x)
        Map yield arguments -> next state (x')

        Symbolically execute loop to relate them.
        """

        # Symbolically execute body
        body_constraints = []
        for op in while_op.after_region.block.ops:
            if isinstance(op, scf.YieldOp):
                break

            constraint = self._process_op(op)
            if constraint is not None:
                body_constraints.append(constraint)

        yield_op = while_op.after_region.block.last_op
        assert isinstance(yield_op, scf.YieldOp)

        # Map yield arguments to next state variables
        next_state_constraints = []
        for var_name, next_ssa in zip(var_names, yield_op.arguments):
            z3_next_var = self.var_to_z3[f"{var_name}'"]
            z3_next_expr = self.ssa_to_z3[next_ssa]
            next_state_constraints.append(z3_next_var == z3_next_expr)

        all_constraints = body_constraints + next_state_constraints
        return z3.And(all_constraints) if all_constraints else z3.BoolVal(True)

    def _extract_property(self, while_op: scf.WhileOp, var_names):
        """
        Get the safety property of the while loop from %assert
        statements.
        """
        parser = Lark(assert_grammar, start="program", parser="lalr")
        generator = AssertToZ3()
        assertions = list()
        for op in while_op.after_region.block.ops:
            if not isinstance(op, noop.NoOperation):
                continue
            assertion = str(op.attributes["info"]).strip('"')
            tree = parser.parse(assertion)
            assertion = generator.transform(tree)
            assert isinstance(assertion, z3.BoolRef)
            assertions.append(assertion)
        LowerNoOp().apply(Context(), self.module)
        return z3.And(assertions)

    def _get_or_compute_z3_expr(self, ssa_value: SSAValue) -> z3.ArithRef:
        """
        Get or compute the Z3 expression for an SSA value.
        This handles values defined outside the loop.
        """
        # FIXME: Figure out some sort of deduplication between this function and _process_op
        # Check if already computed
        if ssa_value in self.ssa_to_z3:
            return self.ssa_to_z3[ssa_value]

        # Need to compute it by looking at the defining operation
        owner = ssa_value.owner

        # Process the defining operation
        if isinstance(owner, bigint.ConstantOp):  # arbitrary precision integer
            assert isinstance(owner.value, builtin.IntegerAttr)
            z3_expr = z3.IntVal(owner.value.value.data)
            self.ssa_to_z3[ssa_value] = z3_expr
            return z3_expr

        elif isinstance(owner, arith.ConstantOp):  # Boolean
            assert isinstance(owner.value, builtin.IntegerAttr)
            assert owner.value.type == builtin.i1
            z3_expr = z3.BoolVal(bool(owner.value.value.data))
            self.ssa_to_z3[ssa_value] = z3_expr
            return z3_expr

        elif isinstance(owner, bigint.AddOp):
            lhs = self._get_or_compute_z3_expr(owner.lhs)
            rhs = self._get_or_compute_z3_expr(owner.rhs)
            z3_expr = lhs + rhs
            self.ssa_to_z3[ssa_value] = z3_expr
            return z3_expr

        elif isinstance(owner, bigint.SubOp):
            lhs = self._get_or_compute_z3_expr(owner.lhs)
            rhs = self._get_or_compute_z3_expr(owner.rhs)
            z3_expr = lhs - rhs
            self.ssa_to_z3[ssa_value] = z3_expr
            return z3_expr

        elif isinstance(owner, bigint.EqOp):
            lhs = self._get_or_compute_z3_expr(owner.lhs)
            rhs = self._get_or_compute_z3_expr(owner.rhs)
            z3_expr = lhs == rhs
            self.ssa_to_z3[ssa_value] = z3_expr
            return z3_expr
        elif isinstance(owner, bigint.NeqOp):
            lhs = self._get_or_compute_z3_expr(owner.lhs)
            rhs = self._get_or_compute_z3_expr(owner.rhs)
            z3_expr = lhs != rhs
            self.ssa_to_z3[ssa_value] = z3_expr
            return z3_expr
        elif isinstance(owner, bigint.GtOp):
            lhs = self._get_or_compute_z3_expr(owner.lhs)
            rhs = self._get_or_compute_z3_expr(owner.rhs)
            z3_expr = lhs > rhs
            self.ssa_to_z3[ssa_value] = z3_expr
            return z3_expr
        elif isinstance(owner, bigint.GteOp):
            lhs = self._get_or_compute_z3_expr(owner.lhs)
            rhs = self._get_or_compute_z3_expr(owner.rhs)
            z3_expr = lhs >= rhs
            self.ssa_to_z3[ssa_value] = z3_expr
            return z3_expr
        elif isinstance(owner, bigint.LtOp):
            lhs = self._get_or_compute_z3_expr(owner.lhs)
            rhs = self._get_or_compute_z3_expr(owner.rhs)
            z3_expr = lhs < rhs
            self.ssa_to_z3[ssa_value] = z3_expr
            return z3_expr
        elif isinstance(owner, bigint.LteOp):
            lhs = self._get_or_compute_z3_expr(owner.lhs)
            rhs = self._get_or_compute_z3_expr(owner.rhs)
            z3_expr = lhs <= rhs
            self.ssa_to_z3[ssa_value] = z3_expr
            return z3_expr

        else:
            raise Exception(
                f"Cannot compute Z3 expression for SSA value defined by {owner}"
            )

    def _process_op(self, op: Operation) -> z3.ExprRef | None:
        """
        Symbolically execute a single operation and return constraints.
        Returns None if the operation just defines a value mapping.
        Returns a constraint if the operation adds a condition.
        """
        # FIXME: Figure out some sort of deduplication between this function and _get_or_compute_z3_expr

        # ===== Boolean Operations =====
        if isinstance(op, arith.ConstantOp):
            assert isinstance(op.value, builtin.IntegerAttr)

            # arith.ConstantOp is only for booleans (i1 type)
            assert op.value.type == builtin.i1
            self.ssa_to_z3[op.result] = z3.BoolVal(bool(op.value.value.data))
            return None

        elif isinstance(op, arith.AndIOp):
            lhs = self._get_or_compute_z3_expr(op.lhs)
            rhs = self._get_or_compute_z3_expr(op.rhs)
            self.ssa_to_z3[op.result] = z3.And(lhs, rhs)
            return None

        elif isinstance(op, arith.OrIOp):
            lhs = self._get_or_compute_z3_expr(op.lhs)
            rhs = self._get_or_compute_z3_expr(op.rhs)
            self.ssa_to_z3[op.result] = z3.Or(lhs, rhs)
            return None

        elif isinstance(op, arith.XOrIOp):
            lhs = self._get_or_compute_z3_expr(op.lhs)
            rhs = self._get_or_compute_z3_expr(op.rhs)
            # Must only happen with booleans
            assert isinstance(lhs, z3.BoolRef)
            self.ssa_to_z3[op.result] = z3.Xor(lhs, rhs)
            return None

        # ===== BigInt Constants =====
        elif isinstance(op, bigint.ConstantOp):
            assert isinstance(op.value, builtin.IntegerAttr)
            self.ssa_to_z3[op.result] = z3.IntVal(op.value.value.data)
            return None

        # ===== BigInt Arithmetic =====
        elif isinstance(op, bigint.AddOp):
            lhs = self._get_or_compute_z3_expr(op.lhs)
            rhs = self._get_or_compute_z3_expr(op.rhs)
            self.ssa_to_z3[op.result] = lhs + rhs
            return None

        elif isinstance(op, bigint.SubOp):
            lhs = self._get_or_compute_z3_expr(op.lhs)
            rhs = self._get_or_compute_z3_expr(op.rhs)
            self.ssa_to_z3[op.result] = lhs - rhs
            return None

        # ===== BigInt Comparisons =====
        elif isinstance(op, bigint.EqOp):
            lhs = self._get_or_compute_z3_expr(op.lhs)
            rhs = self._get_or_compute_z3_expr(op.rhs)
            self.ssa_to_z3[op.result] = lhs == rhs
            return None

        elif isinstance(op, bigint.NeqOp):
            lhs = self._get_or_compute_z3_expr(op.lhs)
            rhs = self._get_or_compute_z3_expr(op.rhs)
            self.ssa_to_z3[op.result] = lhs != rhs
            return None

        elif isinstance(op, bigint.GtOp):
            lhs = self._get_or_compute_z3_expr(op.lhs)
            rhs = self._get_or_compute_z3_expr(op.rhs)
            self.ssa_to_z3[op.result] = lhs > rhs
            return None

        elif isinstance(op, bigint.GteOp):
            lhs = self._get_or_compute_z3_expr(op.lhs)
            rhs = self._get_or_compute_z3_expr(op.rhs)
            self.ssa_to_z3[op.result] = lhs >= rhs
            return None

        elif isinstance(op, bigint.LtOp):
            lhs = self._get_or_compute_z3_expr(op.lhs)
            rhs = self._get_or_compute_z3_expr(op.rhs)
            self.ssa_to_z3[op.result] = lhs < rhs
            return None

        elif isinstance(op, bigint.LteOp):
            lhs = self._get_or_compute_z3_expr(op.lhs)
            rhs = self._get_or_compute_z3_expr(op.rhs)
            self.ssa_to_z3[op.result] = lhs <= rhs
            return None

        else:
            raise Exception(f"Unsupported operation in loop body: {op.name}")

    @staticmethod
    def _var_name(ssa_value: SSAValue):
        owner_op = ssa_value.owner
        if isinstance(owner_op, Block):
            return f"var_{id(ssa_value)}"

        if "l2.var_name" in owner_op.attributes:
            var_name = owner_op.attributes["l2.var_name"]
            assert isinstance(var_name, StringAttr)
            return var_name.data

        raise Exception(
            "variable name attributes are not robust enough! Check core.py assign_stmt method."
        )
        # return f"var_{id(ssa_value)}"
