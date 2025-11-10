# src/ic3/ic3.py

from typing import Set
import z3
from typing import List
from dataclasses import dataclass


@dataclass
class Clause:
    """A clause is a disjunction of literals (CNF)"""

    literals: List[z3.ExprRef]

    def __hash__(self):
        return hash(tuple(sorted([str(literal) for literal in self.literals])))

    def __eq__(self, other):
        if not isinstance(other, Clause):
            return False
        return set(str(literal) for literal in self.literals) == set(
            str(literal) for literal in other.literals
        )

    def to_z3(self) -> z3.BoolRef:
        if not self.literals:
            return z3.BoolVal(False)
        smt_or = z3.Or(self.literals)
        assert isinstance(smt_or, z3.BoolRef)
        return smt_or


@dataclass
class Frame:
    """
    Represents F_i: over-approximation of states reachable in <= i steps.
    Stored as a conjunction of clauses.
    """

    clauses: Set[Clause]
    level: int

    def to_z3(self) -> z3.ExprRef:
        if not self.clauses:
            return z3.BoolVal(True)
        smt_and = z3.And([clause.to_z3() for clause in self.clauses])
        assert isinstance(smt_and, z3.BoolRef)
        return smt_and


class IC3Prover:
    """
    Minimal IC3 implementation for LoopLang.
    """

    def __init__(
        self,
        variables: List[str],
        initial: z3.BoolRef,
        transition: z3.BoolRef,
        property: z3.BoolRef | None,
    ):
        """
        variables: List of program variable names
        initial: I(x) - Initial condition formula
        transition: T(x, x') - Transition relation formula
        property: P(x) - Safety property to prove
        """
        self.vars = variables
        self.initial = initial
        self.transition = transition
        self.property = property if property is not None else z3.BoolVal(True)

        # Create Z3 variables for current and next state
        self.state_vars = {v: z3.Int(v) for v in variables}
        self.next_state_vars = {v: z3.Int(f"{v}'") for v in variables}

        # Frames F_0, F_1, ..., F_k
        # F_0 is always the initial condition
        self.frames: List[Frame] = [Frame(set(), 0)]

    def prove(self):
        raise Exception("IC3Prover is unimplemented!")
