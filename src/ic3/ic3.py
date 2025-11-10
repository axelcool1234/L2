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

    def to_z3(self) -> z3.ExprRef:
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
        smt_and = z3.And([c.to_z3() for c in self.clauses])
        assert isinstance(smt_and, z3.BoolRef)
        return smt_and


class IC3Prover:
    """
    Minimal IC3 implementation for LoopLang.
    """

    def __init__(
        self,
        variables: List[str],
        initial: z3.ExprRef,
        transition: z3.ExprRef,
        property: z3.ExprRef,
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
        self.property = property
        raise Exception("IC3Prover is unimplemented!")

    def prove(self):
        raise Exception("IC3Prover is unimplemented!")
