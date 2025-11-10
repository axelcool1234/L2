# src/ic3/ic3.py

from typing import Set
import z3
from typing import List
from dataclasses import dataclass
from itertools import count


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
    Stored as a conjunction of clauses (CNF).
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
        self.frames: List[Frame] = []

    def prove(self):
        """
        Main IC3 loop.

        Returns:
            True if property holds
            False if counterexample found
            None if max iterations reached
        """
        # Notes:
        # A formula F implies another formula G, written F -> G,
        # if every satisfying assignment of F satisfies G.

        # Initially, the satisfiability of I ∧ ¬P and I ∧ T ∧ ¬P'
        # are checked to detect 0-step and 1-step counterexamples.
        solver = z3.Solver()
        solver.add(self.initial)
        solver.add(z3.Not(self.property))
        if solver.check() == z3.sat:  # I ∧ ¬P
            print("Property violated in 0-step (initial state)")
            return False
        solver = z3.Solver()
        solver.add(self.initial)
        solver.add(self.transition)
        solver.add(z3.Not(self._prime_formula(self.property)))
        if solver.check() == z3.sat:  # I ∧ T ∧ ¬P'
            print("Property violated in 1-step")
            return False

        # Initialize F_0, F_1, ... to assume that P is invariant,
        # while their clause sets are initialized to empty.
        self.frames.append(Frame(set(), 0))  # F_0 := I, clauses(F_0) := {}
        self.frames.append(Frame(set(), 1))  # F_1 := P, clauses(F_1) := {}
        # As a formula, each F_i for i > 0 is interpreted as P ∧ /\ clauses(F_i).
        # F_0 is interpreted as I ∧ /\ clauses(F_0).

        for k in count(1):
            # Assertions:
            # (1): for all i >= 0, I -> F_i
            # (2): for all i >= 0, F_i -> P
            # (3): for all i >  0, clauses(F_i+1) ⊆ clauses(F_i)
            # (4): for all 0 <= i < k, F_i ∧ T -> F'_i+1
            # (5): for all i > k, |clauses(F_i)| = 0

            # Extend with new frame
            self.frames.append(Frame(set(), k + 1))

            if not self.strengthen(k):
                print(f"IC3: Found counterexample at iteration {k}")
                return False

            self.propagate_clauses(k)

            # If during the process of propagating clauses forward it is discovered
            # that clauses(F_i) == clauses(F_i+1) for some i, the proof is complete:
            # F_i is an inductive strengthening of P, proving P is an invariant.
            for i in range(1, k + 1):
                if self.frames[i].clauses == self.frames[i + 1].clauses:
                    print(f"IC3: Converged at frame {i}!")
                    return True
        return None

    def strengthen(self, k: int):
        # Strengthens F_i for 1 <= i <= k
        # so that F_i-states are at least
        # k - i + 1 steps away from violating P
        # Iterates until F_k excludes all states
        # that lead to a violation of P in one
        # step.
        raise Exception("IC3Prover is unimplemented!")

    def propagate_clauses(self, k: int):
        # Propagate clauses forward through F_1, F_2, ..., F_k+1.
        for i in range(1, k + 1):
            for clause in self.frames[i].clauses:
                pass
        raise Exception("IC3Prover is unimplemented!")

    def _prime_formula(self, formula: z3.BoolRef):
        """Returns the primed form of a given formula"""
        # Applying prime to a formula, F', is the same as
        # priming all of its variables.
        subst = [(self.state_vars[v], self.next_state_vars[v]) for v in self.vars]
        result = z3.substitute(formula, subst)
        assert isinstance(result, z3.BoolRef)
        return result
