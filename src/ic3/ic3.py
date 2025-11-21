# src/ic3/ic3.py

from typing import Tuple
from typing import Set
import z3
from typing import List
from dataclasses import dataclass
from itertools import count


@dataclass
class Clause:
    """A clause is a disjunction of literals (CNF)"""

    literals: List[z3.BoolRef]

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

    assumption: z3.BoolRef
    clauses: Set[Clause]
    level: int

    def to_z3(self) -> z3.ExprRef:
        if not self.clauses:
            return self.assumption
        smt_and = z3.And(
            [clause.to_z3() for clause in self.clauses] + [self.assumption]
        )
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
        debug=False,
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

        self._debug = debug

    def _debug_print(self, string: str, dump=False):
        if self._debug:
            print(f"[IC3] {string}")
            if dump:
                self._dump_frames()

    def _dump_frames(self):
        """Print detailed frame information"""
        if not self._debug:
            return
        print("=" * 80)
        print("FRAME DUMP:")
        for i, frame in enumerate(self.frames):
            print(f"  Frame F_{i} (level {frame.level}):")
            print(f"    Assumption: {frame.assumption}")
            if frame.clauses:
                print(f"    Clauses ({len(frame.clauses)}):")
                for j, clause in enumerate(frame.clauses):
                    print(f"      C{j}: {clause.literals}")
            else:
                print("    Clauses: (empty)")
            print()
        print("=" * 80)

    def _debug_solver(self, solver, context="Unknown context"):
        """Enhanced solver state printing with context"""
        if not self._debug:
            return
        print(f"[SOLVER] {context}")
        print("-" * 60)
        assertions = solver.assertions()
        if assertions:
            print(f"Solver has {len(assertions)} assertion(s):")
            for i, assertion in enumerate(assertions):
                print(f"  {i + 1}: {assertion}")
        else:
            print("Solver has no assertions")

        # Check solver state
        check_result = solver.check()
        print(f"Solver check result: {check_result}")

        if check_result == z3.sat:
            model = solver.model()
            print("Model (variable assignments):")
            relevant_vars = []
            for decl in model.decls():
                if any(var_name in str(decl) for var_name in self.vars):
                    relevant_vars.append((decl, model[decl]))

            if relevant_vars:
                for decl, val in sorted(relevant_vars, key=lambda x: str(x[0])):
                    print(f"  {decl} = {val}")
            else:
                print("  (no relevant variable assignments found)")

        print("-" * 60)

    def prove(self):
        """
        Main IC3 loop.

        Returns:
            True if property holds
            False if counterexample found
            None if it couldn't be determined
        """
        # Notes:
        # A formula F implies another formula G, written F -> G,
        # if every satisfying assignment of F satisfies G.

        # Initially, the satisfiability of I ∧ ¬P and I ∧ T ∧ ¬P'
        # are checked to detect 0-step and 1-step counterexamples.
        solver = z3.Solver()
        solver.add(self.initial)
        solver.add(z3.Not(self.property))
        self._debug_solver(solver, "Checking 0-step counterexample: I ∧ ¬P")
        if solver.check() == z3.sat:  # I ∧ ¬P
            self._debug_print("Property violated in 0-step (initial state)", dump=True)
            return False

        solver = z3.Solver()
        solver.add(self.initial)
        solver.add(self.transition)
        solver.add(z3.Not(self._prime_formula(self.property)))
        self._debug_solver(solver, "Checking 1-step counterexample: I ∧ T ∧ ¬P'")
        if solver.check() == z3.sat:  # I ∧ T ∧ ¬P'
            self._debug_print("Property violated in 1-step", dump=True)
            return False

        # Initialize F_0, F_1, ... to assume that P is invariant,
        # while their clause sets are initialized to empty.
        self.frames.append(
            Frame(self.initial, set(), 0)
        )  # F_0 := I, clauses(F_0) := {}
        self.frames.append(
            Frame(self.property, set(), 1)
        )  # F_1 := P, clauses(F_1) := {}
        # As a formula, each F_i for i > 0 is interpreted as P ∧ /\ clauses(F_i).
        # F_0 is interpreted as I ∧ /\ clauses(F_0).

        for k in count(1):
            self._debug_print(f"Starting IC3 iteration {k}")

            # Assertions:
            # (1): for all i >= 0, I -> F_i
            # (2): for all i >= 0, F_i -> P
            # (3): for all i >  0, clauses(F_i+1) ⊆ clauses(F_i)
            # (4): for all 0 <= i < k, F_i ∧ T -> F'_i+1
            # (5): for all i >  k, |clauses(F_i)| = 0

            # Extend with new frame: F_k+1 := P
            self.frames.append(Frame(self.property, set(), k + 1))
            self._debug_print(f"Extended with new frame F_{k + 1}")

            # Strengthen F_k
            self._debug_print(f"Strengthening frame F_{k}")
            if not self.strengthen(k):
                self._debug_print(f"Found counterexample at iteration {k}", dump=True)
                return False

            # Propagate clauses
            self._debug_print(f"Propagating clauses forward from F_1 to F_{k}")
            self.propagate_clauses(k)

            # Check for convergence:
            # If during the process of propagating clauses forward it is discovered
            # that clauses(F_i) == clauses(F_i+1) for some i, the proof is complete:
            # F_i is an inductive strengthening of P, proving P is an invariant.
            for i in range(1, k + 1):
                if self.frames[i].clauses == self.frames[i + 1].clauses:
                    self._debug_print(f"Converged! F_{i} = F_{i + 1}", dump=True)
                    return True

            self._debug_print(f"Completed iteration {k}, continuing...", dump=True)
        return None

    def strengthen(self, k: int):
        # Strengthens F_i for 1 <= i <= k
        # so that F_i-states are at least
        # k - i + 1 steps away from violating P
        # Iterates until F_k excludes all states
        # that lead to a violation of P in one
        # step.

        # While sat(F_k ∧ T ∧ ¬P') i.e. while a bad state exists
        # The bad state s being one of which is one step away from
        # violating P.
        strengthen_iteration = 0
        while True:
            strengthen_iteration += 1
            self._debug_print(f"Strengthen iteration {strengthen_iteration} for F_{k}")

            # Is there a state s in F_k that can take a single
            # transition into a state s' that violates P?
            solver = z3.Solver()
            solver.add(self.frames[k].to_z3())
            solver.add(self.transition)
            solver.add(z3.Not(self._prime_formula(self.property)))

            self._debug_solver(
                solver, f"Checking for bad states in F_{k}: F_{k} ∧ T ∧ ¬P'"
            )

            if solver.check() == z3.unsat:
                # If the answer to the above question is no,
                # F_k is strong enough
                self._debug_print(f"F_{k} is strong enough (no bad states found)")
                break

            # If the answer to the above question is yes,
            # there is a pair of states (s, s') such that
            # s satisfies F_k, (s, s') satisfies T, and
            # s' violates P (i.e. s' ⊨ ¬P)
            state = self._extract_cube(
                solver.model()
            )  # this state can reach a bad state in one step
            self._debug_print(f"Found bad state: {state}")

            self._debug_print("Inductively generalizing state...")
            n = self.inductively_generalize(state, k - 2, k)
            if n is None:
                self._debug_print("Failed to generalize - counterexample found")
                return False  # Counterexample

            self._debug_print(f"State generalized at level {n}, pushing forward...")
            # After finding that a state s can reach a bad state and
            # discovering ¬s is inductive relative to some F_n, we need
            # to "push" this generalization forward to higher frames
            # until we can block s at level k.
            if self.push_generalization({(n + 1, state)}, k) is False:
                self._debug_print(
                    "Failed to push generalization - counterexample found"
                )
                return False  # Counterexample
        return True

    def inductively_generalize(self, state: z3.BoolRef, min_level: int, k: int):
        # Given:
        # Current frame sequence F_0, ..., F_k
        # A state s such that s ∈ F_k and there's a transition T(s, s') where
        # s' ⊨ ¬P
        #
        # We want to find a clause c ⊆ ¬s (i.e. a generalization of "not s")
        # such that F_i ∧ T -> c' for some i.
        #
        # That means c is inductive relative to F_i - if all F_i-states satisfy
        # c, all their successors will too.
        #
        # If we can find such c, we can add it to F_i+1 (and below) to "block" not
        # only s, but also many "similar" states that might lead to violations in
        # <= k steps (this is because c is a generalization of "not s").
        #
        # Intuition behind this method:
        # If s can reach a bad state, that means ¬s is not inductive at the
        # frontier F_k. However, it might be inductive relative to some earlier
        # frame F_i (which describes smaller reachable sets).
        #
        # So IC3 here is trying to find the weakest F_i for which ¬s is inductive.
        #
        # If we can find that, we know that no state reachable within i steps can
        # transition into s. We can then add a clause c to F_i+1 that prevent
        # s-like states from ever appearing in deeper frames.
        #
        # What does "weakest F_i" mean?
        # In logic, a weaker formula is one that allows more states (i.e. it's
        # easier to satisfy). Formally: F_a is weaker than F_b if F_b -> F_a
        #
        # That is, every F_b-state also satisfies F_a, but not vice versa.
        # In IC3:
        # I -> F_0 -> F_1 -> F_2 -> ... -> F_k
        #
        # So:
        # F_0 is the strongest (smallest set of states), F_k is the weakest (largest
        # set of states).
        #
        # So when we want the weakest F_i for which ¬s is inductive, we want to push
        # the blocking clause c as far forward as possible (to the latest, weakest
        # level where it still remains inductive) so it cuts off as few states as
        # necessary.
        #
        # Why are we conjoining clause c to each F_0, ..., F_i+1?
        # The goal is to ensure the inductive relationships among frames remain valid:
        # We know c is inductive RELATIVE to F_i, i.e. F_i ∧ T -> c'. This means that
        # if all F_i-states satisfy c, then all their successors (the F_i+1-states)
        # also satisfy c. However, to preserve this fact, we must guarantee all
        # F_i-states DO satisfy c. The safe way to do that is to conjoin c to every
        # frame up to and including F_i+1. This ensures that the inductive step
        # F_i ∧ T -> F_i+k' still holds and that the chain F_0 -> F_1 -> ... -> F_i+1
        # remains monotonic.
        #
        # Monotonicity refers to the logical ordering of the IC3 frames. It means as
        # we go forward through the sequences of frames F_0, F_1, F_2, ..., each one
        # is weaker (larger set of states). This relationship MUST hold.
        #
        # A sequence of formulas F_0, F_1, F_2, ..., is monotonic if:
        # F_0 -> F_1 -> F_2 -> ... -> F_k
        #
        # That is, every state that satisfies F_i also satisfies F_i+1. Equivalently,
        # the set of states that satisfy a frame is non-decreasing:
        # states(F_0) ⊆ states(F_1) ⊆ states(F_2) ... ⊆ states(F_k)
        #
        # Why does IC3 require monotonicity?
        # Each F_i is meant to represent the states reachable in <= i steps. If you
        # reach a state in <= i steps, you can certainly reach it in <= i + 1 steps.
        # So the reachable sets must satisfy:
        # R_0 ⊆ R_1 ⊆ R_2 ... ⊆ R_k
        #
        # The frames are over-approximations of these reachable sets, and they must
        # follow the same property to remain sound. This is why the algorithm
        # enforces:
        # clauses(F_i+1) ⊆ clauses(F_i)
        #
        # Which means F_i -> F_i+1. Which also means states(F_i) ⊆ states(F_i+1).
        #
        # So why does c need to be conjoined to frames F_0, ..., F_i+1?
        # If we were to add c only to some of the earlier frames, it could break
        # monotonicity. Think about it like this:
        # states(F_0) ⊆ states(F_1) ⊆ states(F_2) ... ⊆ states(F_k)
        #
        # Adding a clause c to a later frame strengthens it, which means the set
        # of states shrink. This could mean the set of states of an earlier frame
        # is no longer a subset, breaking IC3's required non-decreasing monotonicity.
        # To fix this, we simply conjoin c to all previous frames so that they all
        # strengthen and thus their set of states shrink.
        #
        # clauses(F_i+1) ⊆ clauses(F_i) may seem backwards - how is that the same
        # as F_i -> F_i+1 or states(F_i) ⊆ states(F_i+1)?
        #
        # More clauses = stronger formula = smaller set of states

        # Check if state is reachable from I
        # (if it is, P is not an invariant).
        if min_level < 0:  # and sat(F_0 ∧ T ∧ ¬s ∧ s')
            solver = z3.Solver()
            solver.add(self.frames[0].to_z3())
            solver.add(self.transition)
            solver.add(z3.Not(state))
            solver.add(self._prime_formula(state))

            self._debug_solver(
                solver, "Checking if state is reachable from initial: F_0 ∧ T ∧ ¬s ∧ s'"
            )

            if solver.check() == z3.sat:
                # State has an initial predecessor.
                self._debug_print(
                    "State is reachable from initial state - counterexample!"
                )
                return None

        # Find maximum i where ¬s is NOT inductive relative to F_i
        # For ¬s to be inductive relative to F_i, the following
        # has to be true:
        # F_i ∧ ¬s ∧ T -> ¬s'
        # ¬(F_i ∧ ¬s ∧ T) ∨ ¬s'
        #
        # To check if the above is true in all states,
        # we need to show that the opposite is unsat.
        # So we negate it:
        # ¬(¬(F_i ∧ ¬s ∧ T) ∨ ¬s')
        # F_i ∧ ¬s ∧ T ∧ s'
        #
        # Since we do NOT want ¬s to be inductive relative to F_i,
        # that means we want sat.
        for i in range(max(1, min_level + 1), k + 1):
            solver = z3.Solver()
            solver.add(self.frames[i].to_z3())
            solver.add(self.transition)
            solver.add(z3.Not(state))
            solver.add(self._prime_formula(state))

            self._debug_solver(
                solver,
                f"Checking if ¬s is NOT inductive relative to F_{i}: F_{i} ∧ T ∧ ¬s ∧ s'",
            )

            if solver.check() == z3.sat:
                # ¬s is not inductive relative to F_i
                self._debug_print(
                    f"¬s is NOT inductive relative to F_{i}, generating clause at level {i - 1}"
                )
                self.generate_clause(state, i - 1, k)
                return i - 1

        self._debug_print(
            f"¬s is inductive relative to all frames up to F_{k}, generating clause at level {k}"
        )
        self.generate_clause(state, k, k)
        return k

    def generate_clause(self, state: z3.BoolRef, i: int, k: int):
        """
        Generate a clause from ¬state that's inductive relative to F_i.
        Add it to F_1, ..., F_i+1.
        """
        self._debug_print(f"Generating clause from state at level {i}")
        clause = self._generalize(state, self.frames[i])
        self._debug_print(f"Generated clause: {clause.literals}")

        for j in range(1, i + 2):
            self.frames[j].clauses.add(clause)
            self._debug_print(f"Added clause to F_{j}")

    def push_generalization(self, states: Set[Tuple[int, z3.BoolRef]], k: int):
        """Pushes inductive generalization to higher frame levels."""
        # Insight: If a state s is not inductive relative to F_i, apply inductive
        # generalization to its F_i-state predecessors. The states variable {(i, s)}
        # represents the knowledge that s is inductive relative to F_i-1 and F_i
        # excludes s. In the loop we choose the minimal level i state in the set
        # to guarantee that none of the other states in the set can be a predecessor
        # of it.

        # If we blocked state s (the state that can reach a bad state in one step)
        # at level i, but s is still in F_k, we need to find predecessors of s
        # and block them recursively until s is blocked at F_k
        #
        # states is a set of (level, state) pairs representing states that need
        # to be blocked (and it includes which level they've been blocked so far)
        #
        # This method is initially called with {(n+1, state)}, k. n is the level
        # where ¬s was found to be inductive. So s has been blocked up to level
        # n+1.
        push_iteration = 0
        while True:
            push_iteration += 1
            self._debug_print(f"Push generalization iteration {push_iteration}")

            # (level, state)
            # pick state with minimum level in the set
            (n, state) = min(states, key=lambda state: state[0])
            self._debug_print(f"Processing state at level {n}: {state}")

            if n > k:
                self._debug_print(f"All states blocked above level {k}, push complete")
                return True  # All states blocked above k, so we are done.

            solver = z3.Solver()
            solver.add(self.frames[n].to_z3())
            solver.add(self.transition)
            solver.add(self._prime_formula(state))

            self._debug_solver(
                solver, f"Checking for predecessors in F_{n}: F_{n} ∧ T ∧ s'"
            )

            # if sat(F_n ∧ T ∧ s'):
            if solver.check() == z3.sat:
                # Found a predecessor
                # This means there exists a state in F_n that can transition to
                # state s. This predecessor is blocking us from pushing state s
                # higher. So we recursively block the predecessor first. We do
                # this by adding it to the states set.
                predecessor = self._extract_cube(solver.model())
                self._debug_print(f"Found predecessor: {predecessor}")
                self._debug_print("Recursively blocking predecessor...")

                m = self.inductively_generalize(predecessor, n - 2, k)
                if m is None:
                    self._debug_print(
                        "Failed to generalize predecessor - counterexample found"
                    )
                    return False

                states.add((m + 1, predecessor))
                self._debug_print(f"Added predecessor to states set at level {m + 1}")
            else:
                self._debug_print(
                    "No predecessors found, pushing state to higher level"
                )
                m = self.inductively_generalize(state, n, k)
                if m is None:
                    self._debug_print(
                        "Failed to generalize state - counterexample found"
                    )
                    return False

                states.discard((n, state))
                states.add((m + 1, state))
                self._debug_print(
                    f"State moved from level {n} to level {m + 1} in states set"
                )

    def propagate_clauses(self, k: int):
        """
        Propagate clauses forward through F_1, ..., F_k.
        If F_i ∧ T -> c', add c to clauses(F_i+1).
        """
        # Propagate clauses forward through F_1, F_2, ..., F_k+1.
        # A clause is only propagated to F_i+1 if it is inductive
        # relative to F_i. To determine this, it needs to satisfy
        # consecution relative to F_i.
        propagated_count = 0

        for i in range(1, k + 1):
            frame_propagated = 0
            self._debug_print(f"Propagating clauses from F_{i} to F_{i + 1}")

            for clause in self.frames[i].clauses:
                consecution_result = self._check_consecution(clause, self.frames[i])

                if consecution_result is None:
                    # Clause doesn't violate consecution relative to F_i,
                    # which means it's inductive relative to F_i. This
                    # means it can be safely conjoined with clauses(F_i+1)
                    self.frames[i + 1].clauses.add(clause)
                    frame_propagated += 1
                    propagated_count += 1
                else:
                    self._debug_print(
                        f"Clause {clause.literals} cannot be propagated (consecution fails)"
                    )

            self._debug_print(
                f"Propagated {frame_propagated} clauses from F_{i} to F_{i + 1}"
            )

        self._debug_print(f"Total clauses propagated: {propagated_count}")

    def _generalize(self, cube: z3.BoolRef, frame: Frame):
        """
        Find minimal inductive subclause of ¬cube relative to frame.
        Implements the "down" algorithm.
        """
        # Convert cube (conjunction) to clause (disjunction of negations)
        literals: List[z3.BoolRef] = []
        if z3.is_and(cube):
            for arg in cube.children():
                literal = z3.Not(arg)
                assert isinstance(literal, z3.BoolRef)
                literals.append(literal)
        else:
            literal = z3.Not(cube)
            assert isinstance(literal, z3.BoolRef)
            literals.append(literal)

        clause = Clause(literals)
        # TODO: Implement the down algorithm from https://theory.stanford.edu/~arbrad/papers/fsis.pdf
        # For now, just pretend the clause we have is the most generalized.
        return clause

    def _prime_formula(self, formula: z3.BoolRef):
        """Returns the primed form of a given formula"""
        # Applying prime to a formula, F', is the same as
        # priming all of its variables.
        subst = [(self.state_vars[v], self.next_state_vars[v]) for v in self.vars]
        result = z3.substitute(formula, subst)
        assert isinstance(result, z3.BoolRef)
        return result

    def _extract_cube(self, model: z3.ModelRef):
        """
        Extract a cube (conjunction of literals) from a model.
        A cube represents a concrete state.
        """
        literals = []
        for v in self.vars:
            val = model.eval(self.state_vars[v], model_completion=True)
            literals.append(self.state_vars[v] == val)
        cube = z3.And(literals)
        assert isinstance(cube, z3.BoolRef)
        return cube

    def _check_consecution(self, clause: Clause, frame: Frame):
        """
        Check if frame ∧ clause ∧ T -> clause'.
        Returns None if holds, predecessor cube if fails.
        """
        # F ∧ c ∧ T -> c' is equivalent to:
        # ¬(F ∧ c ∧ T) ∨ c'
        #
        # To check if the above is true in all states,
        # we need to show that the opposite is unsat.
        # So we negate it:
        # ¬(¬(F ∧ c ∧ T) ∨ c')
        # ¬¬(F ∧ c ∧ T) ∧ ¬c')
        # F ∧ c ∧ T ∧ ¬c'
        # If the above is SAT, this clause violates its
        # primed counterpart clause' in one step.
        solver = z3.Solver()
        solver.add(frame.to_z3())
        solver.add(clause.to_z3())
        solver.add(self.transition)
        solver.add(z3.Not(self._prime_formula(clause.to_z3())))

        self._debug_solver(
            solver,
            f"Checking consecution for clause {clause.literals}: F_{frame.level} ∧ c ∧ T ∧ ¬c'",
        )

        if solver.check() == z3.sat:
            # Found predecessor
            return self._extract_cube(solver.model())
        else:
            return None

    def _check_initiation(self, clause: Clause) -> bool:
        """Check if I -> clause"""
        # I -> c
        # ¬I ∨ c
        #
        # To check if the above is true in all states,
        # we need to show that the opposite is unsat.
        # So we negate it:
        # ¬(¬I ∨ c)
        # I ∧ ¬c
        # If the above is SAT, this clause fails
        # to satisfy initiation.
        solver = z3.Solver()
        solver.add(self.initial)
        solver.add(z3.Not(clause.to_z3()))

        self._debug_solver(
            solver, f"Checking initiation for clause {clause.literals}: I ∧ ¬c"
        )

        result = solver.check() == z3.unsat
        self._debug_print(f"Initiation check result: {result}")
        return result
