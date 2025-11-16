from dialects.noop import NoOperation
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.ir.core import Operation
from xdsl.pattern_rewriter import op_type_rewrite_pattern
from xdsl.pattern_rewriter import RewritePattern
from xdsl.pattern_rewriter import GreedyRewritePatternApplier
from xdsl.pattern_rewriter import PatternRewriteWalker
from xdsl.dialects.builtin import ModuleOp
from xdsl.context import Context
from xdsl.passes import ModulePass


class LowerNoOperation(RewritePattern):
    """Lower NoOperation to nothing."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        if not isinstance(op, NoOperation):
            return
        rewriter.erase_op(op)


class LowerNoOp(ModulePass):
    """
    Lower noop to nothing.
    """

    name = "lower-noop"

    def apply(self, ctx: Context, op: ModuleOp):
        PatternRewriteWalker(
            GreedyRewritePatternApplier([LowerNoOperation()])
        ).rewrite_module(op)
