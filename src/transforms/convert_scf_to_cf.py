# src/transforms/convert_scf_to_cf.py

# ruff: noqa: F405
from xdsl.transforms.convert_scf_to_cf import *  # noqa: F403
from xdsl.dialects.scf import WhileOp, ConditionOp
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint


class WhileLowering(RewritePattern):
    """
    Lowers `scf.while` to conditional branching.

    Creates a CFG subgraph where the 'before' region checks the condition
    and the 'after' region contains the loop body. Both regions are inlined
    and their terminators are rewritten to organize control flow.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, while_op: WhileOp, rewriter: PatternRewriter, /):
        condition_block = while_op.parent_block()
        assert condition_block is not None

        # Split the current block before the WhileOp to create the inlining point
        continuation_block = condition_block.split_before(while_op)

        # Get the before and after regions
        before_region = while_op.before_region
        after_region = while_op.after_region
        before_block = before_region.first_block
        after_block = after_region.first_block
        assert before_block is not None
        assert after_block is not None

        # Get the last blocks and terminators BEFORE inlining
        before_last_block = before_region.last_block
        after_last_block = after_region.last_block
        assert before_last_block is not None
        assert after_last_block is not None

        before_terminator = before_last_block.last_op
        after_terminator = after_last_block.last_op
        assert isinstance(before_terminator, ConditionOp)
        assert isinstance(after_terminator, YieldOp)

        # Inline the after region first (so it comes after before region)
        rewriter.inline_region(
            after_region, BlockInsertPoint.before(continuation_block)
        )

        # Inline the before region
        rewriter.inline_region(before_region, BlockInsertPoint.before(after_block))

        # Branch to the "before" region from the condition block
        rewriter.insert_op(
            BranchOp(before_block, *while_op.arguments),
            InsertPoint.at_end(condition_block),
        )

        # Replace the condition terminator in the before region
        cond_args = list(before_terminator.args)

        # Replace with conditional branch: if true go to after, else go to continuation
        rewriter.replace_op(
            before_terminator,
            ConditionalBranchOp(
                before_terminator.condition,
                after_block,
                cond_args,
                continuation_block,
                (),
            ),
        )

        # Replace the yield terminator in the after region
        # Replace with branch back to before block (the latch)
        rewriter.replace_op(
            after_terminator,
            BranchOp(before_block, *after_terminator.operands),
        )

        # Replace the while op with the condition arguments (visible by dominance)
        rewriter.replace_matched_op([], cond_args)


class DoWhileLowering(RewritePattern):
    """
    Optimized lowering for `scf.while` when the 'after' region just forwards
    its arguments (i.e., a do-while loop). This avoids inlining the after
    region completely and branches back to the before region directly.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, while_op: WhileOp, rewriter: PatternRewriter, /):
        # Check if the after region is just forwarding arguments
        after_block = while_op.after_region.first_block
        if after_block is None:
            return

        # Check that after region has only a yield op
        if after_block.first_op != after_block.last_op:
            return

        yield_op = after_block.last_op
        if not isinstance(yield_op, YieldOp):
            return

        # Check that yield just forwards the block arguments
        if len(yield_op.operands) != len(after_block.args):
            return

        for operand, arg in zip(yield_op.operands, after_block.args):
            if operand != arg:
                return

        # Proceed with do-while optimization
        condition_block = while_op.parent_block()
        assert condition_block is not None

        # Split the current block before the WhileOp
        continuation_block = condition_block.split_before(while_op)

        # Get the before region
        before_region = while_op.before_region
        before_block = before_region.first_block
        assert before_block is not None

        # Get the terminator BEFORE inlining
        before_last_block = before_region.last_block
        assert before_last_block is not None
        before_terminator = before_last_block.last_op
        assert isinstance(before_terminator, ConditionOp)

        # Inline only the before region
        rewriter.inline_region(
            before_region, BlockInsertPoint.before(continuation_block)
        )

        # Branch to the "before" region from the condition block
        rewriter.insert_op(
            BranchOp(before_block, *while_op.arguments),
            InsertPoint.at_end(condition_block),
        )

        # Replace the condition terminator to loop back to before block
        cond_args = list(before_terminator.args)

        # Replace with conditional branch that loops back or continues
        rewriter.replace_op(
            before_terminator,
            ConditionalBranchOp(
                before_terminator.condition,
                before_block,  # Loop back to before on true
                cond_args,
                continuation_block,  # Exit on false
                (),
            ),
        )

        # Replace the while op with the condition arguments
        rewriter.replace_matched_op([], cond_args)


class ConvertScfToCf(ModulePass):
    """
    Lower `scf.for` and `scf.if` to unstructured control flow. Extended with a lowering for `scf.while`.
    Implementations are direct translations of the mlir versions found at
    https://github.com/llvm/llvm-project/blob/main/mlir/lib/Conversion/SCFToControlFlow/SCFToControlFlow.cpp
    """

    name = "convert-scf-to-cf"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    DoWhileLowering(),
                    WhileLowering(),
                    SwitchLowering(),
                    IfLowering(),
                    ForLowering(),
                ]
            )
        ).rewrite_module(op)
