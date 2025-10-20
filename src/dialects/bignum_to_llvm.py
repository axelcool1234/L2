# src/dialects/bignum_to_llvm.py

from typing import Union

import xdsl.dialects.arith as arith
from xdsl.context import Context
from xdsl.dialects import llvm
from xdsl.dialects.builtin import (
    IntegerAttr,
    ModuleOp,
    i32,
)
from xdsl.dialects.llvm import LLVMPointerType
from xdsl.dialects.scf import WhileOp
from xdsl.ir import Operation
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from .bignum import (
    AddOp,
    BigNumType,
    ConstantOp,
    EqOp,
    FreeOp,
    GteOp,
    GtOp,
    LteOp,
    LtOp,
    NeqOp,
    PrintlnOp,
    PrintOp,
)


class LowerConstantToLLVMPattern(RewritePattern):
    """Lower `bignum.constant` to `llvm.call @l2_bignum_from_i32`."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        if not isinstance(op, ConstantOp):
            return
        # Extract the integer value from the ConstantOp
        int_val: int = op.value.value.data

        const_op = rewriter.insert(
            arith.ConstantOp(
                IntegerAttr(int_val, i32),
                value_type=i32,
            )
        )

        result_type = llvm.LLVMPointerType.opaque()  # GMP big ints are pointers

        # Call the runtime C function
        call = llvm.CallOp(
            "l2_bignum_from_i32",
            const_op.result,
            return_type=result_type,
        )

        # Replace the ConstantOp with the LLVM call
        rewriter.replace_op(op, new_ops=call, new_results=call.results)


class LowerAddToLLVMPattern(RewritePattern):
    """Lower `bignum.add` to `llvm.call @l2_bignum_add`."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        if not isinstance(op, AddOp):
            return
        lhs = op.lhs
        rhs = op.rhs

        # Result is a pointer to a GMP big integer
        result_type = llvm.LLVMPointerType.opaque()

        # Construct the LLVM call op
        call = llvm.CallOp(
            "l2_bignum_add",  # callee (string -> SymbolRefAttr handled inside __init__)
            lhs,  # first operand (SSAValue or Operation)
            rhs,  # second operand (SSAValue or Operation)
            return_type=result_type,  # opaque pointer return
        )

        # Replace the AddOp with the new CallOp
        rewriter.replace_op(op, new_ops=call, new_results=call.results)


class LowerComparisonToLLVMPattern(RewritePattern):
    """Shared pattern for gt/gte/lt/lte lowering."""

    fn_map = {
        GtOp: "l2_bignum_gt",
        GteOp: "l2_bignum_gte",
        LtOp: "l2_bignum_lt",
        LteOp: "l2_bignum_lte",
        EqOp: "l2_bignum_eq",
        NeqOp: "l2_bignum_neq",
    }

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        if not isinstance(op, Union[GtOp, GteOp, LtOp, LteOp, EqOp, NeqOp]):
            return
        callee = self.fn_map[type(op)]
        lhs = op.lhs
        rhs = op.rhs
        ret_type = llvm.i1

        call = llvm.CallOp(
            callee,
            lhs,
            rhs,
            return_type=ret_type,
        )

        rewriter.replace_op(op, new_ops=call, new_results=call.results)


class LowerPrintToLLVMPattern(RewritePattern):
    """Lower `bignum.print` and `bignum.println` to LLVM calls."""

    fn_map = {
        PrintOp: "l2_bignum_print",
        PrintlnOp: "l2_bignum_println",
    }

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        if not isinstance(op, Union[PrintOp, PrintlnOp]):
            return
        callee = self.fn_map[type(op)]
        ptr = op.val

        call = llvm.CallOp(
            callee,
            ptr,
        )
        rewriter.replace_op(op, new_ops=call, new_results=[])


class LowerFreeToLLVMPattern(RewritePattern):
    """Lower `bignum.free` to `llvm.call @l2_bignum_free`."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        if not isinstance(op, FreeOp):
            return
        ptr = op.val
        call = llvm.CallOp("l2_bignum_free", ptr, return_type=llvm.LLVMVoidType())
        rewriter.replace_op(op, new_ops=call, new_results=[])


class LowerBigNumToLLVM(ModulePass):
    """
    Lower the `bignum` dialect into `llvm` dialect calls
    that correspond to GMP-backed runtime functions.
    """

    name = "lower-bignum-to-llvm"

    def apply(self, ctx: Context, op: ModuleOp):
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LowerConstantToLLVMPattern(),
                    LowerAddToLLVMPattern(),
                    LowerComparisonToLLVMPattern(),
                    LowerPrintToLLVMPattern(),
                    LowerFreeToLLVMPattern(),
                    # LowerBigNumTypePattern(),
                ]
            )
        ).rewrite_module(op)

        # Lower `bignum` type
        for single_op in op.walk():
            if isinstance(single_op, WhileOp):
                # WARNING: Setting a read only property!
                # Convert block arguments
                before_block = single_op.before_region.blocks[0]
                for i, arg in enumerate(before_block.args):
                    if isinstance(arg.type, BigNumType):
                        arg._type = LLVMPointerType.opaque()

                after_block = single_op.after_region.blocks[0]
                for i, arg in enumerate(after_block.args):
                    if isinstance(arg.type, BigNumType):
                        arg._type = LLVMPointerType.opaque()

                # Convert result types
                result_types = [
                    LLVMPointerType.opaque() if isinstance(t, BigNumType) else t
                    for t in single_op.result_types
                ]

                # WARNING: Setting a read only property!
                for r, r_type in zip(single_op.results, result_types):
                    r._type = r_type
