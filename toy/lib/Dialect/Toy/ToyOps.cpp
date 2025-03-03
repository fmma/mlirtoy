
#include "toy/Dialect/Toy/ToyDialect.h"
#include "toy/Dialect/Toy/ToyTypes.h"
#include "toy/Dialect/Toy/ToyOps.h"

#include "mlir/Dialect/CommonFolders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::toy
{

    static void replaceOpWithRegion(PatternRewriter &rewriter, Operation *op,
                                    Region &region, ValueRange blockArgs = {})
    {
        assert(llvm::hasSingleElement(region) && "expected single-region block");
        Block *block = &region.front();
        Operation *terminator = block->getTerminator();
        ValueRange results = terminator->getOperands();
        rewriter.inlineBlockBefore(block, op, blockArgs);
        rewriter.replaceOp(op, results);
        rewriter.eraseOp(terminator);
    }

    OpFoldResult ConstantOp::fold(ConstantOp::FoldAdaptor adaptor)
    {
        return adaptor.getValueAttr();
    }

    OpFoldResult AddOp::fold(AddOp::FoldAdaptor adaptor)
    {
        return constFoldBinaryOp<IntegerAttr, APInt, void>(
            adaptor.getOperands(), [&](APInt a, APInt b)
            { return a + b; });
    }

    OpFoldResult SubOp::fold(SubOp::FoldAdaptor adaptor)
    {
        return constFoldBinaryOp<IntegerAttr, APInt, void>(
            adaptor.getOperands(), [&](APInt a, APInt b)
            { return a - b; });
    }

    OpFoldResult MulOp::fold(MulOp::FoldAdaptor adaptor)
    {
        return constFoldBinaryOp<IntegerAttr, APInt, void>(
            adaptor.getOperands(), [&](APInt a, APInt b)
            { return a * b; });
    }

    OpFoldResult DivOp::fold(DivOp::FoldAdaptor adaptor)
    {
        return constFoldBinaryOp<IntegerAttr, APInt, void>(
            adaptor.getOperands(), [&](APInt a, APInt b)
            { return a.sdiv(b); });
    }

    OpFoldResult NegOp::fold(NegOp::FoldAdaptor adaptor)
    {
        return constFoldUnaryOp<IntegerAttr, APInt, void>(
            adaptor.getOperands(), [&](APInt a)
            { return -a; });
    }

    OpFoldResult EqOp::fold(EqOp::FoldAdaptor adaptor)
    {
        return constFoldBinaryOp<IntegerAttr, APInt, void>(
            adaptor.getOperands(), [&](APInt a, APInt b)
            { return a == b ? APInt(32, 1, true) : APInt(32, 0, true); });
    }

    OpFoldResult LessOp::fold(LessOp::FoldAdaptor adaptor)
    {
        return constFoldBinaryOp<IntegerAttr, APInt, void>(
            adaptor.getOperands(), [&](APInt a, APInt b)
            { return a.slt(b) ? APInt(32, 1, true) : APInt(32, 0, true); });
    }

    OpFoldResult AndOp::fold(AndOp::FoldAdaptor adaptor)
    {
        return constFoldBinaryOp<IntegerAttr, APInt, void>(
            adaptor.getOperands(), [&](APInt a, APInt b)
            { return a & b; });
    }

    OpFoldResult OrOp::fold(OrOp::FoldAdaptor adaptor)
    {
        return constFoldBinaryOp<IntegerAttr, APInt, void>(
            adaptor.getOperands(), [&](APInt a, APInt b)
            { return a | b; });
    }

    OpFoldResult NotOp::fold(NotOp::FoldAdaptor adaptor)
    {
        return constFoldUnaryOp<IntegerAttr, APInt, void>(
            adaptor.getOperands(), [&](APInt a)
            { return !a ? APInt(32, 1, true) : APInt(32, 0, true); });
    }

    struct ElimIf : public OpRewritePattern<IfOp>
    {
        using OpRewritePattern<IfOp>::OpRewritePattern;

        LogicalResult matchAndRewrite(IfOp op, PatternRewriter &rewriter) const override
        {
            BoolAttr condition;
            if (!matchPattern(op.getCondition(), m_Constant(&condition)))
                return failure();

            if (condition.getValue())
                replaceOpWithRegion(rewriter, op, op.getThenRegion());
            else
                replaceOpWithRegion(rewriter, op, op.getElseRegion());

            return success();
        }
    };

    void IfOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context)
    {
        results.add<ElimIf>(context);
    }
}
