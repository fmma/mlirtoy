#include "toy/Transform/Affine/AffineFullUnroll.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Pass/Pass.h"

namespace mlir
{
    namespace toy
    {
        using mlir::affine::AffineForOp;
        using mlir::affine::loopUnrollFull;

        void AffineFullUnrollPassAsTreeWalk::runOnOperation()
        {
            getOperation().walk(
                [&](AffineForOp op)
                {
                    if (failed(loopUnrollFull(op)))
                    {
                        op.emitError("unrolling failed");
                        signalPassFailure();
                    }
                });
        }

        struct AffineFullUnrollPattern : public OpRewritePattern<AffineForOp>
        {
            AffineFullUnrollPattern(mlir::MLIRContext *context)
                : OpRewritePattern<AffineForOp>(context, /*benefit=*/1) {}

            LogicalResult matchAndRewrite(AffineForOp op, PatternRewriter &rewriter) const override
            {
                return loopUnrollFull(op);
            }
        };

        void AffineFullUnrollPassAsPatternRewrite::runOnOperation()
        {
            mlir::RewritePatternSet patterns(&getContext());
            patterns.add<AffineFullUnrollPattern>(&getContext());
            (void)applyPatternsGreedily(getOperation(), std::move(patterns));
        }
    }
}
