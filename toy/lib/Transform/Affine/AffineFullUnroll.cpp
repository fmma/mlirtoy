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

#define GEN_PASS_DEF_AFFINEFULLUNROLLPATTERNREWRITE
#define GEN_PASS_DEF_AFFINEFULLUNROLLTREEWALK
#include "toy/Transform/Affine/Passes.h.inc"

        using mlir::affine::AffineForOp;
        using mlir::affine::loopUnrollFull;

        struct AffineFullUnrollTreeWalk : impl::AffineFullUnrollTreeWalkBase<AffineFullUnrollTreeWalk>
        {
            using AffineFullUnrollTreeWalkBase::AffineFullUnrollTreeWalkBase;

            void runOnOperation()
            {
                getOperation()->walk(
                    [&](AffineForOp op)
                    {
                        if (failed(loopUnrollFull(op)))
                        {
                            op.emitError("unrolling failed");
                            signalPassFailure();
                        }
                    });
            }
        };

        struct AffineFullUnrollPattern : public OpRewritePattern<AffineForOp>
        {
            AffineFullUnrollPattern(mlir::MLIRContext *context)
                : OpRewritePattern<AffineForOp>(context, /*benefit=*/1) {}

            LogicalResult matchAndRewrite(AffineForOp op, PatternRewriter &rewriter) const override
            {
                return loopUnrollFull(op);
            }
        };

        struct AffineFullUnrollPatternRewrite : impl::AffineFullUnrollPatternRewriteBase<AffineFullUnrollPatternRewrite>
        {
            using AffineFullUnrollPatternRewriteBase::AffineFullUnrollPatternRewriteBase;

            void runOnOperation()
            {
                mlir::RewritePatternSet patterns(&getContext());
                patterns.add<AffineFullUnrollPattern>(&getContext());
                (void)applyPatternsGreedily(getOperation(), std::move(patterns));
            }
        };
    }
}
