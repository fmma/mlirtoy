#include "toy/Transform/Affine/AffineFullUnroll.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Pass/Pass.h"

namespace mlir::toy
{

#define GEN_PASS_DEF_AFFINEFULLUNROLLPATTERNREWRITE
#define GEN_PASS_DEF_AFFINEFULLUNROLLTREEWALK
#include "toy/Transform/Affine/Passes.h.inc"

    struct IMPL_TW : impl::AffineFullUnrollTreeWalkBase<IMPL_TW>
    {
        void runOnOperation()
        {
            getOperation()->walk(
                [&](mlir::affine::AffineForOp op)
                {
                    if (failed(mlir::affine::loopUnrollFull(op)))
                    {
                        op.emitError("unrolling failed");
                        signalPassFailure();
                    }
                });
        }
    };

    struct IMPL_PRW : impl::AffineFullUnrollPatternRewriteBase<IMPL_PRW>
    {
        struct AffineFullUnrollPattern : public OpRewritePattern<mlir::affine::AffineForOp>
        {
            AffineFullUnrollPattern(mlir::MLIRContext *context) : OpRewritePattern<mlir::affine::AffineForOp>(context, /*benefit=*/1) {}

            LogicalResult matchAndRewrite(mlir::affine::AffineForOp op, PatternRewriter &rewriter) const override
            {
                return mlir::affine::loopUnrollFull(op);
            }
        };

        void runOnOperation()
        {
            mlir::RewritePatternSet patterns(&getContext());
            patterns.add<AffineFullUnrollPattern>(&getContext());
            (void)applyPatternsGreedily(getOperation(), std::move(patterns));
        }
    };
}
