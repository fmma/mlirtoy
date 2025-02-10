#ifndef LIB_TRANSFORM_AFFINE_AFFINEFULLUNROLL_H_
#define LIB_TRANSFORM_AFFINE_AFFINEFULLUNROLL_H_

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir
{
    namespace toy
    {
        class AffineFullUnrollPassAsTreeWalk : public PassWrapper<AffineFullUnrollPassAsTreeWalk, OperationPass<mlir::func::FuncOp>>
        {
        private:
            void runOnOperation() override;

            StringRef getArgument() const final
            {
                return "affine-full-unroll";
            }

            StringRef getDescription() const final
            {
                return "Fully unroll all affine loops";
            }
        };
        class AffineFullUnrollPassAsPatternRewrite : public PassWrapper<AffineFullUnrollPassAsPatternRewrite, OperationPass<mlir::func::FuncOp>>
        {
        private:
            void runOnOperation() override;

            StringRef getArgument() const final { return "affine-full-unroll-rewrite"; }

            StringRef getDescription() const final
            {
                return "Fully unroll all affine loops using pattern rewrite engine";
            }
        };
    }
}

#endif