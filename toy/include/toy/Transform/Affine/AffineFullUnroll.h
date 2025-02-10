#ifndef MLIRTOY_TOY_INCLUDE_TOY_TRANSFORM_AFFINE_AFFINEFULLUNROLL_H_
#define MLIRTOY_TOY_INCLUDE_TOY_TRANSFORM_AFFINE_AFFINEFULLUNROLL_H_

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir
{
    namespace toy
    {
#define GEN_PASS_DECL_AFFINEFULLUNROLLTREEWALK
#define GEN_PASS_DECL_AFFINEFULLUNROLLPATTERNREWRITE
#include "toy/Transform/Affine/Passes.h.inc"
    }
}

#endif