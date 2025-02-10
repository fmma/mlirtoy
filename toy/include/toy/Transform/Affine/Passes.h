#ifndef MLIRTOY_TOY_INCLUDE_TOY_TRANSFORM_AFFINE_PASSES_H_
#define MLIRTOY_TOY_INCLUDE_TOY_TRANSFORM_AFFINE_PASSES_H_

#include "toy/Transform/Affine/AffineFullUnroll.h"

namespace mlir::toy {
    #define GEN_PASS_REGISTRATION
    #include "toy/Transform/Affine/Passes.h.inc"
}

#endif