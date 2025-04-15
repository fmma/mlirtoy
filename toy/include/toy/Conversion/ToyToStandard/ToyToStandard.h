#ifndef MLIRTOY_TOY_INCLUDE_TOY_CONVERSION_TOYTOSTANDARD_TOYTOSTANDARD_H_
#define MLIRTOY_TOY_INCLUDE_TOY_CONVERSION_TOYTOSTANDARD_TOYTOSTANDARD_H_

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace mlir::toy {

#define GEN_PASS_DECL
#include "toy/Conversion/ToyToStandard/ToyToStandard.h.inc"

#define GEN_PASS_REGISTRATION
#include "toy/Conversion/ToyToStandard/ToyToStandard.h.inc"

}

#endif