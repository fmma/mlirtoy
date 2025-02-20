#ifndef MLIRTOY_TOY_INCLUDE_TOY_DIALECT_TOY_TOYOPS_H_
#define MLIRTOY_TOY_INCLUDE_TOY_DIALECT_TOY_TOYOPS_H_

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

#define GET_OP_CLASSES
#include "toy/Dialect/Toy/ToyOps.h.inc"

#endif
