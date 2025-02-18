#include <iostream>

#include "toy/Dialect/Toy/ToyDialect.h"
#include "toy/Dialect/Toy/ToyTypes.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/TypeSwitch.h"

#include "toy/Dialect/Toy/ToyDialect.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "toy/Dialect/Toy/ToyTypes.cpp.inc"

namespace mlir::toy
{
    void ToyDialect::initialize()
    {
        addTypes<
#define GET_TYPEDEF_LIST
#include "toy/Dialect/Toy/ToyTypes.cpp.inc"
            >();

        std::cout << "We have initialized the toy dialect" << std::endl;
    }
}
