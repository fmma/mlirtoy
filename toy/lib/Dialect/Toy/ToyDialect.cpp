#include <iostream>

#include "toy/Dialect/Toy/ToyDialect.h"
#include "toy/Dialect/Toy/ToyTypes.h"
#include "toy/Dialect/Toy/ToyOps.h"

#include "toy/Dialect/Toy/ToyDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "toy/Dialect/Toy/ToyTypes.cpp.inc"

#define GET_OP_CLASSES
#include "toy/Dialect/Toy/ToyOps.cpp.inc"

namespace mlir::toy
{
    void ToyDialect::initialize()
    {
        addTypes<
#define GET_TYPEDEF_LIST
#include "toy/Dialect/Toy/ToyTypes.cpp.inc"
            >();

        addOperations<
#define GET_OP_LIST
#include "toy/Dialect/Toy/ToyOps.cpp.inc"
            >();

        std::cout << "We have initialized the toy dialect" << std::endl;
    }
}
