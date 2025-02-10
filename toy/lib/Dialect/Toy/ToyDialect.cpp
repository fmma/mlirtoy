#include <iostream>

#include "toy/Dialect/Toy/ToyDialect.h"
#include "toy/Dialect/Toy/ToyDialect.cpp.inc"

namespace mlir::toy
{
    void ToyDialect::initialize()
    {
        std::cout << "We have initialized the toy dialect" << std::endl;
    }
}
