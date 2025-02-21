
#include "toy/Dialect/Toy/ToyDialect.h"
#include "toy/Dialect/Toy/ToyTypes.h"
#include "toy/Dialect/Toy/ToyOps.h"

#include "mlir/Dialect/CommonFolders.h"

namespace mlir::toy
{

    OpFoldResult ConstantOp::fold(ConstantOp::FoldAdaptor adaptor)
    {
        return adaptor.getValueAttr();
    }

    OpFoldResult AddOp::fold(AddOp::FoldAdaptor adaptor)
    {
        return constFoldBinaryOp<IntegerAttr, APInt, void>(
            adaptor.getOperands(), [&](APInt a, APInt b)
            { return a + b; });
    }

    OpFoldResult MulOp::fold(MulOp::FoldAdaptor adaptor)
    {
        return constFoldBinaryOp<IntegerAttr, APInt, void>(
            adaptor.getOperands(), [&](APInt a, APInt b)
            { return a * b; });
    }
}
