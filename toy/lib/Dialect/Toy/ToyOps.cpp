
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

    OpFoldResult SubOp::fold(SubOp::FoldAdaptor adaptor)
    {
        return constFoldBinaryOp<IntegerAttr, APInt, void>(
            adaptor.getOperands(), [&](APInt a, APInt b)
            { return a - b; });
    }

    OpFoldResult MulOp::fold(MulOp::FoldAdaptor adaptor)
    {
        return constFoldBinaryOp<IntegerAttr, APInt, void>(
            adaptor.getOperands(), [&](APInt a, APInt b)
            { return a * b; });
    }

    OpFoldResult DivOp::fold(DivOp::FoldAdaptor adaptor)
    {
        return constFoldBinaryOp<IntegerAttr, APInt, void>(
            adaptor.getOperands(), [&](APInt a, APInt b)
            { return a.sdiv(b); });
    }

    OpFoldResult NegOp::fold(NegOp::FoldAdaptor adaptor)
    {
        return constFoldUnaryOp<IntegerAttr, APInt, void>(
            adaptor.getOperands(), [&](APInt a)
            { return -a; });
    }

    OpFoldResult EqOp::fold(EqOp::FoldAdaptor adaptor)
    {
        return constFoldBinaryOp<IntegerAttr, APInt, void>(
            adaptor.getOperands(), [&](APInt a, APInt b)
            { return a == b ? APInt(32, 1, true) : APInt(32, 0, true); });
    }

    OpFoldResult LessOp::fold(LessOp::FoldAdaptor adaptor)
    {
        return constFoldBinaryOp<IntegerAttr, APInt, void>(
            adaptor.getOperands(), [&](APInt a, APInt b)
            { return a.slt(b) ? APInt(32, 1, true) : APInt(32, 0, true); });
    }

    OpFoldResult AndOp::fold(AndOp::FoldAdaptor adaptor)
    {
        return constFoldBinaryOp<IntegerAttr, APInt, void>(
            adaptor.getOperands(), [&](APInt a, APInt b)
            { return a & b; });
    }

    OpFoldResult OrOp::fold(OrOp::FoldAdaptor adaptor)
    {
        return constFoldBinaryOp<IntegerAttr, APInt, void>(
            adaptor.getOperands(), [&](APInt a, APInt b)
            { return a | b; });
    }

    OpFoldResult NotOp::fold(NotOp::FoldAdaptor adaptor)
    {
        return constFoldUnaryOp<IntegerAttr, APInt, void>(
            adaptor.getOperands(), [&](APInt a)
            { return !a ? APInt(32, 1, true) : APInt(32, 0, true); });
    }
}
