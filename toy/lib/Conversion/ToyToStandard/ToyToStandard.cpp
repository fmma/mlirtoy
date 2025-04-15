#include "toy/Conversion/ToyToStandard/ToyToStandard.h"

#include "toy/Dialect/Toy/ToyTypes.h"
#include "toy/Dialect/Toy/ToyOps.h"
#include "toy/Dialect/Toy/ToyDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"

namespace mlir::toy
{

#define GEN_PASS_DEF_TOYTOSTANDARD
#include "toy/Conversion/ToyToStandard/ToyToStandard.h.inc"

    class ToyToStandardTypeConverter : public TypeConverter
    {
    public:
        ToyToStandardTypeConverter(MLIRContext *ctx)
        {
            addConversion([](Type type)
                          { return type; });
            addConversion([ctx](IntType type) -> Type
                          { return IntegerType::get(ctx, 32, IntegerType::SignednessSemantics::Signless); });
        }
    };

    struct ConvertAdd : public OpConversionPattern<AddOp>
    {
        ConvertAdd(TypeConverter &typeConverter, mlir::MLIRContext *context) : OpConversionPattern<AddOp>(typeConverter, context) {}
        LogicalResult matchAndRewrite(AddOp op, AddOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override
        {
            arith::AddIOp addOp = rewriter.create<arith::AddIOp>(op.getLoc(), adaptor.getLhs(), adaptor.getRhs());
            rewriter.replaceOp(op.getOperation(), addOp);
            return success();
        }
    };

    struct ConvertConst : public OpConversionPattern<ConstantOp>
    {
        ConvertConst(TypeConverter &typeConverter, mlir::MLIRContext *context) : OpConversionPattern<ConstantOp>(typeConverter, context) {}

        LogicalResult matchAndRewrite(ConstantOp op, ConstantOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override
        {
            arith::ConstantIntOp constOp = rewriter.create<arith::ConstantIntOp>(op.getLoc(), adaptor.getValue(), IntegerType::get(this->getContext(), 32, IntegerType::SignednessSemantics::Signless));
            rewriter.replaceOp(op.getOperation(), constOp);
            return success();
        }
    };

    struct ToyToStandard : impl::ToyToStandardBase<ToyToStandard>
    {
        using ToyToStandardBase::ToyToStandardBase;

        void runOnOperation() override
        {
            MLIRContext *context = &getContext();
            auto *module = getOperation();

            ConversionTarget target(*context);
            target.addLegalDialect<arith::ArithDialect>();
            target.addIllegalDialect<ToyDialect>();

            ToyToStandardTypeConverter typeConverter(context);
            RewritePatternSet patterns(context);
            patterns.add<ConvertAdd>(typeConverter, context);
            patterns.add<ConvertConst>(typeConverter, context);

            populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
                patterns, typeConverter);
            target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op)
                                                       { return typeConverter.isSignatureLegal(op.getFunctionType()) &&
                                                                typeConverter.isLegal(&op.getBody()); });

            populateReturnOpTypeConversionPattern(patterns, typeConverter);
            target.addDynamicallyLegalOp<func::ReturnOp>(
                [&](func::ReturnOp op)
                { return typeConverter.isLegal(op); });

            populateCallOpTypeConversionPattern(patterns, typeConverter);
            target.addDynamicallyLegalOp<func::CallOp>(
                [&](func::CallOp op)
                { return typeConverter.isLegal(op); });

            populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
            target.markUnknownOpDynamicallyLegal([&](Operation *op)
                                                 { return isNotBranchOpInterfaceOrReturnLikeOp(op) ||
                                                          isLegalForBranchOpInterfaceTypeConversionPattern(op,
                                                                                                           typeConverter) ||
                                                          isLegalForReturnOpTypeConversionPattern(op, typeConverter); });

            if (failed(applyPartialConversion(module, target, std::move(patterns))))
            {
                signalPassFailure();
            }
        }
    };
}
