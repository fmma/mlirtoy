#include "toy/Conversion/ToyToStandard/ToyToStandard.h"

#include "toy/Dialect/Toy/ToyDialect.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::toy
{

#define GEN_PASS_DEF_TOYTOSTANDARD
#include "toy/Conversion/ToyToStandard/ToyToStandard.h.inc"

  struct ToyToStandard : impl::ToyToStandardBase<ToyToStandard>
  {
    using ToyToStandardBase::ToyToStandardBase;

    void runOnOperation() override
    {
      MLIRContext *context = &getContext();
      auto *module = getOperation();
      // TODO: implement pass
    }
  };
}
