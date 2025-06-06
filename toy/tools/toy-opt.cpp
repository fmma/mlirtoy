#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "toy/Transform/Affine/Passes.h"
#include "toy/Dialect/Toy/ToyDialect.h"
#include "toy/Conversion/ToyToStandard/ToyToStandard.h"

int main(int argc, char **argv)
{
    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);
    mlir::registerAllPasses();
    mlir::toy::registerAffinePasses();
    mlir::toy::registerToyToStandardPasses();

    registry.insert<mlir::toy::ToyDialect>();

    auto logicalResult = mlir::MlirOptMain(argc, argv, "Toy language pass driver", registry);

    return mlir::asMainReturnCode(logicalResult);
}
