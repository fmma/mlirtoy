#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "toy/Transform/Affine/AffineFullUnroll.h"


int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllPasses();
  mlir::PassRegistration<mlir::toy::AffineFullUnrollPass>();
  mlir::PassRegistration<mlir::toy::AffineFullUnrollPassAsPatternRewrite>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Toy language pass driver", registry));
}
