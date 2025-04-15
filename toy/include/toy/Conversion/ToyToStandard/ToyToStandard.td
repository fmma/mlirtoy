include "mlir/Pass/PassBase.td"

def ToyToStandard : Pass<"toy-to-standard"> {
  let summary = "Lower `toy` to standard MLIR dialects.";

  let description = [{
    This pass lowers the `toy` dialect to standard MLIR.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::toy::ToyDialect",
  ];
}
