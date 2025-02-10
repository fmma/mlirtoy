include "mlir/Pass/PassBase.td"

def AffineFullUnrollTreeWalk : Pass<"affine-full-unroll-tree-walk"> {
  let summary = "Fully unroll all affine loops using tree walk";
  let description = [{
    Fully unroll all affine loops using the tree walk method.
  }];
  let dependentDialects = ["mlir::affine::AffineDialect"];
}
