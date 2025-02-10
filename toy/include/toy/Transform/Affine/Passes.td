include "mlir/Pass/PassBase.td"

def AffineFullUnrollTreeWalk : Pass<"affine-full-unroll"> {
  let summary = "Fully unroll all affine loops using tree walk";
  let description = [{
    Fully unroll all affine loops using the tree walk method.
  }];
  let dependentDialects = ["mlir::affine::AffineDialect"];
}


def AffineFullUnrollPatternRewrite : Pass<"affine-full-unroll-rewrite"> {
  let summary = "Fully unroll all affine loops using pattern rewrite";
  let description = [{
    Fully unroll all affine loops using the pattern rewrite method.
  }];
  let dependentDialects = ["mlir::affine::AffineDialect"];
}
