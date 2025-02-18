
include "ToyTypes.td"

include "mlir/IR/OpBase.td"

def Toy_AddOp : Op<Toy_Dialect, "add"> {
  let summary = "Addition.";
  let arguments = (ins Int:$lhs, Int:$rhs);
  let results = (outs Int:$output);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($output)";
}
