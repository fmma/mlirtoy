
include "ToyTypes.td"

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"


class BinOp<string mnemonic> : Op<Toy_Dialect, mnemonic, [Pure, ElementwiseMappable, SameOperandsAndResultType]> {
  let arguments = (ins Int:$lhs, Int:$rhs);
  let results = (outs Int:$output);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($output)";
}

class UnOp<string mnemonic> : Op<Toy_Dialect, mnemonic, [Pure, ElementwiseMappable, SameOperandsAndResultType]> {
  let arguments = (ins Int:$input);
  let results = (outs Int:$output);
  let assemblyFormat = "$input attr-dict `:` type($output)";
}

def AddOp : BinOp<"add"> {
  let summary = "Addition.";
  let hasFolder = 1;
}

def MulOp : BinOp<"mul"> {
  let summary = "Multiplication.";
  let hasFolder = 1;
}

def NegOp : UnOp<"neg"> {
  let summary = "Negation.";
}

def FromIntOp : Op<Toy_Dialect, "from_i32", [Pure]> {
  let summary = "Convert i32 to toy int";
  let arguments = (ins AnyInteger:$input);
  let results = (outs Int:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
}

def ConstantOp : Op<Toy_Dialect, "const", [Pure, ConstantLike]> {
  let summary = "Constants";
  let arguments = (ins I32Attr:$value);
  let results = (outs Int:$output);
  let assemblyFormat = "$value attr-dict `:` type($output)";
  let hasFolder = 1;
}
