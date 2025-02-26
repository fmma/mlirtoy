
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

def IfOp : Op<Toy_Dialect, "if", []> {
  let summary = "if-then-else operation";
  let arguments = (ins Int:$condition);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$thenRegion,
                        SizedRegion<1>:$elseRegion);
                        
  let assemblyFormat = "$condition attr-dict `:` type($results) $thenRegion $elseRegion";
}

def YieldOp : Op<Toy_Dialect, "yield", [Pure, Terminator]> {
  
  let summary = "if-then-else operation";
  let arguments = (ins Variadic<Int>:$results);
                        
  let assemblyFormat = "$results attr-dict";
}

def AddOp : BinOp<"add"> {
  let summary = "Addition.";
  let hasFolder = 1;
}

def SubOp : BinOp<"sub"> {
  let summary = "Subtraction.";
  let hasFolder = 1;
}

def MulOp : BinOp<"mul"> {
  let summary = "Multiplication.";
  let hasFolder = 1;
}

def DivOp : BinOp<"div"> {
  let summary = "Division (integer).";
  let hasFolder = 1;
}

def EqOp : BinOp<"eq"> {
  let summary = "Equality check";
  let hasFolder = 1;
}

def LessOp : BinOp<"less"> {
  let summary = "Less than check (<)";
  let hasFolder = 1;
}

def AndOp : BinOp<"and"> {
  let summary = "Logical and";
  let hasFolder = 1;
}

def OrOp : BinOp<"or"> {
  let summary = "Logical or.";
  let hasFolder = 1;
}

def NegOp : UnOp<"neg"> {
  let summary = "Negation.";
  let hasFolder = 1;
}

def NotOp : UnOp<"not"> {
  let summary = "Logical negation.";
  let hasFolder = 1;
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

def GetOp : Op<Toy_Dialect, "get", []> {
  let summary = "Get a value from stdin";
  let results = (outs Int:$output);
  let assemblyFormat = "attr-dict `:` type($output)";
}

def PutOp : Op<Toy_Dialect, "put", []> {
  let summary = "Put a value to stdout";
  let arguments = (ins Int:$input);
  let assemblyFormat = "$input attr-dict";
}
