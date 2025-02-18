include "ToyDialect.td"
include "mlir/IR/AttrTypeBase.td"

// A base class for all types in this dialect
class Toy_Type<string name, string typeMnemonic> : TypeDef<Toy_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def Int : Toy_Type<"Int", "int"> {
  let summary = "An int";

  let description = [{
    The most basic of types.
  }];
}
