include "mlir/IR/DialectBase.td"

def Toy_Dialect : Dialect {
    let name = "toy";
    let summary = "Dialect for the toy language";
    let description = [{
        The toy dialect defines types and ops for the TOY language.
    }];

    let cppNamespace = "::mlir::toy";
}
