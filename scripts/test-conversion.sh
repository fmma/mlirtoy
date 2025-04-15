build/toy/tools/toy-opt --toy-to-standard --convert-arith-to-llvm --convert-func-to-llvm toy_lang/tests/add.mlir  | build/llvm/bin/mlir-runner --entry-point-result=i32
