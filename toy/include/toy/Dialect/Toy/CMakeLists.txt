set(LLVM_TARGET_DEFINITIONS ToyDialect.td)
mlir_tablegen(ToyDialect.h.inc -gen-dialect-decls)
mlir_tablegen(ToyDialect.cpp.inc -gen-dialect-defs)

set(LLVM_TARGET_DEFINITIONS ToyTypes.td)
mlir_tablegen(ToyTypes.h.inc -gen-dialect-decls)
mlir_tablegen(ToyTypes.cpp.inc -gen-dialect-defs)

add_public_tablegen_target(ToyTableGenToyDialect)
