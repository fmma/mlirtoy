set(LLVM_TARGET_DEFINITIONS ToyDialect.td)
mlir_tablegen(ToyDialect.h.inc -gen-dialect-decls)
mlir_tablegen(ToyDialect.cpp.inc -gen-dialect-defs)

set(LLVM_TARGET_DEFINITIONS ToyTypes.td)
mlir_tablegen(ToyTypes.h.inc -gen-typedef-decls)
mlir_tablegen(ToyTypes.cpp.inc -gen-typedef-defs)

set(LLVM_TARGET_DEFINITIONS ToyOps.td)
mlir_tablegen(ToyOps.h.inc -gen-op-decls)
mlir_tablegen(ToyOps.cpp.inc -gen-op-defs)

add_public_tablegen_target(ToyTableGenToyDialect)
