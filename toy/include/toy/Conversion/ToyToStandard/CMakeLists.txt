set(LLVM_TARGET_DEFINITIONS ToyToStandard.td)
mlir_tablegen(ToyToStandard.h.inc -gen-pass-decls -name ToyToStandard)
add_public_tablegen_target(ToyToStandardPassIncGen)
