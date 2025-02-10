add_mlir_library(AffineFullUnroll
    AffineFullUnroll.cpp
    
    DEPENDS
    MLIRAffineFullUnrollPasses

    LINK_LIBS PUBLIC
)
