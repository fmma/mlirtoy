add_mlir_library(AffineFullUnroll
    AffineFullUnroll.cpp
    
    DEPENDS
    ToyTableGenAffineFullUnrollPasses

    LINK_LIBS PUBLIC
)
