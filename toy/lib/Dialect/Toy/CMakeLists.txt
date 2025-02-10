add_mlir_dialect_library(ToyDialect
    ToyDialect.cpp
    
    DEPENDS
    ToyTableGenToyDialect

    LINK_LIBS PUBLIC
)
