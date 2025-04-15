add_mlir_dialect_library(ToyDialect
    ToyDialect.cpp
    ToyOps.cpp

    DEPENDS
    ToyTableGenToyDialect

    LINK_LIBS PUBLIC
)
