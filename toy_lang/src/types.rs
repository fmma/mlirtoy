#[derive(Debug, PartialEq, Clone)]
pub(crate) struct ToyVar(pub(crate) String);

#[derive(Debug, Clone, Copy)]
pub(crate) struct ToyConstant(pub(crate) i32);

#[derive(Clone, Debug, Copy)]
pub(crate) enum ToyPrim {
    Dup,
    Drop,
    Swap,
    Swap2,
    Rot,
    Over,
    Get,
    Put,
    Mul,
    Add,
    Neg,
    And,
    Or
}

#[derive(Debug, Clone)]
pub(crate) enum ToyExpression {
    Concat {
        left: Box<ToyExpression>,
        right: Box<ToyExpression>,
    },
    Prim(ToyPrim),
    Var(ToyVar),
    Constant(ToyConstant),
}

#[derive(Debug)]
pub(crate) struct ToyDef {
    pub(crate) name: ToyVar, 
    pub(crate) body: ToyExpression
}

#[derive(Debug)]
pub(crate) struct ToyProgram {
    pub(crate) defs: Vec<ToyDef>,
    pub(crate) main: ToyExpression,
}
