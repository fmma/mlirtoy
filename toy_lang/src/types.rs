#[derive(Debug, PartialEq, Clone)]
pub(crate) struct ToyVar(pub(crate) String);

#[derive(Debug, Clone, Copy)]
pub(crate) struct ToyConstant(pub(crate) i32);

#[derive(Clone, Debug, Copy)]
pub(crate) enum ToyPrim {
    Dup,
    Drop,
    Swap,
    Rot,
    Over,

    Get,
    Put,

    Add,
    Sub,
    Mul,
    Div,
    Neg,

    Eq,
    Less,
    And,
    Or,
    Not

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
pub struct ToyProgram {
    pub(crate) defs: Vec<ToyDef>,
    pub(crate) main: ToyExpression,
}
