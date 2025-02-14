use crate::types::*;

pub(crate) struct Mlir {
    pub(crate) instructions: Vec<String>,
}

pub(crate) fn gen_mlir_inlining(program: &ToyProgram, e: ToyExpression, mlir: &mut Mlir) {
    match e {
        ToyExpression::Concat { left, right } => {
            gen_mlir_inlining(program, *left, mlir);
            gen_mlir_inlining(program, *right, mlir);
        }
        ToyExpression::Prim(toy_prim) => match toy_prim {
            ToyPrim::Dup => match mlir.instructions.last() {
                Some(i) => {
                    mlir.instructions.push(i.clone());
                }
                None => todo!(),
            },
            ToyPrim::Pop => {
                mlir.instructions.pop();
            }
            ToyPrim::Mul => {
                mlir.instructions
                    .push(format!("toy.mul %x{} %x{} : i32", mlir.instructions.len() - 2, mlir.instructions.len() - 1));
            }
        },
        ToyExpression::Var(toy_var) => match program.defs.iter().find(|d| d.name == toy_var) {
            Some(d) => gen_mlir_inlining(program, d.body.clone(), mlir),
            None => todo!(),
        },
        ToyExpression::Constant(toy_constant) => mlir
            .instructions
            .push(format!("toy.constant {} : i32", toy_constant.0)),
    }
}
