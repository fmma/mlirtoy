use crate::types::*;

#[derive(Clone)]
pub(crate) struct MlirFunc {
    pub(crate) name: String,
    pub(crate) n_args: usize,
    pub(crate) n_vars: usize,
    pub(crate) stack: Vec<String>,
    pub(crate) instructions: Vec<(usize, String)>,
}

pub struct MlirProgram {
    types: Vec<(String, usize, usize)>,
    compiled_defs: Vec<MlirFunc>,
}

pub fn compile_to_mlir(program: &ToyProgram) -> MlirProgram {
    let mut mlir_prog = MlirProgram {
        compiled_defs: vec![],
        types: program
            .defs
            .iter()
            .map(|x| x.t.map(|(t1, t2)| (x.name.0.clone(), t1, t2)))
            .flatten()
            .collect::<Vec<_>>(),
    };
    for def in &program.defs {
        let name = &def.name.0;
        let toy_expression = &def.body;

        compile_def_to_mlir(&mut mlir_prog, name, toy_expression);
    }

    compile_def_to_mlir(&mut mlir_prog, &"main".to_owned(), &program.main);

    return mlir_prog;
}

fn compile_def_to_mlir(mlir_prog: &mut MlirProgram, name: &String, toy_expr: &ToyExpression) {
    let mut mlir = MlirFunc {
        name: name.clone(),
        n_args: 0,
        instructions: vec![],
        n_vars: 0,
        stack: vec![],
    };
    compile_expr_to_mlir(mlir_prog, toy_expr.clone(), &mut mlir);
    mlir_prog.compiled_defs.push(mlir);
}

fn compile_expr_to_mlir(mlir_prog: &MlirProgram, toy_expr: ToyExpression, mlir: &mut MlirFunc) {
    match toy_expr {
        ToyExpression::Concat { left, right } => {
            compile_expr_to_mlir(mlir_prog, *left, mlir);
            compile_expr_to_mlir(mlir_prog, *right, mlir);
        }
        ToyExpression::Branch { left, right } => {
            let b = mlir.pop();

            let mut mlir_l = mlir.clone();
            mlir_l.instructions.clear();
            compile_expr_to_mlir(mlir_prog, *left.clone(), &mut mlir_l);

            let mut mlir_r = mlir.clone();
            mlir_r.instructions.clear();
            mlir_r.n_vars = mlir_l.n_vars;
            compile_expr_to_mlir(mlir_prog, *right.clone(), &mut mlir_r);

            assert_eq!(mlir_l.stack.len(), mlir_r.stack.len());
            assert_eq!(mlir_l.n_args, mlir_r.n_args);

            let n = mlir_l.stack.len();

            let k = mlir_l
                .stack
                .iter()
                .rev()
                .zip(mlir_r.stack.iter().rev())
                .position(|(x, y)| x == y)
                .unwrap_or(n);

            let mut mlir_l = mlir.clone();
            mlir_l.n_vars += k;
            mlir_l.instructions.clear();
            compile_expr_to_mlir(mlir_prog, *left, &mut mlir_l);

            let mut mlir_r = mlir.clone();
            mlir_r.instructions.clear();
            mlir_r.n_vars = mlir_l.n_vars;
            compile_expr_to_mlir(mlir_prog, *right, &mut mlir_r);

            for _ in 0..mlir_l.n_args {
                mlir.pop();
            }

            mlir.emit(
                format!(
                    "toy.if {} : {} {{",
                    b,
                    (0..k)
                        .map(|_x| "!toy.int".to_owned())
                        .collect::<Vec<String>>()
                        .join(", ")
                ),
                k,
            );

            mlir.n_vars = mlir_r.n_vars;
            mlir.n_args = mlir_l.n_args;

            for (o, i) in mlir_l.instructions {
                mlir.instructions.push((o, i));
            }

            let r_l = mlir_l
                .stack
                .iter()
                .rev()
                .take(k)
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(", ");

            mlir.emit(format!("toy.yield {}", r_l), 0);
            mlir.emit(format!("}} {{"), 0);

            for (o, i) in mlir_r.instructions {
                mlir.instructions.push((o, i));
            }

            let r_r = mlir_r
                .stack
                .iter()
                .rev()
                .take(k)
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            mlir.emit(format!("toy.yield {}", r_r), 0);
            mlir.emit(format!("}}"), 0);
        }
        ToyExpression::Prim(toy_prim) => match toy_prim {
            ToyPrim::Dup => {
                let x = mlir.pop();
                mlir.push(x.clone());
                mlir.push(x);
            }
            ToyPrim::Drop => {
                mlir.pop();
            }
            ToyPrim::Swap => {
                let x = mlir.pop();
                let y = mlir.pop();
                mlir.push(x);
                mlir.push(y);
            }
            ToyPrim::Rot => {
                let x = mlir.pop();
                let y = mlir.pop();
                let z = mlir.pop();
                mlir.push(x);
                mlir.push(z);
                mlir.push(y);
            }
            ToyPrim::Over => {
                let x = mlir.pop();
                let y = mlir.pop();
                mlir.push(x.clone());
                mlir.push(y);
                mlir.push(x);
            }
            ToyPrim::Get => {
                mlir.emit("toy.get : !toy.int".to_owned(), 1);
            }
            ToyPrim::Put => {
                let x = mlir.pop();
                mlir.emit(format!("toy.put {}", x), 0);
            }
            ToyPrim::Add => {
                let x = mlir.pop();
                let y = mlir.pop();
                mlir.emit(format!("toy.add {}, {} : !toy.int", x, y), 1);
            }
            ToyPrim::Mul => {
                let x = mlir.pop();
                let y = mlir.pop();
                mlir.emit(format!("toy.mul {}, {} : !toy.int", x, y), 1);
            }
            ToyPrim::Neg => {
                let x = mlir.pop();
                mlir.emit(format!("toy.neg {} : !toy.int", x), 1);
            }
            ToyPrim::And => {
                let x = mlir.pop();
                let y = mlir.pop();
                mlir.emit(format!("toy.and {}, {} : !toy.int", x, y), 1);
            }
            ToyPrim::Or => {
                let x = mlir.pop();
                let y = mlir.pop();
                mlir.emit(format!("toy.or {}, {} : !toy.int", x, y), 1);
            }
            ToyPrim::Sub => {
                let x = mlir.pop();
                let y = mlir.pop();
                mlir.emit(format!("toy.sub {}, {} : !toy.int", x, y), 1);
            }
            ToyPrim::Div => {
                let x = mlir.pop();
                let y = mlir.pop();
                mlir.emit(format!("toy.div {}, {} : !toy.int", x, y), 1);
            }
            ToyPrim::Eq => {
                let x = mlir.pop();
                let y = mlir.pop();
                mlir.emit(format!("toy.eq {}, {} : !toy.int", x, y), 1);
            }
            ToyPrim::Less => {
                let x = mlir.pop();
                let y = mlir.pop();
                mlir.emit(format!("toy.less {}, {} : !toy.int", x, y), 1);
            }
            ToyPrim::Not => {
                let x = mlir.pop();
                mlir.emit(format!("toy.not {} : !toy.int", x), 1);
            }
        },
        ToyExpression::Constant(toy_constant) => {
            mlir.emit(format!("toy.const {} : !toy.int", toy_constant.0), 1);
        }
        ToyExpression::Var(toy_var) => match mlir_prog
            .compiled_defs
            .iter()
            .find(|mlir| mlir.name == toy_var.0)
        {
            Some(mlir_func) => {
                let args = (0..mlir_func.n_args)
                    .map(|_x| format!("{}", mlir.pop()))
                    .collect::<Vec<String>>()
                    .join(", ");
                mlir.emit(
                    format!(
                        "func.call @{} ({}) : {}",
                        toy_var.0,
                        args,
                        mlir_func.clone().print_type()
                    ),
                    mlir_func.stack.len(),
                );
            }
            None => match mlir_prog.types.iter().find(|x| x.0 == toy_var.0) {
                Some((name, n_args, n_results)) => {
                    let args = (0..*n_args)
                        .map(|_x| format!("{}", mlir.pop()))
                        .collect::<Vec<String>>()
                        .join(", ");
                    mlir.emit(
                        format!(
                            "func.call @{} ({}) : {}",
                            name,
                            args,
                            print_type(*n_args, *n_results)
                        ),
                        *n_results,
                    );
                }
                None => panic!("Unbound var {}", toy_var.0),
            },
        },
    }
}

fn print_type(n_args: usize, n_outs: usize) -> String {
    let in_type = format!(
        "({})",
        (0..n_args)
            .map(|_x| "!toy.int".to_owned())
            .collect::<Vec<String>>()
            .join(", ")
    );

    let out_type = format!(
        " -> ({})",
        (0..n_outs)
            .map(|_x| "!toy.int".to_owned())
            .collect::<Vec<String>>()
            .join(", ")
    );

    return format!("{}{}", in_type, out_type);
}

impl MlirFunc {
    fn pop(&mut self) -> String {
        if self.stack.len() > 0 {
            return self.stack.pop().unwrap();
        } else {
            let next_arg = self.n_args;
            self.n_args += 1;
            return format!("%arg_{}", next_arg);
        }
    }

    fn push(&mut self, x: String) {
        return self.stack.push(x);
    }

    fn emit(&mut self, instruction: String, n_out: usize) {
        for _ in 0..n_out {
            self.push(format!("%x{}", self.n_vars));
            self.n_vars += 1;
        }
        self.instructions.push((n_out, instruction));
    }

    fn print_type(self) -> String {
        return print_type(self.n_args, self.stack.len());
    }

    fn to_mlir_string(self) -> String {
        let mut lines: Vec<String> = Vec::new();
        let mut i = 0;

        let args = (0..self.n_args)
            .map(|x| format!("%arg_{} : !toy.int", x))
            .collect::<Vec<String>>()
            .join(", ");

        let out_type = format!(
            "({})",
            (0..self.stack.len())
                .map(|_x| "!toy.int".to_owned())
                .collect::<Vec<String>>()
                .join(", ")
        );

        lines.push(format!(
            "func.func @{}({}) -> {} {{",
            self.name, args, out_type
        ));

        for (n_outs, line) in self.instructions.iter() {
            if *n_outs == 0 {
                lines.push(format!("  {}", line));
            } else {
                let args = (0..*n_outs)
                    .map(|x| format!("%x{}", i + x))
                    .collect::<Vec<String>>()
                    .join(", ");
                lines.push(format!("  {} = {}", args, line));
            }
            i += n_outs;
        }

        let ret_expr = match self.stack.len() {
            0 => "".to_owned(),
            1 => self.stack.first().unwrap().to_string(),
            _ => format!(
                "{}",
                self.stack
                    .iter()
                    //                    .rev()
                    .map(|x| x.to_string())
                    .collect::<Vec<String>>()
                    .join(", ")
            ),
        };

        let ret_type = match self.stack.len() {
            0 => "".to_owned(),
            1 => format!(" : !toy.int"),
            _ => format!(
                " : {}",
                self.stack
                    .iter()
                    .map(|_| "!toy.int".to_owned())
                    .collect::<Vec<String>>()
                    .join(", ")
            ),
        };

        lines.push(format!("  return {}{}", ret_expr, ret_type));
        lines.push(format!("}}\n"));

        return lines.join("\n");
    }
}

impl MlirProgram {
    pub fn to_mlir_string(self) -> String {
        let mut funcs = vec![];
        for mlir in self.compiled_defs.iter() {
            funcs.push(mlir.clone().to_mlir_string());
        }
        return funcs.join("");
    }
}
