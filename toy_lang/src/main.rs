use clap::Parser;
use codegen::{gen_mlir_inlining, Mlir};
use parser::*;

mod types;
mod parser;
mod codegen;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {

    #[arg(short, long)]
    src: String
}

fn main() {
    let args = Args::parse();
    println!("Hello {}!", args.src);
    let (_, program) = nom::Parser::parse(&mut parse_program, &args.src).unwrap();

    let mut mlir = Mlir {
        instructions: vec![]
    };

    let main = program.main.clone();


    print!("{:?}", program);

    gen_mlir_inlining(&program, main, &mut mlir);

    print!("\n\nfunc.func main() {{\n");
    for (i, line) in mlir.instructions.iter().enumerate() {
        print!("  %x{} = {}\n", i, line)
    }
    print!("}}\n\n");
}
