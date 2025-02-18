use clap::Parser;
use codegen::compile_to_mlir;
use parser::*;

mod codegen;
mod parser;
mod types;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    src: String,
}

fn main() {
    let args = Args::parse();
    println!("Hello {}!", args.src);
    let (_, program) = nom::Parser::parse(&mut parse_program, &args.src).unwrap();

    let cc = compile_to_mlir(&program);

    print!("{}", cc.to_mlir_string());
}

#[cfg(test)]
mod tests {
    use std::fs::{read_dir, read_to_string};

    use crate::{compile_to_mlir, parser::parse_program};

    #[test]
    fn it_works() -> Result<(), Box<dyn std::error::Error>> {
        for entry in read_dir("test")? {
            let entry = entry?;
            let path = entry.path();
            let test = path.file_stem().unwrap().to_str().unwrap();
            if path.is_file() {
                if path.extension().unwrap() == "toy" {
                    let src = read_to_string(format!("test/{}.toy", test))?;
                    let expected = read_to_string(format!("test/{}.mlir", test))?;
                    let (_, program) = nom::Parser::parse(&mut parse_program, &src).unwrap();
                    let cc = compile_to_mlir(&program);
                    assert_eq!(cc.to_mlir_string().trim(), expected.trim());
                }
            }
        }
        Ok(())
    }
}
