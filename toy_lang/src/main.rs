use clap::Parser;
use toy_lang::codegen::compile_to_mlir;
use toy_lang::parser::parse;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
    src_file: clap_stdin::FileOrStdin,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let program = parse(&args.src_file.contents()?);

    let cc = compile_to_mlir(&program);

    print!("{}", cc.to_mlir_string());
    Ok(())
}
