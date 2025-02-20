use clap::Parser;
use toy_lang::codegen::compile_to_mlir;
use toy_lang::parser::parse;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    src: String,
}

fn main() {
    let args = Args::parse();
    println!("Hello {}!", args.src);
    let program = parse(&args.src);

    let cc = compile_to_mlir(&program);

    print!("{}", cc.to_mlir_string());
}
