use std::fs::{read_dir, read_to_string};

use toy_lang::codegen::compile_to_mlir;
use toy_lang::parser::parse;

#[test]
fn it_works() -> Result<(), Box<dyn std::error::Error>> {
    println!("{:?}", std::env::current_dir());
    for entry in read_dir("tests")? {
        let entry = entry?;
        let path = entry.path();
        let test = path.file_stem().unwrap().to_str().unwrap();
        if path.is_file() {
            if path.extension().unwrap() == "toy" {
                let src = read_to_string(format!("tests/{}.toy", test))?;
                let expected = read_to_string(format!("tests/{}.mlir", test))?
                    .trim()
                    .to_owned();
                let program = parse(&src);
                let cc = compile_to_mlir(&program);
                let actual = cc.to_mlir_string().trim().to_owned();

                let e_lines = expected.lines().collect::<Vec<&str>>();
                let a_lines = actual.lines().collect::<Vec<&str>>();

                for i in 0..e_lines.len() {
                    if a_lines[i].trim() != e_lines[i].trim() {
                        panic!(
                            "Test {} failed at line {}:\nA: {}\nE: {}\n\nActual output:\n{}",
                            test, i, a_lines[i], e_lines[i], actual
                        )
                    }
                }

                let command_output = std::process::Command::new("../build/toy/tools/toy-opt")
                    .arg(format!("tests/{}.mlir", test))
                    .output()
                    .expect("toy-opt failed");

                if !command_output.status.success() {
                    panic!(
                        "toy-opt failed:\nstdout:\n{}\n\nstderr:\n{}",
                        String::from_utf8_lossy(&command_output.stdout),
                        String::from_utf8_lossy(&command_output.stderr)
                    );
                }
            }
        }
    }
    Ok(())
}
