use std::fs::{read_dir, read_to_string};
use std::io::Write;
use std::process::{Command, Stdio};

use toy_lang::codegen::compile_to_mlir;
use toy_lang::parser::parse;

#[test]
fn run_file_tests() -> Result<(), Box<dyn std::error::Error>> {
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
                let actual = call_toy_opt(compile_toy_to_mlir(src))?;
                
                let expected_lines = expected.lines().collect::<Vec<&str>>();
                let actual_lines = actual.lines().collect::<Vec<&str>>();

                for i in 0..expected_lines.len() {
                    if actual_lines[i].trim() != expected_lines[i].trim() {
                        panic!(
                            "Test {} failed at line {}:\nA: {}\nE: {}\n\nActual output:\n{}",
                            test, i, actual_lines[i], expected_lines[i], actual
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

fn compile_toy_to_mlir(src: String) -> String {
    let program = parse(&src);
    let cc = compile_to_mlir(&program);
    cc.to_mlir_string().trim().to_owned()
}

fn call_toy_opt(mlir_src: String) -> Result<String, Box<dyn std::error::Error>> {
    let mut child =Command::new("../build/toy/tools/toy-opt")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    let mut stdin = child.stdin.take().expect("Could not take stdin");
    stdin.write_all(mlir_src.as_bytes())?;
    drop(stdin);
    let output = child.wait_with_output()?;
    let actual = String::from_utf8(output.stdout)?;
    Ok(actual)
}
