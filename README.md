# mlirtoy

A personal toy programming language for me to learn LLVM and MLIR.

The language is inspired by Jeremy Kun's MLIR tutorial: https://github.com/j2kun/mlir-tutorial.


## Articles - progress

1.  DONE [Build System (Getting Started)](https://jeremykun.com/2023/08/10/mlir-getting-started/)
2.  DONE [Running and Testing a Lowering](https://jeremykun.com/2023/08/10/mlir-running-and-testing-a-lowering/)
3.  DONE [Writing Our First Pass](https://jeremykun.com/2023/08/10/mlir-writing-our-first-pass/)
4.  [Using Tablegen for Passes](https://jeremykun.com/2023/08/10/mlir-using-tablegen-for-passes/)
5.  [Defining a New Dialect](https://jeremykun.com/2023/08/21/mlir-defining-a-new-dialect/)
6.  [Using Traits](https://jeremykun.com/2023/09/07/mlir-using-traits/)
7.  [Folders and Constant Propagation](https://jeremykun.com/2023/09/11/mlir-folders/)
8.  [Verifiers](https://jeremykun.com/2023/09/13/mlir-verifiers/)
9.  [Canonicalizers and Declarative Rewrite Patterns](https://jeremykun.com/2023/09/20/mlir-canonicalizers-and-declarative-rewrite-patterns/)
10. [Dialect Conversion](https://jeremykun.com/2023/10/23/mlir-dialect-conversion/)
11. [Lowering through LLVM](https://jeremykun.com/2023/11/01/mlir-lowering-through-llvm/)
12. [A Global Optimization and Dataflow Analysis](https://jeremykun.com/2023/11/15/mlir-a-global-optimization-and-dataflow-analysis/)
12. [Defining Patterns with PDLL](https://www.jeremykun.com/2024/08/04/mlir-pdll/)


## Prerequisites

gcc
g++
cmake
ccache
lld
ninja

## Building

You may have to adjust the -j in `scripts/llvm-project-build.sh` if the build crashes.
