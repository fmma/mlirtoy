#!/usr/bin/env bash
set -e

repo_root=$(realpath $(dirname $0)/..)
llvm_dir=$repo_root/llvm-project/llvm
build_root=$repo_root/build
build_dir=$build_root/llvm

mkdir -p $build_dir

# for clang use lld_flag="--ld-path=ld.lld"
lld_flag="--use-ld=lld"

cmake $llvm_dir -GNinja -B$build_dir \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_COMPILER=g++ \
      -DCMAKE_C_COMPILER=gcc \
      -DCMAKE_C_COMPILER_LAUNCHER=ccache \
      -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
      -DCMAKE_EXE_LINKER_FLAGS_INIT="$lld_flag" \
      -DCMAKE_MODULE_LINKER_FLAGS_INIT="$lld_flag" \
      -DCMAKE_SHARED_LINKER_FLAGS_INIT="$lld_flag" \
      -DLLVM_ENABLE_PROJECTS="mlir" \
      -DLLVM_BUILD_EXAMPLES=OFF \
      -DLLVM_ENABLE_ASSERTIONS=ON \
      -DLLVM_CCACHE_BUILD=ON \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DLLVM_TARGETS_TO_BUILD="host"
