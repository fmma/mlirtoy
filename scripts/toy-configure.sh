#!/usr/bin/env bash
set -e

repo_root=$(realpath $(dirname $0)/..)
toy_dir=$repo_root/toy
build_root=$repo_root/build
build_dir=$build_root/toy

llvm_build_dir=$build_root/llvm

mkdir -p $build_dir

# for clang use lld_flag="--ld-path=ld.lld"
lld_flag="--use-ld=lld"

cmake $toy_dir -GNinja -B$build_dir \
      -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_CXX_COMPILER=g++ \
      -DCMAKE_C_COMPILER=gcc \
      -DCMAKE_C_COMPILER_LAUNCHER=ccache \
      -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
      -DCMAKE_EXE_LINKER_FLAGS_INIT="$lld_flag" \
      -DCMAKE_MODULE_LINKER_FLAGS_INIT="$lld_flag" \
      -DCMAKE_SHARED_LINKER_FLAGS_INIT="$lld_flag" \
      -DLLVM_DIR="$llvm_build_dir/lib/cmake/llvm" \
      -DMLIR_DIR="$llvm_build_dir/lib/cmake/mlir" \
      -DBUILD_SHARED_LIBS="OFF" \
      -DBUILD_DEPS="ON"
