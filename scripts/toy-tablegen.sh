#!/usr/bin/env bash
set -e

repo_root=$(realpath $(dirname $0)/..)
build_root=$repo_root/build
build_dir=$build_root/toy

ninja -C$build_dir ToyTableGenToyDialect ToyTableGenAffineFullUnrollPasses
