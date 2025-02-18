#!/usr/bin/env bash
set -e

repo_root=$(realpath $(dirname $0)/..)
build_root=$repo_root/build
build_dir=$build_root/toy_lang

cd $repo_root/toy_lang

CARGO_TARGET_DIR=$build_dir cargo test
