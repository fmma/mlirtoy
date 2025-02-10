#!/usr/bin/env bash

set -e

script_dir=$(realpath $(dirname $0))

$script_dir/llvm-project-clean.sh && $script_dir/llvm-project-configure.sh && $script_dir/llvm-project-build.sh
