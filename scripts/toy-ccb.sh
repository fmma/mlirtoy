#!/usr/bin/env bash

set -e

script_dir=$(realpath $(dirname $0))

$script_dir/toy-clean.sh && $script_dir/toy-configure.sh && $script_dir/toy-build.sh
