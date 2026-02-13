#!/bin/bash

crates=(
    hodu_macro_utils

    hodu_cpu_kernels
    hodu_cuda_kernels
    hodu_metal_kernels

    hodu_core
    hodu_nn/macros
    hodu_nn
    hodu_datasets/macros
    hodu_datasets

    hodu_internal
)

if [ -n "$(git status --porcelain)" ]; then
    echo "You have local changes!"
    exit 1
fi

pushd crates

for crate in "${crates[@]}"
do
  echo "Publishing ${crate}"
  cp ../LICENSE "$crate"
  pushd "$crate"
  git add LICENSE
  cargo publish --no-verify --allow-dirty
  popd
  sleep 20
done

popd

echo "Publishing hodu"
cp LICENSE .
git add LICENSE
cargo publish --no-verify --allow-dirty

echo "Cleaning local state"
git reset HEAD --hard
