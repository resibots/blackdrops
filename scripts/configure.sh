#!/bin/bash
cwd=$(pwd)

# go to limbo
cd limbo
mkdir -p exp
cd exp
ln -s ../../ blackdrops
# go back to original directory
cd ../..

# save current directory
./waf configure --libcmaes=${cwd}/install --dart=${cwd}/install --robot_dart=${cwd}/install --exp blackdrops