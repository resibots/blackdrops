#!/bin/bash
cwd=$(pwd)

# configure paths
source ./scripts/paths.sh

# go to limbo
cd limbo
mkdir -p exp
cd exp
ln -s ../../ blackdrops
# go back to limbo
cd ..

# save current directory
./waf configure --libcmaes=${cwd}/install --nlopt=${cwd}/install --dart=${cwd}/install --robot_dart=${cwd}/install --exp blackdrops

# go back to original directory
cd ..