#!/bin/bash
cwd=$(pwd)

# configure paths
source ./scripts/paths.sh

# go to limbo directory
cd deps/limbo

# save current directory
./waf --libcmaes=${cwd}/install --nlopt=${cwd}/install --dart=${cwd}/install --robot_dart=${cwd}/install --exp blackdrops -j4

# go back to original directory
cd ../..