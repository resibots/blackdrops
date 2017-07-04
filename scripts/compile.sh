#!/bin/bash
cwd=$(pwd)

# go to limbo directory
cd limbo

# save current directory
./waf --libcmaes=${cwd}/install --dart=${cwd}/install --robot_dart=${cwd}/install --exp blackdrops

# go back to original directory
cd ..