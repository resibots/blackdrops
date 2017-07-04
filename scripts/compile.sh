#!/bin/bash
cwd=$(pwd)

# configure paths
# configure LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${cwd}/libcmaes/lib/python2.7/dist-packages/:${cwd}/install/lib:${LD_LIBRARY_PATH}

# configure PYTHONPATH
export PYTHONPATH=${cwd}/libcmaes/lib/python2.7/dist-packages/:${PYTHONPATH}

# go to limbo directory
cd limbo

# save current directory
./waf --libcmaes=${cwd}/install --nlopt=${cwd}/install --dart=${cwd}/install --robot_dart=${cwd}/install --exp blackdrops

# go back to original directory
cd ..