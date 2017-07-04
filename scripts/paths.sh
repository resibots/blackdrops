# get current directory
cwd=$(pwd)

# configure paths
# configure LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${cwd}/libcmaes/lib/python2.7/dist-packages/:${cwd}/install/lib:${LD_LIBRARY_PATH}

# configure PYTHONPATH
export PYTHONPATH=${cwd}/libcmaes/lib/python2.7/dist-packages/:${PYTHONPATH}