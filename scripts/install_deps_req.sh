#!/bin/bash
sudo apt-get -qq update
# install Eigen 3, Boost and TBB
sudo apt-get -qq --yes --force-yes install cmake libeigen3-dev libboost-serialization-dev libboost-filesystem-dev libboost-test-dev libboost-program-options-dev libboost-thread-dev
# install google tests for libcmaes
sudo apt-get -qq --yes --force-yes install libgtest-dev autoconf automake libtool libgoogle-glog-dev libgflags-dev

# save current directory
cwd=$(pwd)
# create install dir
mkdir install

# do libgtest fix for libcmaes
cd /usr/src/gtest
sudo mkdir build && cd build
sudo cmake ..
sudo make
sudo cp *.a /usr/lib
# install libcmaes
cd ${cwd}/libcmaes
mkdir build && cd build
# no tbb for libcmaes
cmake -DUSE_TBB=OFF -DUSE_OPENMP=ON -DBUILD_PYTHON=ON -DCMAKE_INSTALL_PREFIX=${cwd}/install ..
make -j4
make install
# go back to original directory
cd ../..

# just as fail-safe
sudo ldconfig

# configure paths
# configure LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${cwd}/libcmaes/lib/python2.7/dist-packages/:${cwd}/install/lib:${LD_LIBRARY_PATH}

# configure PYTHONPATH
export PYTHONPATH=${cwd}/libcmaes/lib/python2.7/dist-packages/:${PYTHONPATH}