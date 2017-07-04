#!/bin/bash
sudo apt-get -qq update
# install Eigen 3, Boost and TBB
sudo apt-get -qq --yes --force-yes install cmake libeigen3-dev libboost-serialization-dev libboost-filesystem-dev libboost-test-dev libboost-program-options-dev libboost-thread-dev libboost-regex-dev
# install google tests for libcmaes
sudo apt-get -qq --yes --force-yes install libgtest-dev autoconf automake libtool libgoogle-glog-dev libgflags-dev

# save current directory
cwd=$(pwd)
# create install dir
mkdir -p install

# do libgtest fix for libcmaes
cd /usr/src/gtest
sudo mkdir -p build && cd build
sudo cmake ..
sudo make
sudo cp *.a /usr/lib
# install libcmaes
cd ${cwd}/libcmaes
mkdir -p build && cd build
# no tbb for libcmaes
cmake -DUSE_TBB=OFF -DUSE_OPENMP=ON -DBUILD_PYTHON=ON -DCMAKE_INSTALL_PREFIX=${cwd}/install ..
make -j4
make install
# go back to original directory
cd ../..

# configure paths
source ./scripts/paths.sh

# installing NLOpt
wget http://members.loria.fr/JBMouret/mirrors/nlopt-2.4.2.tar.gz
tar -zxvf nlopt-2.4.2.tar.gz && cd nlopt-2.4.2
./configure -with-cxx --enable-shared --without-python --without-matlab --without-octave --prefix=${cwd}/install
make install

# just as fail-safe
sudo ldconfig