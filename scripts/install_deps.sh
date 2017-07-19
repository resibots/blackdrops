#!/bin/bash

OS=$(uname)
echo "Detected OS: $OS"

if [ $OS = "Darwin" ]; then
    echo "ERROR: OSX is only for install_deps_req.sh"
    exit 1
fi

sudo apt-get -qq update
# install Eigen 3, Boost and TBB
sudo apt-get --yes --force-yes install cmake libeigen3-dev libtbb-dev libboost-serialization-dev libboost-filesystem-dev libboost-test-dev libboost-program-options-dev libboost-thread-dev libboost-regex-dev
# install google tests for libcmaes
sudo apt-get --yes --force-yes install libgtest-dev autoconf automake libtool libgoogle-glog-dev libgflags-dev

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
cd ${cwd}/deps/libcmaes
mkdir build -p && cd build
cmake -DUSE_TBB=ON -DUSE_OPENMP=OFF -DBUILD_PYTHON=ON -DCMAKE_INSTALL_PREFIX=${cwd}/install ..
make -j4
make install
# go back to original directory
cd ../..

# configure paths
source ./scripts/paths.sh

# installing NLOpt
cd deps
wget http://members.loria.fr/JBMouret/mirrors/nlopt-2.4.2.tar.gz
tar -zxvf nlopt-2.4.2.tar.gz && cd nlopt-2.4.2
./configure -with-cxx --enable-shared --without-python --without-matlab --without-octave --prefix=${cwd}/install
make install
# go back to original directory
cd ../..

# install DART dependencies
sudo apt-add-repository ppa:libccd-debs/ppa -y
sudo apt-add-repository ppa:fcl-debs/ppa -y
sudo apt-add-repository ppa:dartsim/ppa -y
sudo apt-get -qq update
sudo apt-get --yes --force-yes install build-essential pkg-config libassimp-dev libccd-dev libfcl-dev
sudo apt-get --yes --force-yes install libnlopt-dev libbullet-dev libtinyxml-dev libtinyxml2-dev liburdfdom-dev liburdfdom-headers-dev libxi-dev libxmu-dev freeglut3-dev libopenscenegraph-dev
# install DART
cd deps/dart
mkdir -p build && cd build
cmake -DDART_ENABLE_SIMD=ON -DBUILD_PYTHON=ON -DCMAKE_INSTALL_PREFIX=${cwd}/install ..
make -j4
make install
# go back to original directory
cd ../../..

# just as fail-safe
sudo ldconfig

# install robot_dart
cd deps/robot_dart
./waf configure --dart=${cwd}/install --prefix=${cwd}/install
./waf install
# go back to original directory
cd ../..