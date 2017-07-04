#!/bin/bash
sudo apt-get -qq update
# install Eigen 3, Boost and TBB
sudo apt-get -qq --yes --force-yes install cmake libeigen3-dev libtbb-dev libboost-serialization-dev libboost-filesystem-dev libboost-test-dev libboost-program-options-dev libboost-thread-dev libboost-regex-dev libsdl2-dev libsdl2-image-dev
# install google tests for libcmaes
sudo apt-get -qq --yes --force-yes install libgtest-dev autoconf automake libtool libgoogle-glog-dev libgflags-dev

# save current directory
cwd=$(pwd)
# create install dir
mkdir install

# do libgtest fix for libcmaes
cd /usr/src/gtest
sudo mkdir -p build && cd build
sudo cmake ..
sudo make
sudo cp *.a /usr/lib
# install libcmaes
cd ${cwd}/libcmaes
mkdir -p build && cd build
cmake -DUSE_TBB=ON -DUSE_OPENMP=OFF -DBUILD_PYTHON=ON -DCMAKE_INSTALL_PREFIX=${cwd}/install ..
make -j4
make install
# go back to original directory
cd ../..

# install DART dependencies
sudo apt-add-repository -qq --yes --force-yes ppa:libccd-debs/ppa
sudo apt-add-repository -qq --yes --force-yes ppa:fcl-debs/ppa
sudo apt-add-repository -qq --yes --force-yes ppa:dartsim/ppa
sudo apt-get -qq update
sudo apt-get -qq --yes --force-yes install libnlopt-dev libbullet-dev libtinyxml-dev libtinyxml2-dev liburdfdom-dev liburdfdom-headers-dev libxi-dev libxmu-dev freeglut3-dev libopenscenegraph-dev
# install DART
cd dart
mkdir build && cd build
cmake -DBUILD_PYTHON=ON -DCMAKE_INSTALL_PREFIX=${cwd}/install ..
make -j4
make install
# go back to original directory
cd ../..

# install robot_dart
cd robot_dart
./waf configure --dart=${cwd}/install
./waf install
# go back to original directory
cd ..

# just as fail-safe
sudo ldconfig

# configure paths
# configure LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${cwd}/libcmaes/lib/python2.7/dist-packages/:${cwd}/install/lib:${LD_LIBRARY_PATH}

# configure PYTHONPATH
export PYTHONPATH=${cwd}/libcmaes/lib/python2.7/dist-packages/:${PYTHONPATH}