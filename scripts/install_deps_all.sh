#!/bin/bash
set -x

OS=$(uname)
echo "Detected OS: $OS"

if [ $OS = "Darwin" ]; then
    echo "Installing for OSX via brew"
    # general stuffs
    brew install cmake eigen boost tbb autoconf automake
else
    # check if we have Ubuntu or not
    distro_str="$(cat /etc/*-release | grep -s DISTRIB_ID)"
    distro=$(echo $distro_str | cut -f2 -d'=')

    if [ "$distro" != "Ubuntu" ]; then
        echo "ERROR: We need an Ubuntu system to use this script"
        exit 1
    fi

    # get ubuntu version
    version_str="$(cat /etc/*-release | grep -s DISTRIB_RELEASE)"
    version=$(echo $version_str | cut -f2 -d'=')
    major_version=$(echo $version | cut -f1 -d'.')
    minor_version=$(echo $version | cut -f2 -d'.')

    # if less than 14.04, exit
    if [ "$(($major_version))" -lt "14" ]; then
        echo "ERROR: We need Ubuntu >= 14.04 for this script to work"
        exit 1
    fi

    sudo apt-get -qq update
    # install Eigen 3, Boost and TBB
    sudo apt-get --yes --force-yes install cmake libeigen3-dev libtbb-dev libboost-serialization-dev libboost-filesystem-dev libboost-test-dev libboost-program-options-dev libboost-thread-dev libboost-regex-dev libsdl2-dev
    # install google tests for libcmaes
    sudo apt-get --yes --force-yes install libgtest-dev autoconf automake libtool libgoogle-glog-dev libgflags-dev
fi

# save current directory
cwd=$(pwd)
# create install dir
mkdir -p install

# do libgtest fix for libcmaes (linux only)
if [ $OS = "Linux" ]; then
    cd /usr/src/gtest
    sudo mkdir -p build && cd build
    sudo cmake ..
    sudo make
    sudo cp *.a /usr/lib
fi

# install libcmaes
cd ${cwd}/deps/libcmaes
mkdir -p build && cd build
cmake -DUSE_TBB=ON -DUSE_OPENMP=OFF -DBUILD_PYTHON=ON -DCMAKE_INSTALL_PREFIX=${cwd}/install ..
make -j4
make install
# go back to original directory
cd ../../..

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

if [ $OS = "Darwin" ]; then
 # for DART
    brew install assimp
    brew install homebrew/science/libccd
    brew install dartsim/dart/fcl
    brew install open-scene-graph

    brew install tinyxml
    brew install tinyxml2
    brew install ros/deps/urdfdom
else
    # if we have less than 16.04, we need some extra stuff
    if [ "$(($major_version))" -lt "16" ]; then
        sudo apt-add-repository ppa:libccd-debs/ppa -y
        sudo apt-add-repository ppa:fcl-debs/ppa -y
    fi
    sudo apt-add-repository ppa:dartsim/ppa -y
    sudo apt-get -qq update
    sudo apt-get --yes --force-yes install build-essential pkg-config libassimp-dev libccd-dev libfcl-dev
    sudo apt-get --yes --force-yes install libnlopt-dev libbullet-dev libtinyxml-dev libtinyxml2-dev liburdfdom-dev liburdfdom-headers-dev libxi-dev libxmu-dev freeglut3-dev libopenscenegraph-dev   
fi

# install DART
cd deps/dart
mkdir -p build && cd build
cmake -DDART_ENABLE_SIMD=ON -DBUILD_PYTHON=ON -DCMAKE_INSTALL_PREFIX=${cwd}/install ..
make -j4
make install
# go back to original directory
cd ../../..

# just as fail-safe
if [ $OS = "Darwin" ]; then
    update_dyld_shared_cache
else
    sudo ldconfig
fi

# configure paths to find DART related libraries properly
source ./scripts/paths.sh

# install robot_dart
cd deps/robot_dart
./waf configure --dart=${cwd}/install --prefix=${cwd}/install
./waf
./waf install
# go back to original directory
cd ../..
