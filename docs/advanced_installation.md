## Advanced installation and compilation of Black-DROPS code

We will go step by step on how to compile the Black-DROPS code and its dependencies by hand. Before proceeding it is recommended that you read the [basic installation tutorial](installation.md) to get an idea of the process needed to be followed in general.

### Black-DROPS is a limbo experiment

The Black-DROPS code is a limbo experiment (see the [limbo docs](http://www.resibots.eu/limbo/guides/framework.html#what-is-a-limbo-experiment) for more details). This roughly means that the Black-DROPS code should be compiled within the `exp` folder of the limbo framework.

#### Installing limbo and its dependencies

Limbo has quite a few basic dependencies:

- Eigen3
- Boost
- NLOpt (optionally but is recommended that we use it)

For Ubuntu-based distributions we should use the following commands to install Eigen3 and Boost:

```bash
sudo apt-get update
sudo apt-get install libeigen3-dev libboost-serialization-dev libboost-filesystem-dev libboost-test-dev libboost-program-options-dev libboost-thread-dev libboost-regex-dev
```

For OSX with brew:

```bash
brew install eigen
brew install boost
```

We highly recommend that you install NLOpt. Unfortunately, the Ubuntu packages do not provide NLOpt’s C++ bindings. You can get NLOpt here: http://ab-initio.mit.edu/wiki/index.php/NLopt [mirror: http://members.loria.fr/JBMouret/mirrors/nlopt-2.4.2.tar.gz].

For both Ubuntu and OSX systems, we should run something like the following:

```bash
cd /path/to/tmp/folder
wget http://members.loria.fr/JBMouret/mirrors/nlopt-2.4.2.tar.gz
tar -zxvf nlopt-2.4.2.tar.gz && cd nlopt-2.4.2
./configure -with-cxx --enable-shared --without-python --without-matlab --without-octave
sudo make install
```

If you want to install NLOpt somewhere else than `/usr/local/`, you add the `--prefix=/path/to/install/folder` in the `./configure` command.

Now that the basic dependencies of limbo have been installed, we can proceed in cloning the limbo framework (we need the `spt` branch):

```bash
cd /path/to/code/folder
git clone https://github.com/resibots/limbo.git
cd limbo
./waf configure
./waf
```

You can also clone with ssh if desired: ``git clone git@github.com:resibots/limbo.git``.

### Installing libcmaes

Next step is to install the [libcmaes](https://github.com/beniz/libcmaes) library. We have to first install the dependencies of the library.

For Ubuntu systems:

```bash
sudo apt-get install libgtest-dev autoconf automake libtool libgoogle-glog-dev libgflags-dev
```

For OSX:

```bash
brew install cmake autoconf automake
```

For Ubuntu systems, we need to make a minor fix for the gtest:

```bash
cd /usr/src/gtest
sudo mkdir -p build && cd build
sudo cmake ..
sudo make
sudo cp *.a /usr/lib
```

We also highly recommend to install [TBB](https://www.threadingbuildingblocks.org/). For Ubuntu systems:

```bash
sudo apt-get install libtbb-dev
```

For OSX:

```bash
brew install tbb
```

We have slightly modified the library to suit our special use-case and thus we provide our own fork of the library:

```bash
cd /path/to/tmp/folder
git clone https://github.com/resibots/libcmaes.git
cd libcmaes
git checkout fix_flags_native
```

Now let's compile and install the library:

```bash
mkdir -p build && cd build
cmake -DUSE_TBB=ON -DUSE_OPENMP=OFF ..
make -j4
sudo make install
```

If you do not have TBB installed, you should use: `cmake -DUSE_TBB=OFF -DUSE_OPENMP=ON ..`. If you want to install libcmaes somewhere else than `/usr/local`, you should use the `-DCMAKE_INSTALL_PREFIX=/path/to/install/folder` cmake argument.

### Installing DART

If you want to compile and create [DART](http://dartsim.github.io/)-based scenarios, you will need to install the upstream DART by source.

For **Ubuntu systems**, please follow the detailed installation instructions on the [DART documentation website](http://dartsim.github.io/install_dart_on_ubuntu.html#install-required-dependencies). Make sure that you don't forget to add the PPAs as detailed [here](http://dartsim.github.io/install_dart_on_ubuntu.html#adding-personal-package-archives-ppas-for-dart-and-dependencies). What is more, you need to enable the `-DART_ENABLE_SIMD` flag in the CMake configuration. In addition, you need the following optional dependencies: **DART Parsers** and **OpenSceneGraph GUI**. Lastly, you need to checkout to the `release-6.7` branch (and not the one provided in DART's documentation).

For **Ubuntu <= 14.04** one more step is needed:

```bash
sudo apt-add-repository ppa:libccd-debs/ppa
sudo apt-add-repository ppa:fcl-debs/ppa
```

Here's what you should do:

```bash
sudo apt-add-repository ppa:dartsim/ppa
sudo apt-get update

sudo apt-get install build-essential cmake pkg-config git
sudo apt-get install libeigen3-dev libassimp-dev libccd-dev libfcl-dev libboost-regex-dev libboost-system-dev
sudo apt-get install libopenscenegraph-dev

sudo apt-get install libtinyxml-dev libtinyxml2-dev
sudo apt-get install liburdfdom-dev liburdfdom-headers-dev

cd /path/to/tmp/folder
git clone git://github.com/dartsim/dart.git
cd dart
git checkout release-6.7

mkdir build
cd build
cmake -DDART_ENABLE_SIMD=ON ..
make -j4
sudo make install
```

If you want to install DART somewhere else than `/usr/local`, you should use the `-DCMAKE_INSTALL_PREFIX=/path/to/install/folder` cmake argument.

For **OSX systems** with homebrew, please follow the detailed installation instructions on the [DART documentation website](http://dartsim.github.io/install_dart_on_mac.html#install-from-source-using-homebrew). You need to follow the same procedure as for Ubuntu systems. In short you should do the following:

```bash
brew install eigen
brew install assimp
brew install homebrew/science/libccd
brew install dartsim/dart/fcl
brew install boost
brew install open-scene-graph

brew install tinyxml
brew install tinyxml2
brew install ros/deps/urdfdom

cd /path/to/tmp/folder
git clone git://github.com/dartsim/dart.git
cd dart
git checkout release-6.7

mkdir build
cd build
cmake -DDART_ENABLE_SIMD=ON ..
make -j4
sudo make install
```

The last step to be able to run and create DART-based scenarios with the Black-DROPS code is to install the `robot_dart` wrapper:

```bash
cd /path/to/tmp/folder
git clone https://github.com/resibots/robot_dart.git
cd robot_dart
./waf configure
sudo ./waf install
```

If you have installed DART in a different location than the default ones (i.e., other than `/usr/local/` or `/usr/`), you the `--dart=/path/to/dart/install` option in `./waf configure`.

### Installing SDL2

For simple graphics, we use the SDL2 library.

For Ubuntu systems, you should:

```bash
sudo apt-get install libsdl2-dev libsdl2-image-dev
```

For OSX systems:

```bash
brew install sdl2
```

### Installing simple_nn

For neural network policies, we are using the [simple_nn](https://github.com/resibots/simple_nn) library.

You should do the following:

```bash
cd /path/to/tmp/folder
git clone https://github.com/resibots/simple_nn.git
cd simple_nn
./waf configure
sudo ./waf install
```

### Compiling the Black-DROPS code

We've now reached the point where we can compile the Black-DROPS code. You should do the following:

```bash
cd /path/to/limbo/folder
mkdir -p exp
cd exp
git clone https://github.com/resibots/blackdrops.git #(no need for recursive cloning)
cd ..
./waf configure --exp blackdrops
./waf --exp blackdrops -j4
```

Please remember that if you changed the installation location of DART, libcmaes or NLOpt, you should adapt your `LD_LIBRARY_PATH`.

And then every time you make a change to a source file (*\*.hpp or \*.cpp*), you should re-run the compilation command: `./waf --exp blackdrops -j4` (from the root of the limbo folder).

### Running scenarios

- All the executables including your own new scenarios (assuming the compilation produced no errors) should be located in the `/path/to/limbo/build/exp/blackdrops/src/` folder
- For example if we want to run the cartpole scenario without any visualization, we should use: `./path/to/limbo/build/exp/blackdrops/src/classic_control/cartpole_simu [args]` (you can get help on what arguments to use, with `/path/to/binary --help`)
