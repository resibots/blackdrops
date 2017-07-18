## Installation of Black-DROPS code

Since Black-DROPS is a `limbo` experiment (check the [docs](http://www.resibots.eu/limbo/index.html) of limbo for details), there needs to be no installation. Nevertheless, the dependencies must be installed.

### How to properly clone this repo

- Clone the repo *recursively*: `git clone --recursive https://github.com/resibots/blackdrops.git` (or `git clone --recursive git@github.com:resibots/blackdrops.git`)

### Dependencies

#### Required
- Ubuntu (it should work on versions >= 14.04)
- limbo, https://github.com/resibots/limbo (for high-performing Gaussian process regression)
- libcmaes, https://github.com/beniz/libcmaes (for high-quality implementations of CMA-ES variants) --- recommended to use with TBB
- Eigen3 (needed by limbo and libcmaes)
- Boost (needed by limbo)
- NLOpt, http://ab-initio.mit.edu/wiki/index.php/NLopt (needed by limbo)

#### Optional
- TBB, https://www.threadingbuildingblocks.org/ (for parallelization) --- highly recommended
- DART, http://dartsim.github.io/ (for scenarios based on DART) --- recommended
- robot\_dart, https://github.com/resibots/robot_dart (for scenarios based on DART) --- recommended
- SDL2 (for visualization of standard control scenarios; e.g., cart-pole)

### Installation of the dependencies

Some of the dependencies (libcmaes, DART, NLOpt, robot\_dart) require specific installation steps (e.g., compilation from sources). As such, we provide some scripts (under the `scripts` folder) for automatic installation of the dependencies (3 different categories):

#### Install the recommended dependencies

- `cd /path/to/repo/root` **(this is very important as the script assumes that you are in the root of the repo)**
- `./scripts/install_deps.sh`

#### Install all the dependencies

- `cd /path/to/repo/root` **(this is very important as the script assumes that you are in the root of the repo)**
- `./scripts/install_deps_all.sh`

#### Install only the required dependencies

- `cd /path/to/repo/root` **(this is very important as the script assumes that you are in the root of the repo)**
- `./scripts/install_deps_req.sh`

Using the scripts, all of the custom dependencies (limbo, libcmaes, DART, NLOpt, robot\_dart) will be installed in `/path/to/repo/root/install` in order not to pollute your linux distribution. As such, you should update your `LD_LIBRARY_PATH` (or you can source the proper script --- see below). Consequently no `sudo` is required for these dependencies; nevertheless, `sudo` is still required for installing standard packages (like boost-dev packages, libeigen3-dev, etc). 

### Compilation

As the Black-DROPS code is a `limbo` experiment and can sometimes be a bit tricky to compile, we provide the `configure.sh` and `compile.sh` scripts. The former needs to be ran once. The latter should compile all the Black-DROPS code. Even your own new scenarios should be compiled with this script (if the files are in the correct folders --- see [*''How to create your own scenario''*](basic_tutorial.md)). In short you should do the following:

- `cd /path/to/repo/root` **(this is very important as the scripts assume that you are in the root of the repo)**
- `./scripts/configure.sh`
- `./scripts/compile.sh`

And then every time you make a change to a source file (*\*.hpp or \*.cpp*), you should re-run the compilation script. If you want to know in more detail how to compile limbo experiments (i.e, not with the scripts), please check the quite extensive [documentation](http://www.resibots.eu/limbo/index.html) of limbo.
<!--In addition, if you want more fine-tuned compilation of your own scenarios, please check the [advanced tutorial](here).-->

### Running scenarios

- Before running any executable you should source the proper paths: `source ./scripts/paths.sh` **(the script assumes that you are in the root of the repo)**
- All the executables including your own new scenarios (assuming the compilation produced no errors) should be located in the `deps/limbo/build/exp/blackdrops/src/` folder
- For example if we want to run the cartpole scenario without any visualization, we should use: `./deps/limbo/build/exp/blackdrops/src/classic_control/cartpole_simu [args]` (you can get help on what arguments to use, with `/path/to/binary --help`)