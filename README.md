### Black-DROPS algorithm

Code for the IROS 2017 paper: "Black-Box Data-efficient Policy Search for Robotics"

### Citing Black-DROPS

If you use our code for a scientific paper, please cite:

Chatzilygeroudis, K., Rama, R., Kaushik, R., Goepp, D., Vassiliades, V., & Mouret, J.-B. (2017). [Black-Box Data-efficient Policy Search for Robotics](https://arxiv.org/abs/1703.07261). *Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*.

In BibTex:
  
    @inproceedings{chatzilygeroudis2017black,
        title={{Black-Box Data-efficient Policy Search for Robotics}},
        author={Chatzilygeroudis, Konstantinos and Rama, Roberto and Kaushik, Rituraj and Goepp, Dorian and Vassiliades, Vassilis and Mouret, Jean-Baptiste},
        booktitle={Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems},
        year={2017},
        organization={IEEE}
    }

### Using the code

#### Dependencies

##### Required
- Ubuntu (it should work on versions >= 14.04)
- Eigen 3, http://eigen.tuxfamily.org/
- Boost
- limbo, https://github.com/resibots/limbo
- libcmaes, https://github.com/beniz/libcmaes (recommended to use with TBB)

##### Optional
- TBB, https://www.threadingbuildingblocks.org/ (for parallelization) --- highly recommended
- DART, http://dartsim.github.io/ (for scenarios based on DART) --- recommended
- SDL2 (for visualization of standard control scenarios; e.g., cart-pole)

#### How to properly clone this repo

- Clone properly the repo: `git clone --recursive https://github.com/resibots/blackdrops.git` (or `git clone --recursive git@github.com:resibots/blackdrops.git`)

#### Install dependencies

Some of the dependencies (libcmaes, DART) require specific installation steps (e.g., compilation from sources). As such, we provide some scripts for fast installation of the dependencies (3 different categories):

##### Install the recommended dependencies

Run the script, `install_deps.sh`

##### Install all dependencies

Run the script, `install_deps_all.sh`

##### Install only the required dependencies

Run the script `install_deps_req.sh`

#### Compilation

As the Black-DROPS code is a `limbo` experiment (check the [docs](http://www.resibots.eu/limbo/index.html) of limbo for details) and can sometimes be a bit tricky to compile, we provide the `compile.sh` script. This should compile all the Black-DROPS code. Even your own new scenarios should be compiled with this script (if the files are in the correct folders --- check *''How to create your own scenario''*).

If you want to know in more detail how to compile limbo experiments, please check the quite extensive [documentation](http://www.resibots.eu/limbo/index.html) of limbo. In addition, if you want more fine-tuned compilation of your own scenarios, please check the [advanced tutorial](here).

#### Run scenarios

- All the executables including your own new scenarios (assuming the compilation produced no errors) should be located in the `limbo/build` folder
- For example if we want to run the cartpole scenario without any visualization, we should use: `./limbo/build/src/classic_control/cartpole_simu [args]` (you can get help on what arguments to use, with `/path/to/binary --help`)

#### How to create your own scenario

Please look at the [basic tutorial](here). You will find detailed comments on how to create, compile and run your own scenarios.
