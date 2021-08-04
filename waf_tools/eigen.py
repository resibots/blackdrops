#!/usr/bin/env python
# encoding: utf-8
#| Copyright Inria July 2017
#| This project has received funding from the European Research Council (ERC) under
#| the European Union's Horizon 2020 research and innovation programme (grant
#| agreement No 637972) - see http://www.resibots.eu
#|
#| Contributor(s):
#|   - Konstantinos Chatzilygeroudis (konstantinos.chatzilygeroudis@inria.fr)
#|   - Rituraj Kaushik (rituraj.kaushik@inria.fr)
#|   - Roberto Rama (bertoski@gmail.com)
#|
#| This software is the implementation of the Black-DROPS algorithm, which is
#| a model-based policy search algorithm with the following main properties:
#|   - uses Gaussian processes (GPs) to model the dynamics of the robot/system
#|   - takes into account the uncertainty of the dynamical model when
#|                                                      searching for a policy
#|   - is data-efficient or sample-efficient; i.e., it requires very small
#|     interaction time with the system to find a working policy (e.g.,
#|     around 16-20 seconds to learn a policy for the cart-pole swing up task)
#|   - when several cores are available, it can be faster than analytical
#|                                                    approaches (e.g., PILCO)
#|   - it imposes no constraints on the type of the reward function (it can
#|                                                  also be learned from data)
#|   - it imposes no constraints on the type of the policy representation
#|     (any parameterized policy can be used --- e.g., dynamic movement
#|                                              primitives or neural networks)
#|
#| Main repository: http://github.com/resibots/blackdrops
#| Preprint: https://arxiv.org/abs/1703.07261
#|
#| This software is governed by the CeCILL-C license under French law and
#| abiding by the rules of distribution of free software.  You can  use,
#| modify and/ or redistribute the software under the terms of the CeCILL-C
#| license as circulated by CEA, CNRS and INRIA at the following URL
#| "http://www.cecill.info".
#|
#| As a counterpart to the access to the source code and  rights to copy,
#| modify and redistribute granted by the license, users are provided only
#| with a limited warranty  and the software's author,  the holder of the
#| economic rights,  and the successive licensors  have only  limited
#| liability.
#|
#| In this respect, the user's attention is drawn to the risks associated
#| with loading,  using,  modifying and/or developing or reproducing the
#| software by the user in light of its specific status of free software,
#| that may mean  that it is complicated to manipulate,  and  that  also
#| therefore means  that it is reserved for developers  and  experienced
#| professionals having in-depth computer knowledge. Users are therefore
#| encouraged to load and test the software's suitability as regards their
#| requirements in conditions enabling the security of their systems and/or
#| data to be ensured and,  more generally, to use and operate it in the
#| same conditions as regards security.
#|
#| The fact that you are presently reading this means that you have had
#| knowledge of the CeCILL-C license and that you accept its terms.
#|
# JB Mouret - 2009
# Federico Allocati - 2015

"""
Quick n dirty eigen3 detection
"""

import os, glob, types
from waflib.Configure import conf


def options(opt):
    opt.add_option('--eigen', type='string', help='path to eigen', dest='eigen')
    # to-do: rename this to --eigen_lapacke
    opt.add_option('--lapacke_blas', action='store_true', help='enable lapacke/blas if found (required Eigen>=3.3)', dest='lapacke_blas')


def eigen_version(conf, includes_check):
    world_version = -1
    major_version = -1
    minor_version = -1

    config_file = conf.find_file('Eigen/src/Core/util/Macros.h', includes_check)
    with open(config_file) as f:
        config_content = f.readlines()
    for line in config_content:
        world = line.find('#define EIGEN_WORLD_VERSION')
        major = line.find('#define EIGEN_MAJOR_VERSION')
        minor = line.find('#define EIGEN_MINOR_VERSION')
        if world > -1:
            world_version = int(line.split(' ')[-1].strip())
        if major > -1:
            major_version = int(line.split(' ')[-1].strip())
        if minor > -1:
            minor_version = int(line.split(' ')[-1].strip())
        if world_version > 0 and major_version > 0 and minor_version > 0:
            break
    return world_version, major_version, minor_version

@conf
def check_eigen(conf, *k, **kw):
    def get_directory(filename, dirs):
        res = conf.find_file(filename, dirs)
        return res[:-len(filename)-1]
    includes_check = ['/usr/include/eigen3', '/usr/local/include/eigen3', '/usr/include', '/usr/local/include']

    required = kw.get('required', False)
    min_version = kw.get('min_version', (3,3,3))

    # OSX/Mac uses .dylib and GNU/Linux .so
    suffix = 'dylib' if conf.env['DEST_OS'] == 'darwin' else 'so'

    if conf.options.eigen:
        includes_check = [conf.options.eigen]

    try:
        conf.start_msg('Checking for Eigen')
        incl = get_directory('Eigen/Core', includes_check)
        conf.env.INCLUDES_EIGEN = [incl]
        conf.end_msg(incl)

        # LAPACK (optional)
        if conf.options.lapacke_blas:
            conf.start_msg('Checking for LAPACKE/BLAS (optional)')

            if world_version == 3 and major_version >= 3:
                # Check for lapacke and blas
                extra_libs = ['/usr/lib', '/usr/local/lib', '/usr/local/opt/openblas/lib']
                blas_libs = ['blas', 'openblas']
                blas_lib = ''
                blas_path = ''
                for b in blas_libs:
                    try:
                        blas_path = get_directory('lib'+b+'.'+suffix, extra_libs)
                    except:
                        continue
                    blas_lib = b
                    break

                lapacke = False
                lapacke_path = ''
                try:
                    lapacke_path = get_directory('liblapacke.'+suffix, extra_libs)
                    lapacke = True
                except:
                    lapacke = False

                if lapacke or blas_lib != '':
                    conf.env.DEFINES_EIGEN = []
                    if lapacke_path != blas_path:
                        conf.env.LIBPATH_EIGEN = [lapacke_path, blas_path]
                    else:
                        conf.env.LIBPATH_EIGEN = [lapacke_path]
                    conf.env.LIB_EIGEN = []
                    conf.end_msg('LAPACKE: \'%s\', BLAS: \'%s\'' % (lapacke_path, blas_path))
                elif lapacke:
                    conf.end_msg('Found only LAPACKE: %s' % lapacke_path, 'YELLOW')
                elif blas_lib != '':
                    conf.end_msg('Found only BLAS: %s' % blas_path, 'YELLOW')
                else:
                    conf.end_msg('Not found in %s' % str(extra_libs), 'RED')
                if lapacke:
                    conf.env.DEFINES_EIGEN.append('EIGEN_USE_LAPACKE')
                    conf.env.LIB_EIGEN.append('lapacke')
                if blas_lib != '':
                    conf.env.DEFINES_EIGEN.append('EIGEN_USE_BLAS')
                    conf.env.LIB_EIGEN.append(blas_lib)
            else:
                conf.end_msg('Found Eigen version %s: LAPACKE/BLAS can be used only with Eigen>=3.3' % (str(world_version) + '.' + str(major_version) + '.' + str(minor_version)), 'RED')
    except:
        if required:
            conf.fatal('Not found in %s' % str(includes_check))
        conf.end_msg('Not found in %s' % str(includes_check), 'RED')
        return 1

    # check the version
    conf.start_msg('Checking for Eigen version')
    version = eigen_version(conf, includes_check)
    if version < min_version:
        conf.fatal("Found version {}.{}.{} but version {}.{}.{} is required".format(version[0], version[1], version[2], min_version[0], min_version[1], min_version[2]))
    conf.end_msg("{}.{}.{}".format(version[0], version[1], version[2]))
    return 1
