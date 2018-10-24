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

"""
Quick n dirty robot_dart detection
"""

import os
from waflib import Utils, Logs
from waflib.Configure import conf


def options(opt):
  opt.add_option('--robot_dart', type='string', help='path to robot_dart', dest='robot_dart')


@conf
def check_robot_dart(conf, *k, **kw):
    def get_directory(filename, dirs):
        res = conf.find_file(filename, dirs)
        return res[:-len(filename)-1]

    required = kw.get('required', False)

    includes_check = ['/usr/local/include', '/usr/include']
    libs_check = ['/usr/local/lib', '/usr/lib']

    # OSX/Mac uses .dylib and GNU/Linux .so
    lib_suffix = 'dylib' if conf.env['DEST_OS'] == 'darwin' else 'so'

    # # You can customize where you want to check
    # # e.g. here we search also in a folder defined by an environmental variable
    # if 'RESIBOTS_DIR' in os.environ:
    # 	includes_check = [os.environ['RESIBOTS_DIR'] + '/include'] + includes_check
    # 	libs_check = [os.environ['RESIBOTS_DIR'] + '/lib'] + libs_check

    if conf.options.robot_dart:
    	includes_check = [conf.options.robot_dart + '/include']
    	libs_check = [conf.options.robot_dart + '/lib']

    try:
    	conf.start_msg('Checking for robot_dart includes')
        dirs = []
        dirs.append(get_directory('robot_dart/robot.hpp', includes_check))
        dirs.append(get_directory('robot_dart/control/robot_control.hpp', includes_check))
        dirs.append(get_directory('robot_dart/robot_dart_simu.hpp', includes_check))
        dirs.append(get_directory('robot_dart/descriptor/base_descriptor.hpp', includes_check))

        # remove duplicates
        dirs = list(set(dirs))

        conf.end_msg(dirs)
        conf.env.INCLUDES_ROBOT_DART = dirs

        conf.start_msg('Checking for robot_dart library')
        libs_ext = ['.a', lib_suffix]
        lib_found = False
        type_lib = '.a'
        for lib in libs_ext:
            try:
                lib_dir = get_directory('libRobotDARTSimu' + lib, libs_check)
                lib_found = True
                type_lib = lib
                break
            except:
                lib_found = False
        conf.end_msg('libRobotDARTSimu' + type_lib)

        conf.env.LIBPATH_ROBOT_DART = lib_dir
        if type_lib == '.a':
            conf.env.STLIB_ROBOT_DART = 'RobotDARTSimu'
        else:
            conf.env.LIB_ROBOT_DART = 'RobotDARTSimu'
    except:
        if required:
            conf.fatal('Not found')
    	conf.end_msg('Not found', 'RED')
    	return
    return 1