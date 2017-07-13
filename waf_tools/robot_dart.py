#!/usr/bin/env python
# encoding: utf-8
#| Copyright Inria July 2017
#| This project has received funding from the European Research Council (ERC) under
#| the European Union's Horizon 2020 research and innovation programme (grant
#| agreement No 637972) - see http://www.resibots.eu
#|
#| Contributor(s):
#|   - Konstantinos Chatzilygeroudis (konstantinos.chatzilygeroudis@inria.fr)
#|   - Roberto Rama (bertoski@gmail.com)
#|
#| This software is a computer library whose purpose is to optimize continuous,
#| black-box functions. It mainly implements Gaussian processes and Bayesian
#| optimization.
#| Main repository: http://github.com/resibots/blackdrops
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
import boost
import eigen
import dart
from waflib.Configure import conf


def options(opt):
  opt.load('boost')
  opt.load('eigen')
  opt.load('dart')
  opt.add_option('--robot_dart', type='string', help='path to hexapod_dart', dest='robot_dart')


@conf
def check_robot_dart(conf):
    conf.load('boost')
    conf.load('eigen')
    conf.load('dart')
    # In boost you can use the uselib_store option to change the variable the libs will be loaded
    boost_var = 'BOOST_DART'
    conf.check_boost(lib='regex system', min_version='1.46', uselib_store=boost_var)
    conf.check_eigen()
    conf.check_dart()
    includes_check = ['/usr/local/include', '/usr/include']
    libs_check = ['/usr/local/lib', '/usr/lib']
    # You can customize where you want to check
    # e.g. here we search also in a folder defined by an environmental variable
    if 'RESIBOTS_DIR' in os.environ:
    	includes_check = [os.environ['RESIBOTS_DIR'] + '/include'] + includes_check
    	libs_check = [os.environ['RESIBOTS_DIR'] + '/lib'] + libs_check
    if conf.options.robot_dart:
    	includes_check = [conf.options.robot_dart + '/include']
    	libs_check = [conf.options.robot_dart + '/lib']
    try:
    	conf.start_msg('Checking for robot_dart includes')
    	res = conf.find_file('robot_dart/robot.hpp', includes_check)
    	res = res and conf.find_file('robot_dart/robot_control.hpp', includes_check)
    	res = res and conf.find_file('robot_dart/robot_dart_simu.hpp', includes_check)
    	res = res and conf.find_file('robot_dart/descriptors.hpp', includes_check)
    	conf.end_msg('ok')
    	conf.env.INCLUDES_ROBOT_DART = includes_check
    except:
    	conf.end_msg('Not found', 'RED')
    	return
    return 1
