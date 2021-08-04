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
Quick n dirty DART detection
"""

import os
from copy import deepcopy
from waflib.Configure import conf


def options(opt):
    opt.add_option('--dart', type='string', help='path to DART physics engine/sim', dest='dart')

@conf
def check_dart(conf, *k, **kw):
    def fail(msg, required):
        if required:
            conf.fatal(msg)
        conf.end_msg(msg, 'RED')
    def get_directory(filename, dirs):
        res = conf.find_file(filename, dirs)
        return res[:-len(filename)-1]

    required = kw.get('required', False)

    # OSX/Mac uses .dylib and GNU/Linux .so
    suffix = 'dylib' if conf.env['DEST_OS'] == 'darwin' else 'so'

    if conf.options.dart:
        includes_check = [conf.options.dart + '/include']
        libs_check = [conf.options.dart + '/lib']
    else:
        includes_check = ['/usr/local/include', '/usr/include']
        libs_check = ['/usr/local/lib', '/usr/local/lib64', '/usr/lib', '/usr/lib64', '/usr/lib/x86_64-linux-gnu/']

        if 'RESIBOTS_DIR' in os.environ:
            includes_check = [os.environ['RESIBOTS_DIR'] + '/include'] + includes_check
            libs_check = [os.environ['RESIBOTS_DIR'] + '/lib'] + libs_check

    # DART requires assimp library
    assimp_include = []
    assimp_lib = []
    assimp_check = ['/usr/local/include', '/usr/include']
    assimp_libs = ['/usr/local/lib', '/usr/local/lib64', '/usr/lib', '/usr/lib64', '/usr/lib/x86_64-linux-gnu/']
    assimp_found = False
    try:
        assimp_include = get_directory('assimp/scene.h', assimp_check)
        assimp_lib = [get_directory('libassimp.' + suffix, assimp_libs)]
        assimp_found = True
    except:
        assimp_found = False

    # DART has some optional Bullet features
    bullet_check = ['/usr/local/include/bullet', '/usr/include/bullet']
    bullet_libs = ['/usr/local/lib', '/usr/local/lib64', '/usr/lib', '/usr/lib64', '/usr/lib/x86_64-linux-gnu/']
    bullet_include = []
    bullet_lib = []
    bullet_found = False
    try:
        bullet_include = [get_directory('btBulletCollisionCommon.h', bullet_check)]
        bullet_lib = []
        bullet_lib.append(get_directory('libLinearMath.' + suffix, bullet_libs))
        bullet_lib.append(get_directory('libBulletCollision.' + suffix, bullet_libs))
        bullet_lib = list(set(bullet_lib))
        bullet_found = True
    except:
        bullet_found = False

    # DART has some optional ODE features
    ode_check = ['/usr/local/include', '/usr/include']
    ode_libs = ['/usr/local/lib', '/usr/local/lib64', '/usr/lib', '/usr/lib64', '/usr/lib/x86_64-linux-gnu/']
    ode_include = []
    ode_lib = []
    ode_found = False
    try:
        ode_include = [get_directory('ode/collision.h', ode_check)]
        try:
            ode_lib = [get_directory('libode.' + suffix, ode_libs)]
        except:
            ode_lib = [get_directory('libode.a' , ode_libs)]
        ode_found = True
    except:
        ode_found = False

    # DART has some optional Octomap dependency
    octomap_check = ['/usr/local/include', '/usr/include']
    octomap_libs = ['/usr/local/lib', '/usr/local/lib64', '/usr/lib', '/usr/lib64', '/usr/lib/x86_64-linux-gnu/']
    if 'ROS_DISTRO' in os.environ:
        octomap_check.append('/opt/ros/' + os.environ['ROS_DISTRO'] + '/include')
        octomap_libs.append('/opt/ros/' + os.environ['ROS_DISTRO'] + '/lib')
    octomap_include = []
    octomap_lib = []
    octomap_found = False
    try:
        octomap_include = [get_directory('octomap/octomap.h', octomap_check)]
        octomap_lib = [get_directory('liboctomap.' + suffix, octomap_libs)]
        octomap_found = True
    except:
        octomap_found = False

    dart_load_prefix = 'utils'
    dart_include = []
    dart_major = -1
    dart_minor = -1
    dart_patch = -1
    # DART Configurations
    dart_have_nlopt = False
    dart_have_ipopt = False
    dart_have_pagmo = False
    dart_have_snopt = False
    dart_have_bullet = False
    dart_have_ode = False
    dart_have_flann = False
    dart_have_octomap = False

    try:
        conf.start_msg('Checking for DART includes (including io/urdf)')
        config_file = conf.find_file('dart/config.hpp', includes_check)
        with open(config_file) as f:
            config_content = f.readlines()
        for line in config_content:
            major = line.find('#define DART_MAJOR_VERSION')
            minor = line.find('#define DART_MINOR_VERSION')
            patch = line.find('#define DART_PATCH_VERSION')
            if major > -1:
                dart_major = int(line.split(' ')[-1].strip())
            if minor > -1:
                dart_minor = int(line.split(' ')[-1].strip())
            if patch > -1:
                dart_patch = int(line.split(' ')[-1].strip())

            nlopt = line.find('#define HAVE_NLOPT')
            ipopt = line.find('#define HAVE_IPOPT')
            pagmo = line.find('#define HAVE_PAGMO')
            snopt = line.find('#define HAVE_SNOPT')
            bullet = line.find('#define HAVE_BULLET')
            ode = line.find('#define HAVE_ODE')
            flann = line.find('#define HAVE_FLANN')
            octomap = line.find('#define HAVE_OCTOMAP')

            if nlopt > -1:
                boolean = int(line.split(' ')[-1].strip())
                dart_have_nlopt = (boolean == 1)
            if ipopt > -1:
                boolean = int(line.split(' ')[-1].strip())
                dart_have_ipopt = (boolean == 1)
            if pagmo > -1:
                boolean = int(line.split(' ')[-1].strip())
                dart_have_pagmo = (boolean == 1)
            if snopt > -1:
                boolean = int(line.split(' ')[-1].strip())
                dart_have_snopt = (boolean == 1)
            if bullet > -1:
                boolean = int(line.split(' ')[-1].strip())
                dart_have_bullet = (boolean == 1)
            if ode > -1:
                boolean = int(line.split(' ')[-1].strip())
                dart_have_ode = (boolean == 1)
            if flann > -1:
                boolean = int(line.split(' ')[-1].strip())
                dart_have_flann = (boolean == 1)
            if octomap > -1:
                boolean = int(line.split(' ')[-1].strip())
                dart_have_octomap = (boolean == 1)

        if dart_major < 6:
            raise Exception('We need DART version at least 6.0.0')
        if dart_major > 6:
            dart_load_prefix = 'io'
        dart_cxx_flags = ''
        if dart_major > 6 or (dart_major == 6 and dart_minor >= 9):
            dart_cxx_flags = '-std=c++14'

        dart_include = []
        dart_include.append(get_directory('dart/dart.hpp', includes_check))
        dart_include.append(get_directory('dart/'+dart_load_prefix+'/'+dart_load_prefix+'.hpp', includes_check))
        dart_include.append(get_directory('dart/'+dart_load_prefix+'/urdf/urdf.hpp', includes_check))
        dart_include = list(set(dart_include))
        conf.end_msg(str(dart_major)+'.'+str(dart_minor)+'.'+str(dart_patch)+' in '+dart_include[0])

        more_includes = []
        if assimp_found:
            more_includes += assimp_include

        conf.start_msg('Checking for DART libs (including io/urdf)')
        dart_lib = []
        dart_lib.append(get_directory('libdart.' + suffix, libs_check))
        dart_lib.append(get_directory('libdart-'+dart_load_prefix+'.' + suffix, libs_check))
        dart_lib.append(get_directory('libdart-'+dart_load_prefix+'-urdf.' + suffix, libs_check))
        dart_lib = list(set(dart_lib))
        conf.env.INCLUDES_DART = dart_include + more_includes
        conf.env.LIBPATH_DART = dart_lib
        conf.env.LIB_DART = ['dart', 'dart-'+dart_load_prefix, 'dart-'+dart_load_prefix+'-urdf']
        if len(dart_cxx_flags) > 0:
            conf.env.CXXFLAGS_DART = [dart_cxx_flags]
        conf.end_msg(conf.env.LIB_DART)
        conf.start_msg('DART: Checking for Assimp')
        if assimp_found:
            conf.end_msg(assimp_include)
            conf.env.LIBPATH_DART = conf.env.LIBPATH_DART + assimp_lib
            conf.env.LIB_DART.append('assimp')
        else:
            conf.end_msg('Not found - Your programs may not compile', 'RED')

        if dart_have_bullet:
            conf.start_msg('DART: Checking for Bullet Collision libs')
            if bullet_found:
                try:
                    bullet_lib.append(get_directory('libdart-collision-bullet.'+suffix, libs_check))
                    conf.env.INCLUDES_DART = conf.env.INCLUDES_DART + bullet_include
                    conf.env.LIBPATH_DART =  conf.env.LIBPATH_DART + bullet_lib
                    conf.env.LIB_DART.append('LinearMath')
                    conf.env.LIB_DART.append('BulletCollision')
                    conf.env.LIB_DART.append('dart-collision-bullet')
                    conf.end_msg('libs: ' + str(conf.env.LIB_DART[-3:]) + ', bullet: ' + str(bullet_include[0]))
                except:
                    fail('Not found', required)
            else:
                fail('Not found', required)

        if dart_have_ode:
            conf.start_msg('DART: Checking for Ode Collision libs')
            if ode_found:
                try:
                    ode_lib.append(get_directory('libdart-collision-ode.'+suffix, libs_check))
                    conf.env.INCLUDES_DART = conf.env.INCLUDES_DART + ode_include
                    conf.env.LIBPATH_DART =  conf.env.LIBPATH_DART + ode_lib
                    conf.env.LIB_DART.append('ode')
                    conf.env.LIB_DART.append('dart-collision-ode')
                    conf.end_msg('libs: ' + str(conf.env.LIB_DART[-2:]) + ', ode: ' + str(ode_include[0]))
                except:
                    fail('Not found', required)
            else:
                fail('Not found', required)

        if dart_have_octomap:
            conf.start_msg('DART: Checking for Octomap libs')
            if octomap_found:
                try:
                    conf.env.INCLUDES_DART = conf.env.INCLUDES_DART + octomap_include
                    conf.env.LIBPATH_DART =  conf.env.LIBPATH_DART + octomap_lib
                    conf.env.LIB_DART.append('octomap')
                    conf.env.LIB_DART.append('octomath')
                    conf.end_msg('libs: ' + str(conf.env.LIB_DART[-1:]) + ', octomap: ' + str(octomap_include[0]))
                except:
                    fail('Not found', required)
            else:
                fail('Not found', required)

        # remove duplicates
        conf.env.INCLUDES_DART = list(set(conf.env.INCLUDES_DART))
        conf.env.LIBPATH_DART = list(set(conf.env.LIBPATH_DART))
    except:
        if dart_major < 6 and dart_major > -1:
            fail('We need DART >= 6.0.0', required)
        else:
            fail('Not found', required)
        return
    return 1
