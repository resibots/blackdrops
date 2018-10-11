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
        libs_check = ['/usr/local/lib', '/usr/lib', '/usr/lib/x86_64-linux-gnu/']

        if 'RESIBOTS_DIR' in os.environ:
            includes_check = [os.environ['RESIBOTS_DIR'] + '/include'] + includes_check
            libs_check = [os.environ['RESIBOTS_DIR'] + '/lib'] + libs_check

    # DART requires assimp library
    assimp_include = []
    assimp_lib = []
    assimp_check = ['/usr/local/include', '/usr/include']
    assimp_libs = ['/usr/local/lib', '/usr/lib', '/usr/lib/x86_64-linux-gnu/']
    assimp_found = False
    try:
        assimp_include = get_directory('assimp/scene.h', assimp_check)
        assimp_lib = [get_directory('libassimp.' + suffix, assimp_libs)]
        assimp_found = True
    except:
        assimp_found = False

    # DART has some optional Bullet features
    bullet_check = ['/usr/local/include/bullet', '/usr/include/bullet']
    bullet_libs = ['/usr/local/lib', '/usr/lib', '/usr/lib/x86_64-linux-gnu/']
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

    # DART requires OSG library for their graphic version
    osg_include = []
    osg_lib = []
    osg_check = ['/usr/local/include', '/usr/include']
    osg_libs = ['/usr/local/lib', '/usr/lib', '/usr/lib/x86_64-linux-gnu']
    osg_found = False
    osg_comp = ['osg', 'osgViewer', 'osgManipulator', 'osgGA', 'osgDB', 'osgShadow', 'OpenThreads']
    try:
        for f in osg_comp:
            osg_include = [get_directory(f + '/Version', osg_check)]
            osg_lib = [get_directory('lib' + f + '.' + suffix, osg_libs)]
            osg_found = True
        osg_include = list(set(osg_include))
        osg_lib = list(set(osg_lib))
    except:
        osg_found = False

    dart_load_prefix = 'utils'
    dart_include = []
    dart_major = -1
    dart_minor = -1
    dart_patch = -1

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
            if dart_major > 0 and dart_minor > 0  and dart_patch > 0:
                break

        if dart_major < 6:
            raise Exception('We need DART version at least 6.0.0')
        if dart_major > 6:
            dart_load_prefix = 'io'

        dart_include = []
        dart_include.append(get_directory('dart/dart.hpp', includes_check))
        dart_include.append(get_directory('dart/'+dart_load_prefix+'/'+dart_load_prefix+'.hpp', includes_check))
        dart_include.append(get_directory('dart/'+dart_load_prefix+'/urdf/urdf.hpp', includes_check))
        dart_include = list(set(dart_include))
        conf.end_msg(str(dart_major)+'.'+str(dart_minor)+'.'+str(dart_patch)+' in '+dart_include[0])

        gui_include = []
        try:
            conf.start_msg('Checking for DART gui includes')
            gui_include.append(get_directory('dart/gui/gui.hpp', includes_check))
            gui_include.append(get_directory('dart/gui/osg/osg.hpp', includes_check))
            gui_include = list(set(gui_include))
            conf.end_msg(gui_include[0])
        except:
            conf.end_msg('Not found', 'RED')

        more_includes = []
        if osg_found:
            more_includes += osg_include
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
        conf.end_msg(conf.env.LIB_DART)
        conf.start_msg('DART: Checking for Assimp')
        if assimp_found:
            conf.end_msg(assimp_include)
            conf.env.LIBPATH_DART = conf.env.LIBPATH_DART + assimp_lib
            conf.env.LIB_DART.append('assimp')
        else:
            conf.end_msg('Not found - Your programs may not compile', 'RED')

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
                conf.end_msg('Not found', 'RED')
        else:
            conf.end_msg('Not found', 'RED')

        # remove duplicates
        conf.env.INCLUDES_DART = list(set(conf.env.INCLUDES_DART))
        conf.env.LIBPATH_DART = list(set(conf.env.LIBPATH_DART))

        try:
            conf.start_msg('DART: Checking for gui libs')
            dart_gui_lib = []
            dart_gui_lib.append(get_directory('libdart-gui.' + suffix, libs_check))
            dart_gui_lib.append(get_directory('libdart-gui-osg.' + suffix, libs_check))

            conf.env.INCLUDES_DART_GRAPHIC = deepcopy(conf.env.INCLUDES_DART)
            conf.env.LIBPATH_DART_GRAPHIC = deepcopy(conf.env.LIBPATH_DART) + dart_gui_lib
            conf.env.LIB_DART_GRAPHIC = deepcopy(conf.env.LIB_DART) + ['dart-gui', 'dart-gui-osg']
            conf.end_msg(conf.env.LIB_DART_GRAPHIC[-2:])
            conf.start_msg('DART: Checking for OSG (optional)')
            if osg_found:
                conf.env.INCLUDES_DART_GRAPHIC += osg_include
                conf.env.LIBPATH_DART_GRAPHIC += osg_lib
                conf.env.LIB_DART_GRAPHIC += osg_comp
                conf.end_msg(osg_comp)
            else:
                conf.end_msg('Not found - Your graphical programs may not compile/link', 'RED')
            conf.get_env()['BUILD_GRAPHIC'] = True

            # remove duplicates
            conf.env.INCLUDES_DART_GRAPHIC = list(set(conf.env.INCLUDES_DART_GRAPHIC))
            conf.env.LIBPATH_DART_GRAPHIC = list(set(conf.env.LIBPATH_DART_GRAPHIC))
        except:
            conf.end_msg('Not found', 'RED')
    except:
        if dart_major < 6 and dart_major > -1:
            fail('We need DART >= 6.0.0', required)
        else:
            fail('Not found', required)
        return
    return 1
