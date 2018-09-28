#! /usr/bin/env python
# encoding: utf-8
# Konstantinos Chatzilygeroudis - 2015

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
        if dart_major > 6 or (dart_major == 6 and dart_minor > 4):
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
