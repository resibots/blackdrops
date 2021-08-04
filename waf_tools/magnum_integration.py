#! /usr/bin/env python
# encoding: utf-8
# Konstantinos Chatzilygeroudis - 2018

"""
Quick n dirty MagnumIntegration detection
"""

# This is not checking the extra dependencies
# nor supports all the integrations

import os
import re
from waflib import Utils, Logs
from waflib.Configure import conf
import copy

import magnum

def options(opt):
    opt.add_option('--magnum_integration_install_dir', type='string', help='path to magnum plugins install directory', dest='magnum_integration_install_dir')

def get_magnum_integration_components():
    magnum_integration_components = ['Bullet', 'Dart', 'Eigen']

    magnum_integration_dependencies = {}
    for component in magnum_integration_components:
        magnum_integration_dependencies[component] = []

    magnum_integration_magnum_dependencies = {}
    for component in magnum_integration_components:
        magnum_integration_magnum_dependencies[component] = ['Magnum']
        if component == 'Bullet':
            magnum_integration_magnum_dependencies[component] += ['SceneGraph', 'Shaders']
        elif component == 'Dart':
            magnum_integration_magnum_dependencies[component] += ['SceneGraph', 'Primitives', 'MeshTools']

    return magnum_integration_components, magnum_integration_dependencies, magnum_integration_magnum_dependencies

def get_magnum_integration_dependency_libs(bld, components, magnum_integration_var = 'MagnumIntegration', magnum_var = 'Magnum', corrade_var = 'Corrade'):
    magnum_integration_components, magnum_integration_dependencies, magnum_integration_magnum_dependencies = get_magnum_integration_components()

    # only check for components that can exist
    requested_components = list(set(components.split()).intersection(magnum_integration_components))
    # add dependencies
    for lib in requested_components:
        requested_components = requested_components + magnum_integration_dependencies[lib]
    # remove duplicates
    requested_components = list(set(requested_components))

    # first sanity checks
    # Check if requested components are found
    for component in requested_components:
        if not bld.env['INCLUDES_%s_%s' % (magnum_integration_var, component)]:
            bld.fatal('%s was not found! Cannot proceed!' % component)

    # now get the libs in correct order
    sorted_components = [requested_components[0]]
    for i in range(len(requested_components)):
        component = requested_components[i]
        if component in sorted_components:
            continue
        k = 0
        for j in range(len(sorted_components)):
            k = j
            dep = sorted_components[j]
            if dep in magnum_integration_dependencies[component]:
                break

        sorted_components.insert(k, component)

    sorted_components = [magnum_integration_var+'_'+c for c in sorted_components]

    magnum_components = ''
    for component in requested_components:
        for lib in magnum_integration_magnum_dependencies[component]:
            magnum_components = magnum_components + ' ' + lib

    return ' '.join(sorted_components) + ' ' + magnum.get_magnum_dependency_libs(bld, magnum_components, magnum_var, corrade_var)

@conf
def check_magnum_integration(conf, *k, **kw):
    def get_directory(filename, dirs, full = False):
        res = conf.find_file(filename, dirs)
        if not full:
            return res[:-len(filename)-1]
        return res[:res.rfind('/')]
    def find_in_string(data, text):
        return data.find(text)

    # Check compiler version (for gcc); I am being a bit more strong (Magnum could be built with 4.7 but needs adjustment)
    if conf.env.CXX_NAME in ["gcc", "g++"] and int(conf.env['CC_VERSION'][0]+conf.env['CC_VERSION'][1]) < 48:
        msg = 'MagnumIntegration cannot be setup with GCC < 4.8!'
        if required:
            conf.fatal(msg)
        Logs.pprint('RED', msg)
        return

    includes_check = ['/usr/local/include', '/usr/include', '/opt/local/include', '/sw/include']
    libs_check = ['/usr/lib', '/usr/local/lib', '/opt/local/lib', '/sw/lib', '/lib', '/usr/lib/x86_64-linux-gnu/', '/usr/lib64']
    bins_check = ['/usr/bin', '/usr/local/bin', '/opt/local/bin', '/sw/bin', '/bin']

    # Magnum depends on several libraries and we cannot make the assumption that
    # someone installed all of them in the same directory!
    # to-do: a better? solution would be to create different scripts for each dependency
    if conf.options.magnum_integration_install_dir:
        includes_check = [conf.options.magnum_integration_install_dir + '/include'] + includes_check
        libs_check = [conf.options.magnum_integration_install_dir + '/lib'] + libs_check
        bins_check = [conf.options.magnum_integration_install_dir + '/bin'] + bins_check

    # OSX/Mac uses .dylib and GNU/Linux .so
    suffix = 'dylib' if conf.env['DEST_OS'] == 'darwin' else 'so'

    required = kw.get('required', False)
    requested_components = kw.get('components', None)
    if requested_components == None:
        requested_components = []
    else:
        requested_components = requested_components.split()

    magnum_var = kw.get('magnum', 'Magnum')

    # MagnumIntegration require Magnum
    if not conf.env['INCLUDES_%s' % magnum_var]:
        msg = 'Magnum needs to be configured! Cannot proceed!'
        if required:
            conf.fatal(msg)
        Logs.pprint('RED', msg)
        return

    magnum_integration_var = kw.get('uselib_store', 'MagnumIntegration')
    # to-do: enforce C++11/14

    magnum_integration_components, magnum_integration_dependencies, magnum_integration_magnum_dependencies = get_magnum_integration_components()

    # magnum_integration_includes = copy.deepcopy(conf.env['INCLUDES_%s_Magnum' % magnum_var])
    # magnum_integration_libpaths = copy.deepcopy(conf.env['LIBPATH_%s_Magnum' % magnum_var])
    # magnum_integration_libs = copy.deepcopy(conf.env['LIB_%s_Magnum' % magnum_var])
    magnum_integration_includes = []
    magnum_integration_libpaths = []
    magnum_integration_libs = []

    magnum_integration_component_includes = {}
    magnum_integration_component_libpaths = {}
    magnum_integration_component_libs = {}

    # only check for components that can exist
    requested_components = list(set(requested_components).intersection(magnum_integration_components))
    # add dependencies
    for lib in requested_components:
        requested_components = requested_components + magnum_integration_dependencies[lib]
    # remove duplicates
    requested_components = list(set(requested_components))

    for component in requested_components:
        conf.start_msg('Checking for ' + component + ' Magnum Integration')
        # magnum_integration_component_includes[component] = copy.deepcopy(conf.env['INCLUDES_%s_Magnum' % magnum_var])
        # magnum_integration_component_libpaths[component] = copy.deepcopy(conf.env['LIBPATH_%s_Magnum' % magnum_var])
        # magnum_integration_component_libs[component] = copy.deepcopy(conf.env['LIB_%s_Magnum' % magnum_var])
        magnum_integration_component_includes[component] = []
        magnum_integration_component_libpaths[component] = []
        magnum_integration_component_libs[component] = []

        component_name = component
        component_file = 'Integration'

        if component == 'Bullet':
            component_name = 'BulletIntegration'
        elif component == 'Dart':
            component_name = 'DartIntegration'
            component_file = 'DartIntegration'
        elif component == 'Eigen':
            component_name = 'EigenIntegration'

        try:
            include_dir = get_directory('Magnum/'+component_name+'/'+component_file+'.h', includes_check)
            magnum_integration_includes = magnum_integration_includes + [include_dir]

            magnum_integration_component_includes[component] = magnum_integration_component_includes[component] + [include_dir]
            if component != 'Eigen':
                lib = 'Magnum' + component_name
                lib_dir = get_directory('lib'+lib+'.'+suffix, libs_check, True)
                magnum_integration_libs.append(lib)
                magnum_integration_libpaths = magnum_integration_libpaths + [lib_dir]

                magnum_integration_component_libpaths[component] = magnum_integration_component_libpaths[component] + [lib_dir]
                magnum_integration_component_libs[component].append(lib)
        except:
            if required:
                conf.fatal('Not found')
            conf.end_msg('Not found', 'RED')
            # if optional, continue?
            continue
        conf.end_msg(include_dir)

        # extra dependencies
        # to-do: check for bullet and Dart
        # if component == 'AssimpImporter':
        #     # AssimpImporter requires Assimp
        #     conf.start_msg(component + ': Checking for Assimp')
        #     try:
        #         assimp_inc = get_directory('assimp/anim.h', includes_check)

        #         magnum_integration_includes = magnum_integration_includes + [assimp_inc]
        #         magnum_integration_component_includes[component] = magnum_integration_component_includes[component] + [assimp_inc]

        #         lib_dir = get_directory('libassimp.'+suffix, libs_check)
        #         magnum_integration_libpaths = magnum_integration_libpaths + [lib_dir]
        #         magnum_integration_libs.append('assimp')

        #         magnum_integration_component_libpaths[component] = magnum_integration_component_libpaths[component] + [lib_dir]
        #         magnum_integration_component_libs[component].append('assimp')
        #     except:
        #         if required:
        #             conf.fatal('Not found')
        #         conf.end_msg('Not found', 'RED')
        #         # if optional, continue?
        #         continue
        #     conf.end_msg(assimp_inc)

    if len(magnum_integration_libs) > 0:
        conf.start_msg(magnum_integration_var + ' libs:')
        conf.end_msg(magnum_integration_libs)

    # remove duplicates
    magnum_integration_includes = list(set(magnum_integration_includes))
    magnum_integration_libpaths = list(set(magnum_integration_libpaths))

    # set environmental variables
    conf.env['INCLUDES_%s' % magnum_integration_var] = magnum_integration_includes
    conf.env['LIBPATH_%s' % magnum_integration_var] = magnum_integration_libpaths
    conf.env['LIB_%s' % magnum_integration_var] = magnum_integration_libs
    conf.env['DEFINES_%s' % magnum_integration_var] = copy.deepcopy(conf.env['DEFINES_%s' % magnum_var])

    # set component libs
    for component in requested_components:
        for lib in magnum_integration_dependencies[component]:
            magnum_integration_component_includes[component] = magnum_integration_component_includes[component] + magnum_integration_component_includes[lib]
            magnum_integration_component_libpaths[component] = magnum_integration_component_libpaths[component] + magnum_integration_component_libpaths[lib]
            magnum_integration_component_libs[component] = magnum_integration_component_libs[component] + magnum_integration_component_libs[lib]

        conf.env['INCLUDES_%s_%s' % (magnum_integration_var, component)] = list(set(magnum_integration_component_includes[component]))
        if len(magnum_integration_component_libs[component]) > 0:
            conf.env['LIBPATH_%s_%s' % (magnum_integration_var, component)] = list(set(magnum_integration_component_libpaths[component]))
            conf.env['LIB_%s_%s' % (magnum_integration_var, component)] = list(set(magnum_integration_component_libs[component]))

        # copy the C++ defines; we want them to be available on all Magnum builds
        conf.env['DEFINES_%s_%s' % (magnum_integration_var, component)] = copy.deepcopy(conf.env['DEFINES_%s' % magnum_integration_var])
    # set C++ flags
    conf.env['CXX_FLAGS_%s' % magnum_integration_var] = copy.deepcopy(conf.env['CXX_FLAGS_%s' % magnum_var])