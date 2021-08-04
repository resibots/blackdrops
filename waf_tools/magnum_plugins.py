#! /usr/bin/env python
# encoding: utf-8
# Konstantinos Chatzilygeroudis - 2018

"""
Quick n dirty MagnumPlugins detection
"""

# this tools should be used for static plugin detection
# NOT AVAILABLE YET; use this file only for sanity check
# i.e. to check if a plugin is installed

import os
import re
from waflib import Utils, Logs
from waflib.Configure import conf
import copy

import magnum

def options(opt):
    opt.add_option('--magnum_plugins_install_dir', type='string', help='path to magnum plugins install directory', dest='magnum_plugins_install_dir')

def get_magnum_plugins_components():
    magnum_plugins_components = ['AnyAudioImporter', 'AnyImageConverter', 'AnyImageImporter', 'AnySceneImporter', 'AssimpImporter', 'DdsImporter', 'DevIlImageImporter', 'DrFlacAudioImporter', 'DrWavAudioImporter', 'FreeTypeFont', 'HarfBuzzFont', 'JpegImporter', 'MiniExrImageConverter', 'OpenGexImporter', 'PngImageConverter', 'PngImporter', 'StanfordImporter', 'StbImageConverter', 'StbImageImporter', 'StbTrueTypeFont', 'StbVorbisAudioImporter']

    magnum_plugins_dependencies = {}
    for component in magnum_plugins_components:
        magnum_plugins_dependencies[component] = []
    magnum_plugins_dependencies['AssimpImporter'] = ['AnyImageImporter']
    magnum_plugins_dependencies['OpenGexImporter'] = ['AnyImageImporter']
    magnum_plugins_dependencies['HarfBuzzFont'] = ['FreeTypeFont']

    pat_audio = re.compile('.+AudioImporter$')
    pat_all_fonts = re.compile('.+(Font|FontConverter)$')
    magnum_plugins_magnum_dependencies = {}
    for component in magnum_plugins_components:
        magnum_plugins_magnum_dependencies[component] = ['Magnum']
        if re.match(pat_audio, component):
            magnum_plugins_magnum_dependencies[component] += ['Audio']
        elif re.match(pat_all_fonts, component):
            magnum_plugins_magnum_dependencies[component] += ['Text']

    return magnum_plugins_components, magnum_plugins_dependencies, magnum_plugins_magnum_dependencies

def get_magnum_plugins_dependency_libs(bld, components, magnum_plugins_var = 'MagnumPlugins', magnum_var = 'Magnum', corrade_var = 'Corrade'):
    magnum_plugins_components, magnum_plugins_dependencies, magnum_plugins_magnum_dependencies = get_magnum_plugins_components()

    # only check for components that can exist
    requested_components = list(set(components.split()).intersection(magnum_plugins_components))
    # add dependencies
    for lib in requested_components:
        requested_components = requested_components + magnum_plugins_dependencies[lib]
    # remove duplicates
    requested_components = list(set(requested_components))

    # first sanity checks
    # Check if requested components are found
    for component in requested_components:
        if not bld.env['INCLUDES_%s_%s' % (magnum_plugins_var, component)]:
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
            if dep in magnum_plugins_dependencies[component]:
                break

        sorted_components.insert(k, component)

    sorted_components = [magnum_plugins_var+'_'+c for c in sorted_components]

    magnum_components = ''
    for component in requested_components:
        for lib in magnum_plugins_magnum_dependencies[component]:
            magnum_components = magnum_components + ' ' + lib

    return ' '.join(sorted_components) + ' ' + magnum.get_magnum_dependency_libs(bld, magnum_components, magnum_var, corrade_var)

@conf
def check_magnum_plugins(conf, *k, **kw):
    def get_directory(filename, dirs, full = False):
        res = conf.find_file(filename, dirs)
        if not full:
            return res[:-len(filename)-1]
        return res[:res.rfind('/')]
    def find_in_string(data, text):
        return data.find(text)

    # Check compiler version (for gcc); I am being a bit more strong (Magnum could be built with 4.7 but needs adjustment)
    if conf.env.CXX_NAME in ["gcc", "g++"] and int(conf.env['CC_VERSION'][0]+conf.env['CC_VERSION'][1]) < 48:
        msg = 'MagnumPlugins cannot be setup with GCC < 4.8!'
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
    if conf.options.magnum_plugins_install_dir:
        includes_check = [conf.options.magnum_plugins_install_dir + '/include'] + includes_check
        libs_check = [conf.options.magnum_plugins_install_dir + '/lib'] + libs_check
        bins_check = [conf.options.magnum_plugins_install_dir + '/bin'] + bins_check

    # OSX/Mac uses .dylib and GNU/Linux .so
    suffix = 'dylib' if conf.env['DEST_OS'] == 'darwin' else 'so'
    modules_suffix = 'so'

    required = kw.get('required', False)
    requested_components = kw.get('components', None)
    if requested_components == None:
        requested_components = []
    else:
        requested_components = requested_components.split()

    magnum_var = kw.get('magnum', 'Magnum')

    # MagnumPlugins require Magnum
    if not conf.env['INCLUDES_%s' % magnum_var]:
        msg = 'Magnum needs to be configured! Cannot proceed!'
        if required:
            conf.fatal(msg)
        Logs.pprint('RED', msg)
        return

    magnum_plugins_var = kw.get('uselib_store', 'MagnumPlugins')
    # to-do: enforce C++11/14

    magnum_plugins_components, magnum_plugins_dependencies, magnum_plugins_magnum_dependencies = get_magnum_plugins_components()

    # magnum_plugins_includes = copy.deepcopy(conf.env['INCLUDES_%s_Magnum' % magnum_var])
    # magnum_plugins_libpaths = copy.deepcopy(conf.env['LIBPATH_%s_Magnum' % magnum_var])
    # magnum_plugins_libs = copy.deepcopy(conf.env['LIB_%s_Magnum' % magnum_var])
    magnum_plugins_includes = []
    magnum_plugins_libpaths = []
    magnum_plugins_libs = []

    magnum_plugins_component_includes = {}
    magnum_plugins_component_libpaths = {}
    magnum_plugins_component_libs = {}

    # only check for components that can exist
    requested_components = list(set(requested_components).intersection(magnum_plugins_components))
    # add dependencies
    for lib in requested_components:
        requested_components = requested_components + magnum_plugins_dependencies[lib]
    # remove duplicates
    requested_components = list(set(requested_components))

    pat_audio = re.compile('.+AudioImporter$')
    pat_importer = re.compile('.+Importer$')
    pat_font = re.compile('.+Font$')
    pat_img_conv = re.compile('.+ImageConverter$')
    pat_font_conv = re.compile('.+FontConverter$')
    pat_all_fonts = re.compile('.+(Font|FontConverter)$')

    for component in requested_components:
        conf.start_msg('Checking for ' + component + ' Magnum Plugin')
        # magnum_plugins_component_includes[component] = copy.deepcopy(conf.env['INCLUDES_%s_Magnum' % magnum_var])
        # magnum_plugins_component_libpaths[component] = copy.deepcopy(conf.env['LIBPATH_%s_Magnum' % magnum_var])
        # magnum_plugins_component_libs[component] = copy.deepcopy(conf.env['LIB_%s_Magnum' % magnum_var])
        magnum_plugins_component_includes[component] = []
        magnum_plugins_component_libpaths[component] = []
        magnum_plugins_component_libs[component] = []

        lib_path_suffix = ''
        component_file = component

        if re.match(pat_audio, component):
            lib_path_suffix = 'audioimporters'
            component_file = component.replace("AudioImporter", "Importer")

            # check if Magnum Audio is available
            if not conf.env['INCLUDES_%s_Audio' % magnum_var]:
                msg = 'AudioImporters require Magnum Audio! Cannot proceed!'
                if required:
                    conf.fatal(msg)
                Logs.pprint('RED', msg)
                return
            # add includes/paths/libs
            # magnum_plugins_includes = magnum_plugins_includes + conf.env['INCLUDES_%s_Audio' % magnum_var]
            # magnum_plugins_libpaths = magnum_plugins_libpaths + conf.env['LIBPATH_%s_Audio' % magnum_var]
            # magnum_plugins_libs = magnum_plugins_libs + conf.env['LIB_%s_Audio' % magnum_var]

            # magnum_plugins_component_includes[component] = magnum_plugins_component_includes[component] + conf.env['INCLUDES_%s_Audio' % magnum_var]
            # magnum_plugins_component_libpaths[component] = magnum_plugins_component_libpaths[component] + conf.env['LIBPATH_%s_Audio' % magnum_var]
            # magnum_plugins_component_libs[component] = magnum_plugins_component_libs[component] + conf.env['LIB_%s_Audio' % magnum_var]
        elif re.match(pat_importer, component):
            lib_path_suffix = 'importers'
        elif re.match(pat_font, component):
            lib_path_suffix = 'fonts'
        elif re.match(pat_img_conv, component):
            lib_path_suffix = 'imageconverters'
        elif re.match(pat_font_conv, component):
            lib_path_suffix = 'fontconverters'

        if re.match(pat_all_fonts, component):
            # check if Magnum Text is available
            if not conf.env['INCLUDES_%s_Text' % magnum_var]:
                msg = 'Font and FontConverter plugins require Magnum Text! Cannot proceed!'
                if required:
                    conf.fatal(msg)
                Logs.pprint('RED', msg)
                return
            # add includes/paths/libs
            # magnum_plugins_includes = magnum_plugins_includes + conf.env['INCLUDES_%s_Text' % magnum_var]
            # magnum_plugins_libpaths = magnum_plugins_libpaths + conf.env['LIBPATH_%s_Text' % magnum_var]
            # magnum_plugins_libs = magnum_plugins_libs + conf.env['LIB_%s_Text' % magnum_var]

            # magnum_plugins_component_includes[component] = magnum_plugins_component_includes[component] + conf.env['INCLUDES_%s_Text' % magnum_var]
            # magnum_plugins_component_libpaths[component] = magnum_plugins_component_libpaths[component] + conf.env['LIBPATH_%s_Text' % magnum_var]
            # magnum_plugins_component_libs[component] = magnum_plugins_component_libs[component] + conf.env['LIB_%s_Text' % magnum_var]

        if lib_path_suffix != '':
            lib_path_suffix = lib_path_suffix + '/'

        try:
            include_dir = get_directory('MagnumPlugins/'+component+'/'+component_file+'.h', includes_check)
            magnum_plugins_includes = magnum_plugins_includes + [include_dir]
            lib = component
            # we need the full lib_dir in order to be able to link to the plugins
            # or not? because they are loaded dynamically
            lib_dir = get_directory('magnum/'+lib_path_suffix+lib+'.'+modules_suffix, libs_check, True)
            # magnum_plugins_libs.append(lib)
            # magnum_plugins_libpaths = magnum_plugins_libpaths + [lib_dir]

            magnum_plugins_component_includes[component] = magnum_plugins_component_includes[component] + [include_dir]
            # magnum_plugins_component_libpaths[component] = magnum_plugins_component_libpaths[component] + [lib_dir]
            # magnum_plugins_component_libs[component].append(lib)
        except:
            if required:
                conf.fatal('Not found')
            conf.end_msg('Not found', 'RED')
            # if optional, continue?
            continue
        conf.end_msg(include_dir)

        # extra dependencies
        if component == 'AssimpImporter':
            # AssimpImporter requires Assimp
            conf.start_msg(component + ': Checking for Assimp')
            try:
                assimp_inc = get_directory('assimp/anim.h', includes_check)

                magnum_plugins_includes = magnum_plugins_includes + [assimp_inc]
                magnum_plugins_component_includes[component] = magnum_plugins_component_includes[component] + [assimp_inc]

                lib_dir = get_directory('libassimp.'+suffix, libs_check)
                magnum_plugins_libpaths = magnum_plugins_libpaths + [lib_dir]
                magnum_plugins_libs.append('assimp')

                magnum_plugins_component_libpaths[component] = magnum_plugins_component_libpaths[component] + [lib_dir]
                magnum_plugins_component_libs[component].append('assimp')
            except:
                if required:
                    conf.fatal('Not found')
                conf.end_msg('Not found', 'RED')
                # if optional, continue?
                continue
            conf.end_msg(assimp_inc)
        elif component == 'DevIlImageImporter':
            # DevIlImageImporter requires DevIl
            msg = component + ' is not supported with WAF'
            if required:
                conf.fatal(msg)
            conf.end_msg(msg, 'RED')
        elif component == 'FreeTypeFont':
            # FreeTypeFont requires FreeType2?
            conf.start_msg(component + ': Checking for FreeType')
            try:
                freetype_inc = get_directory('freetype/ft2build.h', includes_check, True)

                magnum_plugins_includes = magnum_plugins_includes + [freetype_inc]
                magnum_plugins_component_includes[component] = magnum_plugins_component_includes[component] + [freetype_inc]

                lib_dir = get_directory('libfreetype.'+suffix, libs_check)
                magnum_plugins_libpaths = magnum_plugins_libpaths + [lib_dir]
                magnum_plugins_libs.append('freetype')

                magnum_plugins_component_libpaths[component] = magnum_plugins_component_libpaths[component] + [lib_dir]
                magnum_plugins_component_libs[component].append('freetype')
            except:
                if required:
                    conf.fatal('Not found')
                conf.end_msg('Not found', 'RED')
                # if optional, continue?
                continue
            conf.end_msg(freetype_inc)
        elif component == 'HarfBuzzFont':
            # HarfBuzzFont requires FreeType and HarfBuzz
            msg = component + ' is not supported with WAF'
            if required:
                conf.fatal(msg)
            conf.end_msg(msg, 'RED')
        elif component == 'JpegImporter':
            # JpegImporter requires JPEG
            conf.start_msg(component + ': Checking for JPEG')
            try:
                jpeg_inc = get_directory('jpeglib.h', includes_check)

                magnum_plugins_includes = magnum_plugins_includes + [jpeg_inc]
                magnum_plugins_component_includes[component] = magnum_plugins_component_includes[component] + [jpeg_inc]

                lib_dir = get_directory('libjpeg.'+suffix, libs_check)
                magnum_plugins_libpaths = magnum_plugins_libpaths + [lib_dir]
                magnum_plugins_libs.append('jpeg')

                magnum_plugins_component_libpaths[component] = magnum_plugins_component_libpaths[component] + [lib_dir]
                magnum_plugins_component_libs[component].append('jpeg')
            except:
                if required:
                    conf.fatal('Not found')
                conf.end_msg('Not found', 'RED')
                # if optional, continue?
                continue
            conf.end_msg(jpeg_inc)
        elif component == 'PngImageConverter' or component == 'PngImporter':
            # PngImageConverter and PngImporter require PNG
            conf.start_msg(component + ': Checking for PNG')
            try:
                png_inc = get_directory('png.h', includes_check)

                magnum_plugins_includes = magnum_plugins_includes + [png_inc]
                magnum_plugins_component_includes[component] = magnum_plugins_component_includes[component] + [png_inc]

                lib_dir = get_directory('libpng.'+suffix, libs_check)
                magnum_plugins_libpaths = magnum_plugins_libpaths + [lib_dir]
                magnum_plugins_libs.append('png')

                magnum_plugins_component_libpaths[component] = magnum_plugins_component_libpaths[component] + [lib_dir]
                magnum_plugins_component_libs[component].append('png')
            except:
                if required:
                    conf.fatal('Not found')
                conf.end_msg('Not found', 'RED')
                # if optional, continue?
                continue
            conf.end_msg(png_inc)

    if len(magnum_plugins_libs) > 0:
        conf.start_msg(magnum_plugins_var + ' libs:')
        conf.end_msg(magnum_plugins_libs)

    # remove duplicates
    magnum_plugins_includes = list(set(magnum_plugins_includes))
    magnum_plugins_libpaths = list(set(magnum_plugins_libpaths))

    # set environmental variables
    conf.env['INCLUDES_%s' % magnum_plugins_var] = magnum_plugins_includes
    conf.env['LIBPATH_%s' % magnum_plugins_var] = magnum_plugins_libpaths
    conf.env['LIB_%s' % magnum_plugins_var] = magnum_plugins_libs
    conf.env['DEFINES_%s' % magnum_plugins_var] = copy.deepcopy(conf.env['DEFINES_%s' % magnum_var])

    # set component libs
    for component in requested_components:
        # for lib in magnum_plugins_dependencies[component]:
        #     magnum_plugins_component_includes[component] = magnum_plugins_component_includes[component] + magnum_plugins_component_includes[lib]
        #     magnum_plugins_component_libpaths[component] = magnum_plugins_component_libpaths[component] + magnum_plugins_component_libpaths[lib]
        #     magnum_plugins_component_libs[component] = magnum_plugins_component_libs[component] + magnum_plugins_component_libs[lib]

        conf.env['INCLUDES_%s_%s' % (magnum_plugins_var, component)] = list(set(magnum_plugins_component_includes[component]))
        conf.env['LIBPATH_%s_%s' % (magnum_plugins_var, component)] = list(set(magnum_plugins_component_libpaths[component]))
        conf.env['LIB_%s_%s' % (magnum_plugins_var, component)] = list(set(magnum_plugins_component_libs[component]))

        # copy the C++ defines; we want them to be available on all Magnum builds
        conf.env['DEFINES_%s_%s' % (magnum_plugins_var, component)] = copy.deepcopy(conf.env['DEFINES_%s' % magnum_plugins_var])
    # set C++ flags
    conf.env['CXX_FLAGS_%s' % magnum_plugins_var] = copy.deepcopy(conf.env['CXX_FLAGS_%s' % magnum_var])