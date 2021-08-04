#! /usr/bin/env python
# encoding: utf-8
# Konstantinos Chatzilygeroudis - 2018

"""
Quick n dirty Magnum detection
"""

import os
import io
import re
from waflib import Utils, Logs
from waflib.Configure import conf
import copy

def options(opt):
        opt.add_option('--magnum_install_dir', type='string', help='path to magnum install directory', dest='magnum_install_dir')

def get_magnum_components():
    magnum_components = ['Audio', 'DebugTools', 'MeshTools', 'Primitives', 'SceneGraph', 'Shaders', 'Shapes', 'Text', 'TextureTools', 'Trade', 'GlfwApplication', 'GlutApplication', 'GlxApplication', 'Sdl2Application', 'XEglApplication', 'WindowlessCglApplication', 'WindowlessEglApplication', 'WindowlessGlxApplication', 'WindowlessIosApplication', 'WindowlessWglApplication', 'WindowlessWindowsEglApplicatio', 'CglContext', 'EglContext', 'GlxContext', 'WglContext', 'OpenGLTester', 'MagnumFont', 'MagnumFontConverter', 'ObjImporter', 'TgaImageConverter', 'TgaImporter', 'WavAudioImporter', 'distancefieldconverter', 'fontconverter', 'imageconverter', 'info', 'al-info']
    magnum_dependencies = {}
    for component in magnum_components:
        magnum_dependencies[component] = []
    magnum_dependencies['Shapes'] = ['SceneGraph']
    magnum_dependencies['Text'] = ['TextureTools']
    magnum_dependencies['DebugTools'] = ['MeshTools', 'Primitives', 'SceneGraph', 'Shaders']
    magnum_dependencies['Primitives'] = ['Trade']
    # to-do: OpenGLTester deps should be defined after the configurations have been detected
    magnum_dependencies['MagnumFont'] = ['TgaImporter', 'Text', 'TextureTools']
    magnum_dependencies['MagnumFontConverter'] = ['TgaImageConverter', 'Text', 'TextureTools']
    magnum_dependencies['ObjImporter'] = ['MeshTools']
    magnum_dependencies['WavAudioImporter'] = ['Audio']

    pat_lib = re.compile('^(Audio|DebugTools|MeshTools|Primitives|SceneGraph|Shaders|Shapes|Text|TextureTools|Trade|AndroidApplication|GlfwApplication|GlutApplication|GlxApplication|Sdl2Application|XEglApplication|WindowlessCglApplication|WindowlessEglApplication|WindowlessGlxApplication|WindowlessIosApplication|WindowlessWglApplication|WindowlessWindowsEglApplication|CglContext|EglContext|GlxContext|WglContext|OpenGLTester)$')
    pat_plugin = re.compile('^(MagnumFont|MagnumFontConverter|ObjImporter|TgaImageConverter|TgaImporter|WavAudioImporter)$')
    pat_bin = re.compile('^(distancefieldconverter|fontconverter|imageconverter|info|al-info)$')
    magnum_component_type = {}
    for component in magnum_components:
        magnum_component_type[component] = ''
        if re.match(pat_lib, component):
            magnum_component_type[component] = 'lib'
        if re.match(pat_plugin, component):
            magnum_component_type[component] = 'plugin'
        if re.match(pat_bin, component):
            magnum_component_type[component] = 'bin'

    return copy.deepcopy(magnum_components), copy.deepcopy(magnum_component_type), copy.deepcopy(magnum_dependencies)

def get_magnum_dependency_libs(bld, components, magnum_var = 'Magnum', corrade_var = 'Corrade'):
    magnum_components, magnum_component_type, magnum_dependencies = get_magnum_components()

    # only check for components that can exist
    requested_components = list(set(components.split()).intersection(magnum_components))
    # add dependencies
    for lib in requested_components:
        requested_components = requested_components + magnum_dependencies[lib]
    # remove duplicates
    requested_components = list(set(requested_components))
    # remove non-lib components
    requested_components = [c for c in requested_components if magnum_component_type[c] == 'lib']

    # first sanity checks
    # Magnum requires Corrade
    if not bld.env['INCLUDES_%s' % corrade_var]:
        bld.fatal('Magnum requires Corrade! Cannot proceed!')
    if not bld.env['INCLUDES_%s_Utility' % corrade_var]:
        bld.fatal('Magnum requires Corrade Utility library! Cannot proceed!')
    if not bld.env['INCLUDES_%s_PluginManager' % corrade_var]:
        bld.fatal('Magnum requires Corrade PluginManager library! Cannot proceed!')

    # Check if requested components are found
    for component in requested_components:
        if not bld.env['INCLUDES_%s_%s' % (magnum_var, component)]:
            bld.fatal('%s was not found! Cannot proceed!' % component)

    sorted_components = []
    if len(requested_components) > 0:
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
                if dep in magnum_dependencies[component]:
                    break

            sorted_components.insert(k, component)

        sorted_components = [magnum_var+'_'+c for c in sorted_components]

    # Corrade PluginManager will link to all the inner-dependencies in Corrade in the correct order
    return ' '.join(sorted_components) + ' ' + magnum_var + '_Magnum ' + corrade_var + '_PluginManager '

@conf
def check_magnum(conf, *k, **kw):
    def get_directory(filename, dirs, full = False):
        res = conf.find_file(filename, dirs)
        if not full:
            return res[:-len(filename)-1]
        return res[:res.rfind('/')]
    def find_in_string(data, text):
        return data.find(text)
    def fatal(required, msg):
        if required:
            conf.fatal(msg)
        Logs.pprint('RED', msg)

    required = kw.get('required', False)

    # Check compiler version (for gcc); I am being a bit more strong (Magnum could be built with 4.7 but needs adjustment)
    if conf.env.CXX_NAME in ["gcc", "g++"] and int(conf.env['CC_VERSION'][0]+conf.env['CC_VERSION'][1]) < 48:
        fatal(required, 'Magnum cannot be setup with GCC < 4.8!')
        return

    includes_check = ['/usr/local/include', '/usr/include', '/opt/local/include', '/sw/include']
    libs_check = ['/usr/lib', '/usr/local/lib', '/opt/local/lib', '/sw/lib', '/lib', '/usr/lib/x86_64-linux-gnu/', '/usr/lib64']
    bins_check = ['/usr/bin', '/usr/local/bin', '/opt/local/bin', '/sw/bin', '/bin']
    if conf.env['DEST_OS'] == 'darwin':
        includes_check = includes_check + ['/System/Library/Frameworks/OpenGL.framework/Headers', '/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks/OpenGL.framework/Versions/A/Headers/']
        libs_check = libs_check + ['/System/Library/Frameworks/OpenGL.framework/Libraries', '/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks/OpenGL.framework/Versions/A/Libraries/']

    # OSX/Mac uses .dylib and GNU/Linux .so
    suffix = 'dylib' if conf.env['DEST_OS'] == 'darwin' else 'so'
    modules_suffix = 'so'

    # Magnum depends on several libraries and we cannot make the assumption that
    # someone installed all of them in the same directory!
    # to-do: a better? solution would be to create different scripts for each dependency
    if conf.options.magnum_install_dir:
        includes_check = [conf.options.magnum_install_dir + '/include'] + includes_check
        libs_check = [conf.options.magnum_install_dir + '/lib'] + libs_check
        bins_check = [conf.options.magnum_install_dir + '/bin'] + bins_check

    requested_components = kw.get('components', None)
    if requested_components == None:
        requested_components = []
    else:
        requested_components = requested_components.split()

    corrade_var = kw.get('corrade', 'Corrade')

    # Magnum requires Corrade
    if not conf.env['INCLUDES_%s' % corrade_var]:
        fatal(required, 'Magnum requires Corrade! Cannot proceed!')
        return
    if not conf.env['INCLUDES_%s_Utility' % corrade_var]:
        fatal(required, 'Magnum requires Corrade Utility library! Cannot proceed!')
        return
    if not conf.env['INCLUDES_%s_PluginManager' % corrade_var]:
        fatal(required, 'Magnum requires Corrade PluginManager library! Cannot proceed!')
        return

    magnum_includes = []
    magnum_libpaths = []
    magnum_libs = []
    magnum_bins = []

    magnum_var = kw.get('uselib_store', 'Magnum')
    # to-do: enforce C++11/14

    magnum_possible_configs = ["BUILD_DEPRECATED", "BUILD_STATIC", "BUILD_MULTITHREADED", "TARGET_GL", "TARGET_GLES", "TARGET_GLES2", "TARGET_GLES3", "TARGET_DESKTOP_GLES", "TARGET_WEBGL", "TARGET_HEADLESS"]
    magnum_config = []

    magnum_components, magnum_component_type, magnum_dependencies = get_magnum_components()

    magnum_component_includes = {}
    magnum_component_libpaths = {}
    magnum_component_libs = {}
    magnum_component_bins = {}

    try:
        # to-do: support both debug and release builds
        conf.start_msg('Checking for Magnum includes')
        magnum_include_dir = get_directory('Magnum/Magnum.h', includes_check)
        magnum_includes = magnum_includes + [magnum_include_dir, magnum_include_dir + '/MagnumExternal/OpenGL']
        conf.end_msg(magnum_include_dir)

        conf.start_msg('Checking for Magnum lib')
        magnum_lib_path = get_directory('libMagnum.'+suffix, libs_check)
        magnum_libpaths = magnum_libpaths + [magnum_lib_path]
        magnum_libs = magnum_libs + ['Magnum']
        conf.end_msg(['Magnum'])

        conf.start_msg('Getting Magnum configuration')
        config_file = conf.find_file('Magnum/configure.h', includes_check)
        with io.open(config_file, errors = 'ignore') as f:
            config_content = f.read()
        for config in magnum_possible_configs:
            index = find_in_string(config_content, '#define MAGNUM_' + config)
            if index > -1:
                magnum_config.append(config)
        conf.end_msg(magnum_config)

        if 'TARGET_GL' in magnum_config:
            # to-do: make it work for other platforms; now only for desktop and only for GL
            conf.start_msg('Magnum: Checking for OpenGL includes')
            opengl_files = ['GL/gl.h', 'gl.h']
            gl_not_found = False
            for gl_file in opengl_files:
                try:
                    opengl_include_dir = get_directory(gl_file, includes_check)
                    gl_not_found = False
                    break
                except:
                    gl_not_found = True
            if gl_not_found:
                fatal(required, 'Not found')
                return
            magnum_includes = magnum_includes + [opengl_include_dir]
            conf.end_msg(opengl_include_dir)

            conf.start_msg('Magnum: Checking for OpenGL lib')
            opengl_lib_dir = get_directory('libOpenGL.'+suffix, libs_check)
            magnum_libpaths = magnum_libpaths + [opengl_lib_dir]
            magnum_libs = magnum_libs + ['OpenGL']
            conf.end_msg(['OpenGL'])

            conf.start_msg('Magnum: Checking for MagnumGL lib')
            gl_lib_dir = get_directory('libMagnumGL.'+suffix, libs_check)
            magnum_libpaths = magnum_libpaths + [gl_lib_dir]
            magnum_libs = magnum_libs + ['MagnumGL']
            conf.end_msg(['MagnumGL'])
        else:
            fatal(required, 'At the moment only desktop OpenGL is supported by WAF')
            return

        conf.start_msg('Checking for Magnum components')
        # only check for components that can exist
        requested_components = list(set(requested_components).intersection(magnum_components))
        # add dependencies
        for lib in requested_components:
            requested_components = requested_components + magnum_dependencies[lib]
        # remove duplicates
        requested_components = list(set(requested_components))

        for component in requested_components:
            magnum_component_includes[component] = []
            magnum_component_libpaths[component] = []
            magnum_component_libs[component] = []
            magnum_component_bins[component] = []

            # get component type
            component_type = magnum_component_type[component]
            if component_type == 'lib':
                pat_app = re.compile('.+Application')
                pat_context = re.compile('.+Context')

                component_file = component
                if component == 'MeshTools':
                    component_file = 'CompressIndices'
                if component == 'Primitives':
                    component_file = 'Cube'
                if component == 'TextureTools':
                    component_file = 'Atlas'

                lib_type = suffix
                include_prefix = component
                # Applications
                if re.match(pat_app, component):
                    # to-do: all of them are static?
                    lib_type = 'a'
                    include_prefix = 'Platform'

                include_dir = get_directory('Magnum/'+include_prefix+'/'+component_file+'.h', includes_check)
                lib = 'Magnum'+component
                lib_dir = get_directory('lib'+lib+'.'+lib_type, libs_check)

                magnum_component_includes[component] = magnum_component_includes[component] + [include_dir]
                magnum_component_libpaths[component] = magnum_component_libpaths[component] + [lib_dir]
                magnum_component_libs[component].append(lib)

                # Applications
                if re.match(pat_app, component):
                    if component == 'GlfwApplication':
                        # GlfwApplication requires GLFW3
                        # conf.start_msg('Magnum: Checking for GLFW3 includes')
                        glfw_inc = get_directory('GLFW/glfw3.h', includes_check)

                        magnum_component_includes[component] = magnum_component_includes[component] + [glfw_inc]

                        # conf.start_msg('Magnum: Checking for GLFW3 lib')
                        libs_glfw = ['glfw3', 'glfw']
                        glfw_found = False
                        for lib_glfw in libs_glfw:
                            try:
                                lib_dir = get_directory('lib'+lib_glfw+'.'+suffix, libs_check)
                                glfw_found = True

                                magnum_component_libpaths[component] = magnum_component_libpaths[component] + [lib_dir]
                                magnum_component_libs[component].append(lib_glfw)
                                break
                            except:
                                glfw_found = False

                        # GlfwApplication needs the libGLX.so library
                        try:
                            lib_dir = get_directory('libGLX.'+suffix, libs_check)
                            magnum_component_libpaths[component] = magnum_component_libpaths[component] + [lib_dir]
                            magnum_component_libs[component] = magnum_component_libs[component] + ['GLX']
                        except:
                            glfw_found = False

                        # GlfwApplication needs the libdl.so library
                        try:
                            lib_dir = get_directory('libdl.'+suffix, libs_check)
                            magnum_component_libpaths[component] = magnum_component_libpaths[component] + [lib_dir]
                            magnum_component_libs[component].append('dl')
                        except:
                            glfw_found = False

                        if not glfw_found:
                            fatal(required, 'Not found')
                            return
                    elif component == 'GlutApplication':
                        # GlutApplication requires GLUT
                        # conf.start_msg('Magnum: Checking for GLUT includes')
                        glut_inc = get_directory('GL/freeglut.h', includes_check)

                        magnum_component_includes[component] = magnum_component_includes[component] + [glut_inc]

                        # conf.start_msg('Magnum: Checking for GLUT lib')
                        libs_glut = ['glut', 'glut32']
                        glut_found = False
                        for lib_glut in libs_glut:
                            try:
                                lib_dir = get_directory('lib'+lib_glut+'.'+suffix, libs_check)
                                glut_found = True

                                magnum_component_libpaths[component] = magnum_component_libpaths[component] + [lib_dir]
                                magnum_component_libs[component].append(lib_glut)
                                break
                            except:
                                glut_found = False

                        if not glut_found:
                            fatal(required, 'Not found')
                            return
                    elif component == 'Sdl2Application':
                        # Sdl2Application requires SDL2
                        conf.check_cfg(path='sdl2-config', args='--cflags --libs', package='', uselib_store='MAGNUM_SDL')

                        magnum_component_includes[component] = magnum_component_includes[component] + conf.env['INCLUDES_MAGNUM_SDL']
                        magnum_component_libpaths[component] = magnum_component_libpaths[component] + conf.env['LIBPATH_MAGNUM_SDL']
                        magnum_component_libs[component] = magnum_component_libs[component] + conf.env['LIB_MAGNUM_SDL']
                        # Sdl2Application needs the libdl.so library
                        try:
                            lib_dir = get_directory('libdl.'+suffix, libs_check)
                            magnum_component_libpaths[component] = magnum_component_libpaths[component] + [lib_dir]
                            magnum_component_libs[component].append('dl')
                        except:
                            fatal(required, 'Not found')
                            return
                        # to-do: maybe copy flags?
                    elif component == 'WindowlessEglApplication':
                        # WindowlessEglApplication requires EGL
                        egl_inc = get_directory('EGL/egl.h', includes_check)

                        magnum_component_includes[component] = magnum_component_includes[component] + [egl_inc]

                        libs_egl = ['EGL']
                        egl_found = False
                        for lib_egl in libs_egl:
                            try:
                                lib_dir = get_directory('lib'+lib_egl+'.so', libs_check)
                                egl_found = True

                                magnum_component_libpaths[component] = magnum_component_libpaths[component] + [lib_dir]
                                magnum_component_libs[component].append(lib_egl)
                                break
                            except:
                                egl_found = False

                        if not egl_found:
                            fatal(required, 'Not found')
                            return
                    elif component == 'WindowlessGlxApplication' or component == 'GlxApplication':
                        # [Windowless]GlxApplication requires GLX. X11
                        glx_inc = get_directory('GL/glx.h', includes_check)

                        magnum_component_includes[component] = magnum_component_includes[component] + [glx_inc]

                        libs_glx = ['GLX', 'X11']
                        glx_found = False
                        for lib_glx in libs_glx:
                            try:
                                lib_dir = get_directory('lib'+lib_glx+'.so', libs_check)
                                glx_found = True

                                magnum_component_libpaths[component] = magnum_component_libpaths[component] + [lib_dir]
                                magnum_component_libs[component].append(lib_glx)
                                # break
                            except:
                                glx_found = False

                        if not glx_found:
                            fatal(required, 'Not found')
                            return
                    elif component not in ['WindowlessCglApplication', 'WindowlessWglApplication']:
                        # to-do: support all other applications
                        msg = 'Component ' + component + ' is not yet supported by WAF'
                        fatal(required, msg)
                        return

                if re.match(pat_context, component) and component not in ['CglContext', 'WglContext']:
                    # to-do: support all other contexts
                    msg = 'Component ' + component + ' is not yet supported by WAF'
                    fatal(required, msg)
                    return

                # Audio lib required OpenAL
                if component == 'Audio':
                    # conf.start_msg('Magnum: Checking for OpenAL includes')
                    includes_audio = ['AL', 'OpenAL']
                    openal_found = False
                    for inc in includes_audio:
                        try:
                            # we need the full include dir
                            incl_audio = get_directory(inc+'/al.h', includes_check, True)
                            openal_found = True

                            magnum_component_includes[component] = magnum_component_includes[component] + [incl_audio]
                            break
                        except:
                            openal_found = False

                    if not openal_found:
                        fatal(required, 'Not found')
                        return

                    # conf.start_msg('Magnum: Checking for OpenAL lib')
                    libs_audio = ['OpenAL', 'al', 'openal', 'OpenAL32']
                    openal_found = False
                    for lib_audio in libs_audio:
                        try:
                            lib_dir = get_directory('lib'+lib_audio+'.'+suffix, libs_check)
                            openal_found = True

                            magnum_component_libpaths[component] = magnum_component_libpaths[component] + [lib_dir]
                            magnum_component_libs[component].append(lib_audio)
                            break
                        except:
                            openal_found = False

                    if not openal_found:
                        fatal(required, 'Not found')
                        return
            elif component_type == 'plugin':
                pat_audio = re.compile('.+AudioImporter$')
                pat_importer = re.compile('.+Importer$')
                pat_font = re.compile('.+Font$')
                pat_img_conv = re.compile('.+ImageConverter$')
                pat_font_conv = re.compile('.+FontConverter$')

                lib_path_suffix = ''
                component_file = component

                if re.match(pat_audio, component):
                    lib_path_suffix = 'audioimporters'
                    component_file = component.replace("AudioImporter", "Importer")
                elif re.match(pat_importer, component):
                    lib_path_suffix = 'importers'
                elif re.match(pat_font, component):
                    lib_path_suffix = 'fonts'
                elif re.match(pat_img_conv, component):
                    lib_path_suffix = 'imageconverters'
                elif re.match(pat_font_conv, component):
                    lib_path_suffix = 'fontconverters'

                if lib_path_suffix != '':
                    lib_path_suffix = lib_path_suffix + '/'

                include_dir = get_directory('MagnumPlugins/'+component+'/'+component_file+'.h', includes_check)
                lib = component
                # we need the full lib_dir in order to be able to link to the plugins
                # or not? because they are loaded dynamically
                # we need to set the libpath for the static plugins only
                lib_dir = get_directory('magnum/'+lib_path_suffix+lib+'.'+modules_suffix, libs_check, True)

                magnum_component_includes[component] = magnum_component_includes[component] + [include_dir]
                # magnum_component_libpaths[component] = magnum_component_libpaths[component] + [lib_dir]
                # magnum_component_libs[component].append(lib)
            elif component_type == 'bin':
                bin_name = 'magnum-'+component
                executable = conf.find_file(bin_name, bins_check)

                magnum_component_bins[component] = magnum_component_bins[component] + [executable]

        conf.end_msg(requested_components)

        # set environmental variables
        conf.env['INCLUDES_%s' % magnum_var] = magnum_includes
        conf.env['LIBPATH_%s' % magnum_var] = magnum_libpaths
        conf.env['LIB_%s' % magnum_var] = magnum_libs
        if conf.env['DEST_OS'] == 'darwin':
            conf.env['FRAMEWORK_%s_Magnum' % magnum_var] = ['OpenGL', 'Foundation']
        conf.env['EXEC_%s' % magnum_var] = magnum_bins

        # set main Magnum component
        conf.env['INCLUDES_%s_Magnum' % magnum_var] = magnum_includes
        conf.env['LIBPATH_%s_Magnum' % magnum_var] = magnum_libpaths
        conf.env['LIB_%s_Magnum' % magnum_var] = magnum_libs
        if conf.env['DEST_OS'] == 'darwin':
            conf.env['FRAMEWORK_%s_Magnum' % magnum_var] = ['OpenGL', 'Foundation']
        conf.env['EXEC_%s_Magnum' % magnum_var] = magnum_bins

        # Plugin directories
        magnum_plugins_dir = magnum_lib_path + '/magnum'
        magnum_plugins_font_dir = magnum_plugins_dir + '/fonts'
        magnum_plugins_fontconverter_dir = magnum_plugins_dir + '/fontconverters'
        magnum_plugins_imageconverter_dir = magnum_plugins_dir + '/imageconverters'
        magnum_plugins_importer_dir = magnum_plugins_dir + '/importers'
        magnum_plugins_audioimporter_dir = magnum_plugins_dir + '/audioimporters'

        # conf.env['%s_PLUGINS_DIR' % magnum_var] = magnum_plugins_dir
        # conf.env['%s_PLUGINS_FONT_DIR' % magnum_var] = magnum_plugins_font_dir
        # conf.env['%s_PLUGINS_FONTCONVERTER_DIR' % magnum_var] = magnum_plugins_fontconverter_dir
        # conf.env['%s_PLUGINS_IMAGECONVERTER_DIR' % magnum_var] = magnum_plugins_imageconverter_dir
        # conf.env['%s_PLUGINS_IMPORTER_DIR' % magnum_var] = magnum_plugins_importer_dir
        # conf.env['%s_PLUGINS_AUDIOIMPORTER_DIR' % magnum_var] = magnum_plugins_audioimporter_dir

        # set C++ defines
        conf.env['DEFINES_%s' % magnum_var] = []
        conf.env['DEFINES_%s' % magnum_var].append('%s_PLUGINS_DIR="%s"' % (magnum_var.upper(), magnum_plugins_dir))
        conf.env['DEFINES_%s' % magnum_var].append('%s_PLUGINS_FONT_DIR="%s"' % (magnum_var.upper(), magnum_plugins_font_dir))
        conf.env['DEFINES_%s' % magnum_var].append('%s_PLUGINS_FONTCONVERTER_DIR="%s"' % (magnum_var.upper(), magnum_plugins_fontconverter_dir))
        conf.env['DEFINES_%s' % magnum_var].append('%s_PLUGINS_IMAGECONVERTER_DIR="%s"' % (magnum_var.upper(), magnum_plugins_imageconverter_dir))
        conf.env['DEFINES_%s' % magnum_var].append('%s_PLUGINS_IMPORTER_DIR="%s"' % (magnum_var.upper(), magnum_plugins_importer_dir))
        conf.env['DEFINES_%s' % magnum_var].append('%s_PLUGINS_AUDIOIMPORTER_DIR="%s"' % (magnum_var.upper(), magnum_plugins_audioimporter_dir))
        # for config in magnum_config: # No need for these defines
        #     conf.env['DEFINES_%s' % magnum_var].append(config)
        if conf.env['DEST_OS'] == 'darwin':
            conf.env['DEFINES_%s' % magnum_var].append('%s_MAC_OSX' % magnum_var.upper())

        # copy C++ defines to Magnum::Magnum component; we want them to be available on all Magnum builds
        conf.env['DEFINES_%s_Magnum' % magnum_var] = copy.deepcopy(conf.env['DEFINES_%s' % magnum_var])

        # set component libs
        for component in requested_components:
            conf.env['INCLUDES_%s_%s' % (magnum_var, component)] = magnum_component_includes[component]
            conf.env['LIBPATH_%s_%s' % (magnum_var, component)] = magnum_component_libpaths[component]
            conf.env['LIB_%s_%s' % (magnum_var, component)] = magnum_component_libs[component]
            conf.env['EXEC_%s_%s' % (magnum_var, component)] = magnum_component_bins[component]
    except:
        if required:
            conf.fatal('Not found')
        conf.end_msg('Not found', 'RED')
        return
    return 1
