#! /usr/bin/env python
# encoding: utf-8
# Konstantinos Chatzilygeroudis - 2018

"""
Quick n dirty Corrade detection
"""

import os
import io
from waflib import Utils, Logs
from waflib.Configure import conf
import copy

def options(opt):
        opt.add_option('--corrade_install_dir', type='string', help='path to corrade install directory', dest='corrade_install_dir')

@conf
def check_corrade(conf, *k, **kw):
    # Corrade will include all include dirs, libpaths and libraries
    # Corrade_X where X is one of ['Containers', 'PluginManager', 'TestSuite', 'Interconnect', 'Utility', 'rc']
    # will include all the inner-dependencies in Corrade in the correct order
    # i.e., Corrade_PluginManager will also have Containers, Utility and rc apart from its own things
    def get_directory(filename, dirs):
        res = conf.find_file(filename, dirs)
        return res[:-len(filename)-1]
    def find_in_string(data, text):
        return data.find(text)

    required = kw.get('required', False)

    # Check compiler version (for gcc); I am being a bit more strong (Corrade could be built with 4.7 but needs adjustment)
    if conf.env.CXX_NAME in ["gcc", "g++"] and int(conf.env['CC_VERSION'][0]+conf.env['CC_VERSION'][1]) < 48:
        msg = 'Corrade cannot be setup with GCC < 4.8!'
        if required:
            conf.fatal(msg)
        Logs.pprint('RED', msg)
        return

    includes_check = ['/usr/local/include', '/usr/include', '/opt/local/include', '/sw/include']
    libs_check = ['/usr/lib', '/usr/local/lib', '/opt/local/lib', '/sw/lib', '/lib', '/usr/lib/x86_64-linux-gnu/', '/usr/lib64']
    bins_check = ['/usr/bin', '/usr/local/bin', '/opt/local/bin', '/sw/bin', '/bin']

    # OSX/Mac uses .dylib and GNU/Linux .so
    suffix = 'dylib' if conf.env['DEST_OS'] == 'darwin' else 'so'

    if conf.options.corrade_install_dir:
        includes_check = [conf.options.corrade_install_dir + '/include']
        libs_check = [conf.options.corrade_install_dir + '/lib']
        bins_check = [conf.options.corrade_install_dir + '/bin']

    requested_components = kw.get('components', None)
    if requested_components == None:
        requested_components = []
    else:
        requested_components = requested_components.split()

    corrade_includes = []
    corrade_libpaths = []
    corrade_libs = []
    corrade_bins = []

    corrade_var = kw.get('uselib_store', 'Corrade')
    # to-do: enforce C++11/14

    corrade_possible_configs = ["GCC47_COMPATIBILITY", "MSVC2015_COMPATIBILITY", "MSVC2017_COMPATIBILITY", "BUILD_DEPRECATED", "BUILD_STATIC", "TARGET_UNIX", "TARGET_APPLE", "TARGET_IOS", "TARGET_IOS_SIMULATOR", "TARGET_WINDOWS", "TARGET_WINDOWS_RT", "TARGET_EMSCRIPTEN", "TARGET_ANDROID", "TESTSUITE_TARGET_XCTEST", "UTILITY_USE_ANSI_COLORS"]
    corrade_config = []

    corrade_components = ['Containers', 'PluginManager', 'TestSuite', 'Interconnect', 'Utility', 'rc']
    corrade_dependencies = {}
    corrade_dependencies['Containers'] = ['Utility']
    corrade_dependencies['Interconnect'] = ['Utility']
    corrade_dependencies['PluginManager'] = ['Containers', 'Utility', 'rc']
    corrade_dependencies['TestSuite'] = ['Utility']
    corrade_dependencies['Utility'] = ['Containers', 'rc']
    corrade_dependencies['rc'] = []

    corrade_component_type = {}
    corrade_component_type['Containers'] = 'include'
    corrade_component_type['Interconnect'] = 'lib'
    corrade_component_type['PluginManager'] = 'lib'
    corrade_component_type['TestSuite'] = 'lib'
    corrade_component_type['Utility'] = 'lib'
    corrade_component_type['rc'] = 'bin'

    corrade_component_includes = {}
    corrade_component_libpaths = {}
    corrade_component_libs = {}
    corrade_component_bins = {}

    corrade_flags = '-Wall \
                    -Wextra \
                    -Wold-style-cast \
                    -Winit-self \
                    -Werror=return-type \
                    -Wmissing-declarations \
                    -pedantic \
                    -fvisibility=hidden'
    corrade_flags = corrade_flags.split()

    if conf.env.CXX_NAME in ["gcc", "g++"]:
        corrade_flags.append('-Wzero-as-null-pointer-constant')
        corrade_flags.append('-Wdouble-promotion')
    if conf.env.CXX_NAME in ["clang", "clang++", "llvm"]:
        corrade_flags.append('-Wmissing-prototypes')
        corrade_flags.append('-Wno-shorten-64-to-32')

    try:
        conf.start_msg('Checking for Corrade includes')
        corrade_include_dir = get_directory('Corrade/Corrade.h', includes_check)
        corrade_includes = [corrade_include_dir]
        conf.end_msg(corrade_include_dir)

        conf.start_msg('Getting Corrade configuration')
        config_file = conf.find_file('Corrade/configure.h', includes_check)
        with io.open(config_file, errors = 'ignore') as f:
            config_content = f.read()
        for config in corrade_possible_configs:
            index = find_in_string(config_content, '#define CORRADE_' + config)
            if index > -1:
                corrade_config.append(config)
        conf.end_msg(corrade_config)

        conf.start_msg('Checking for Corrade components')
        # only check for components that can exist
        requested_components = list(set(requested_components).intersection(corrade_components))
        # add dependencies
        for lib in requested_components:
            requested_components = requested_components + corrade_dependencies[lib]
        # remove duplicates
        requested_components = list(set(requested_components))

        for component in requested_components:
            # initialize component libs/bins/paths
            corrade_component_libpaths[component] = []
            corrade_component_libs[component] = []
            corrade_component_bins[component] = []
            # get general Corrade include
            corrade_component_includes[component] = [copy.deepcopy(corrade_include_dir)]
            # get component type
            component_type = corrade_component_type[component]
            if component_type == 'include':
                include_dir = get_directory('Corrade/'+component+'/'+component+'.h', includes_check)
                corrade_includes = corrade_includes + [include_dir]
                corrade_component_includes[component] = corrade_component_includes[component] + [include_dir]
            elif component_type == 'lib':
                include_dir = get_directory('Corrade/'+component+'/'+component+'.h', includes_check)
                corrade_includes = corrade_includes + [include_dir]
                lib = 'Corrade'+component
                lib_dir = get_directory('lib'+lib+'.'+suffix, libs_check)
                corrade_libs.append(lib)
                corrade_libpaths = corrade_libpaths + [lib_dir]

                corrade_component_libpaths[component] = corrade_component_libpaths[component] + [lib_dir]
                corrade_component_libs[component].append(lib)
                if component == 'PluginManager' or component == 'Utility':
                    # PluginManager/Utility need the libdl.so library
                    try:
                        lib_dir = get_directory('libdl.'+suffix, libs_check)
                        corrade_component_libpaths[component] = corrade_component_libpaths[component] + [lib_dir]
                        corrade_component_libs[component].append('dl')
                    except:
                        pass
                # to-do: Do additional stuff for TestSuite
                # to-do: Do additional stuff for Utility
            elif component_type == 'bin':
                bin_name = 'corrade-'+component
                executable = conf.find_file(bin_name, bins_check)
                corrade_bins = corrade_bins + [executable]

                corrade_component_bins[component] = corrade_component_bins[component] + [executable]
        #remove duplicates
        corrade_includes = list(set(corrade_includes))
        corrade_libpaths = list(set(corrade_libpaths))
        corrade_libs = list(set(corrade_libs))
        conf.end_msg(corrade_libs + corrade_bins)

        # set environmental variables
        conf.env['INCLUDES_%s' % corrade_var] = corrade_includes
        conf.env['LIBPATH_%s' % corrade_var] = corrade_libpaths
        conf.env['LIB_%s' % corrade_var] = corrade_libs
        conf.env['EXEC_%s' % corrade_var] = corrade_bins
        # set component libs
        for component in requested_components:
            for lib in corrade_dependencies[component]:
                corrade_component_includes[component] = corrade_component_includes[component] + corrade_component_includes[lib]
                corrade_component_libpaths[component] = corrade_component_libpaths[component] + corrade_component_libpaths[lib]
                corrade_component_libs[component] = corrade_component_libs[component] + corrade_component_libs[lib]
                corrade_component_bins[component] = corrade_component_bins[component] + corrade_component_bins[lib]

            conf.env['INCLUDES_%s_%s' % (corrade_var, component)] = list(set(corrade_component_includes[component]))
            conf.env['LIBPATH_%s_%s' % (corrade_var, component)] = list(set(corrade_component_libpaths[component]))
            conf.env['LIB_%s_%s' % (corrade_var, component)] = list(set(corrade_component_libs[component]))
            conf.env['EXEC_%s_%s' % (corrade_var, component)] = list(set(corrade_component_bins[component]))
        # set C++ flags
        conf.env['CXX_FLAGS_%s' % corrade_var] = corrade_flags
    except:
        if required:
            conf.fatal('Not found')
        conf.end_msg('Not found', 'RED')
        return
    return 1

@conf
def corrade_enable_pedantic_flags(conf, corrade_var = 'Corrade'):
    # this has to be called after check_corrade
    conf.env['CXXFLAGS'] = list(set(conf.env['CXXFLAGS'] + conf.env['CXX_FLAGS_%s' % corrade_var]))


# Corrade TestSuite
def corrade_add_test(bld, source, use='', uselib='', includes='.', cxxflags='', defines='', target='', corrade_var = 'Corrade'):
    if 'CorradeTestSuite' not in bld.env['LIB_%s_TestSuite' % corrade_var]:
        bld.fatal('Corrade TestSuite is not found!')
    source_list = source.split()
    if not target:
        target_name = source_list[0].replace('.cpp', '')
    else:
        target_name = target

    bld.program(features='cxx test',
                    source=source,
                    target=target_name,
                    includes=includes,
                    uselib=uselib + ' %s_TestSuite' % corrade_var,
                    defines=defines,
                    cxxflags=bld.env['CXXFLAGS'] + cxxflags.split(),
                    use=use)

def summary(bld):
    from waflib.Tools import waf_unit_test
    lst = getattr(bld, 'utest_results', [])
    total = 0
    tfail = 0
    if lst:
        total = len(lst)
        tfail = len([x for x in lst if x[1]])
    waf_unit_test.summary(bld)
    # Logs.pprint('CYAN', 'Test execution summary')
    # Logs.pprint('CYAN', '  tests that pass %s' % str(total-tfail)+'/'+str(total))
    # for l in lst:
    #     if l[1]:
    #         continue
    #     Logs.pprint('CYAN', l[0])
    #     Logs.pprint('GREEN', l[2])
    # Logs.pprint('CYAN', '  tests that fail %s' % str(tfail)+'/'+str(total))
    # for l in lst:
    #     if not l[1]:
    #         continue
    #     Logs.pprint('CYAN', l[0])
    #     Logs.pprint('RED', l[2])
    if tfail > 0:
        bld.fatal("Build failed, because some tests failed!")

from waflib.Task import Task
import time
class readFile(Task):
    def run(self):
        # trick to trigger the re-compilation
        # i.e., write a different value each time
        self.outputs[0].write(str(time.time()))

    def scan(self):
        # scan to see if resource files have changed
        config_file = self.inputs[0]
        config_file_path = config_file.abspath()[:config_file.abspath().rfind('/')]
        try:
            with io.open(config_file.abspath(), errors = 'ignore') as f:
                config_content = f.readlines()
        except:
            self.fatal('Could not load file \'' + config_file.abspath() + '\'')

        dependencies = []

        import re
        filename_regex = '^[ \t]*filename[ \t]*=[ \t]*"?([^"]+)"?[ \t]*$'
        prog = re.compile(filename_regex, re.MULTILINE)
        for line in config_content:
            results = re.match(prog, line)
            if results:
                filename = results.group(1).strip()
                if filename[0] != '/':
                    filename = config_file_path + '/' + filename
                filename = str(filename)
                dependencies.append(self.generator.bld.root.find_node(filename))
        return (dependencies, time.time())

# Corrade Resource
def corrade_add_resource(bld, name, config_file, corrade_var = 'Corrade'):
    if not any('corrade-rc' in b for b in bld.env['EXEC_%s_rc' % corrade_var]):
        bld.fatal('corrade-bin is not found!')
    corrade_bin = (" ".join(s for s in bld.env['EXEC_%s_rc' % corrade_var] if 'corrade-rc' in s)).split()[0]

    target_resource = 'resource_' + name + '.cpp'
    target_depends = target_resource + '.depends'
    name_depends = name + '-dependencies'

    config_filename = config_file
    if not isinstance(config_file, str):
        config_filename = config_file.path_from(bld.path)
    tmp_name = os.path.realpath(config_filename)
    full_config_path = tmp_name[:tmp_name.rfind('/')]
    short_config = tmp_name[tmp_name.rfind('/')+1:]

    read1 = readFile(env=bld.env)
    if isinstance(config_file, str):
        read1.set_inputs(bld.path.find_node(config_file))
    else:
        read1.set_inputs(config_file)
    read1.set_outputs(bld.path.find_or_declare(name_depends))
    bld.add_to_group(read1)

    bld(rule='cp ${SRC} ${TGT}', source=config_file, target=target_depends)
    bld(rule=corrade_bin + ' ' + name + ' ' + full_config_path+'/'+short_config + ' ${TGT}', source=read1.outputs, target=target_resource)

    return target_resource

# Corrade PluginManager
def corrade_add_plugin(bld, name, config_file, source, use='', uselib='', includes='.', cxxflags='', defines='', corrade_var = 'Corrade'):
    if 'CorradePluginManager' not in bld.env['LIB_%s_PluginManager' % corrade_var]:
        bld.fatal('Corrade PluginManager is not found!')
    name = name.strip()
    if '/' in name or '\\' in name:
        bld.fatal('Name of the plugin should not be a path!')
    plugin_lib = bld.program(features = 'cxx cxxshlib',
                                source=source,
                                includes=bld.env['INCLUDES_%s_PluginManager' % corrade_var] + includes.split(),
                                use=use,
                                uselib=uselib,
                                cxxflags=bld.env['CXXFLAGS'] + cxxflags.split(),
                                defines=defines.split() + ['CORRADE_DYNAMIC_PLUGIN'],
                                target=name)
    plugin_lib.env.cxxshlib_PATTERN = '%s.'+suffix
    bld(rule='cp ${SRC} ${TGT}', source=bld.path.make_node(config_file), target=bld.path.get_bld().make_node(config_file[config_file.rfind('/') + 1:]))

    # to-do: add installation

def corrade_add_static_plugin(bld, name, config_file, source, use='', uselib='', includes='.', cxxflags='', defines='', corrade_var = 'Corrade'):
    if 'CorradePluginManager' not in bld.env['LIB_%s_PluginManager' % corrade_var]:
        bld.fatal('Corrade PluginManager is not found!')
    name = name.strip()
    if '/' in name or '\\' in name:
        bld.fatal('Name of the plugin should not be a path!')
    resource_file = 'resources_' + name + '.conf'
    config_node = bld.path.make_node(config_file)
    config_file_full = config_node.abspath()

    resource_node = bld.path.get_bld().make_node(resource_file)
    bld(rule='echo "group=CorradeStaticPlugin_' + name + '\n[file]\nfilename=\"' + config_file_full + '\"\nalias=' + name + '.conf" > ${TGT}', source=config_node, target=resource_node)

    # I don't think we need this
    # bld(rule='cp ${SRC} ${TGT}', source=config_node, target=bld.path.get_bld().make_node(config_file))

    resource = corrade_add_resource(bld, name, resource_node)
    bld.program(features = 'cxx cxxstlib',
                    source=source + ' ' + resource,
                    includes=bld.env['INCLUDES_%s_PluginManager' % corrade_var] + includes.split(),
                    use=use,
                    uselib=uselib,
                    cxxflags=bld.env['CXXFLAGS'] + cxxflags.split(),
                    defines=defines.split() + ['CORRADE_STATIC_PLUGIN'],
                    target=name)

    # to-do: add installation