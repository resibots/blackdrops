#! /usr/bin/env python
# encoding: utf-8
# Konstantinos Chatzilygeroudis - 2015

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
