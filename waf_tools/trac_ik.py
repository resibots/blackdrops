#! /usr/bin/env python
# encoding: utf-8
# Konstantinos Chatzilygeroudis - 2015

"""
Quick n dirty TRAC_IK detection
"""

import os
from waflib.Configure import conf


def options(opt):
    opt.add_option('--trac_ik', type='string', help='path to trac_ik', dest='trac_ik')

@conf
def check_trac_ik(conf):
    if conf.options.trac_ik:
        includes_check = [conf.options.ros + '/include']
        libs_check = [conf.options.ros + '/lib']
    else:
        includes_check = ['/usr/local/include', '/usr/include']
        libs_check = ['/usr/local/lib/', '/usr/lib']

    try:
        conf.start_msg('Checking for TRAC IK includes')
        res = conf.find_file('trac_ik/trac_ik.hpp', includes_check)
        incl = res[:-len('trac_ik/trac_ik.hpp')]
        conf.end_msg('ok')
        lib = 'trac_ik_no_ros'
        conf.start_msg('Checking for TRAC IK lib')
        res = conf.find_file('lib'+lib+'.so', libs_check)
        lib_path = res[:-len('lib'+lib+'.so')]
        conf.end_msg('ok')
        conf.env.INCLUDES_TRAC_IK = [incl]
        conf.env.LIBPATH_TRAC_IK = [lib_path]
        conf.env.LIB_TRAC_IK = [lib]
    except:
        conf.end_msg('Not found', 'RED')
        return 1
    return 1
