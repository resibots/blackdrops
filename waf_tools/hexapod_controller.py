#! /usr/bin/env python
# encoding: utf-8
# Konstantinos Chatzilygeroudis - 2015

"""
Quick n dirty hexapod_controller detection
"""

import os
from waflib.Configure import conf


def options(opt):
    opt.add_option('--controller', type='string', help='path to hexapod_controller', dest='controller')

@conf
def check_hexapod_controller(conf):
    includes_check = ['/usr/local/include', '/usr/include']

    if 'RESIBOTS_DIR' in os.environ:
        includes_check = [os.environ['RESIBOTS_DIR'] + '/include'] + includes_check

    if conf.options.controller:
        includes_check = [conf.options.controller + '/include']

    try:
        conf.start_msg('Checking for hexapod_controller includes')
        res = conf.find_file('hexapod_controller/hexapod_controller_simple.hpp', includes_check)
        conf.end_msg('ok')
        conf.env.INCLUDES_HEXAPOD_CONTROLLER = includes_check
    except:
        conf.end_msg('Not found', 'RED')
        return
    return 1