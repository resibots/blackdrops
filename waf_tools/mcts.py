#! /usr/bin/env python
# encoding: utf-8
# Konstantinos Chatzilygeroudis - 2015

"""
Quick n dirty mcts detection
"""

import os
from waflib.Configure import conf


def options(opt):
    opt.add_option('--mcts', type='string', help='path to mcts library', dest='mcts')

@conf
def check_mcts(conf):
    includes_check = ['/usr/local/include', '/usr/include']
    libs_check = ['/usr/local/lib', '/usr/lib']

    if 'RESIBOTS_DIR' in os.environ:
        includes_check = [os.environ['RESIBOTS_DIR'] + '/include'] + includes_check
        libs_check = [os.environ['RESIBOTS_DIR'] + '/lib'] + libs_check

    if conf.options.mcts:
        includes_check = [conf.options.mcts + '/include']
        libs_check = [conf.options.mcts + '/lib']

    try:
        conf.start_msg('Checking for mcts includes')
        res = conf.find_file('mcts/uct.hpp', includes_check)
        res = conf.find_file('mcts/defaults.hpp', includes_check)
        res = conf.find_file('mcts/macros.hpp', includes_check)
        conf.end_msg('ok')
        conf.env.INCLUDES_HEXAPOD_CONTROLLER = includes_check
    except:
        conf.end_msg('Not found', 'RED')
        return
    return 1
