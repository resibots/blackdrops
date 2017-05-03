#! /usr/bin/env python
# encoding: utf-8
# JB Mouret - 2009
# Federico Allocati - 2015

"""
Quick n dirty eigen3 detection
"""

from waflib.Configure import conf


def options(opt):
    opt.add_option('--eigen', type='string', help='path to eigen', dest='eigen')


@conf
def check_eigen(conf):
    if conf.options.eigen:
        includes_check = [conf.options.eigen]
    else:
        includes_check = ['/usr/include/eigen3', '/usr/local/include/eigen3', '/usr/include', '/usr/local/include']

    conf.start_msg('Checking for Eigen')
    try:
        res = conf.find_file('Eigen/Core', includes_check)
    except:
        res = False

    if res:
        conf.env.INCLUDES_EIGEN = includes_check
        conf.end_msg('ok')
    else:
        if conf.options.eigen:
            msg = 'Not found in %s' % conf.options.eigen
        else:
            msg = 'Not found, use --eigen=/path/to/eigen'
        conf.end_msg(msg, 'RED')
	return 1
