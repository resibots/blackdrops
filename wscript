#! /usr/bin/env python
import sys
sys.path.insert(0, sys.path[0]+'/waf_tools')

import os
import limbo

def options(opt):
    opt.load('sdl')
    opt.load('robot_dart')
    opt.load('libdynamixel')
    opt.load('ros')
    opt.load('kdl')
    opt.load('trac_ik')

def configure(conf):
    conf.get_env()['BUILD_ROBOT'] = False
    conf.load('sdl')
    conf.load('robot_dart')
    conf.load('libdynamixel')
    conf.load('ros')
    conf.load('kdl')
    conf.load('trac_ik')

    conf.check_sdl()
    conf.check_robot_dart()
    conf.check_libdynamixel()
    conf.check_ros()
    conf.check_kdl()
    conf.check_trac_ik()

    conf.env.LIB_THREADS = ['pthread']


def build(bld):
    bld.recurse('src/')
