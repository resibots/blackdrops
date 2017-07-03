#! /usr/bin/env python
import sys
sys.path.insert(0, sys.path[0]+'/waf_tools')

import os
import limbo

def options(opt):
    opt.load('sdl')
    opt.load('robot_dart')

def configure(conf):
    conf.load('sdl')
    conf.load('robot_dart')

    conf.check_sdl()
    conf.check_robot_dart()

    conf.env.LIB_THREADS = ['pthread']


def build(bld):
    bld.recurse('src/')
