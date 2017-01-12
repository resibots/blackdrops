#! /usr/bin/env python
import sys
sys.path.insert(0, sys.path[0]+'/waf_tools')

import os
import limbo
import mcts

def options(opt):
    opt.load('sdl')

def configure(conf):
    conf.load('sdl')
    conf.check_sdl()


def build(bld):
    libs = 'TBB EIGEN BOOST LIMBO LIBCMAES NLOPT SFERES2 BOOST_CHRONO RT SDL '
    cxxflags = bld.get_env()['CXXFLAGS']
    cxxflags += ['-D NODSP']

    # limbo.create_variants(bld,
    #                   source='test.cpp',
    #                   includes='. ../../src ../ ./include',
    #                   target='test',
    #                   uselib=libs,
    #                   uselib_local='limbo',
    #                   variants = ['SIMU'])

    limbo.create_variants(bld,
                      source='test_cp.cpp',
                      includes='. ../../src ../ ./include',
                      target='test_cp',
                      uselib=libs,
                      uselib_local='limbo',
                      variants = ['SIMU', 'SIMU DATA', 'SIMU INTACT', 'SIMU MEDIAN', 'SIMU MEDIAN DATA', 'SIMU GPPOLICY', 'SIMU GPPOLICY DATA', 'SIMU GPPOLICY INTACT', 'SIMU GPPOLICY MEDIAN', 'SIMU GPPOLICY MEDIAN DATA'])

    # limbo.create_variants(bld,
    #                   source='ode_test.cpp',
    #                   includes='. ../../src ../ ./include',
    #                   target='ode_test',
    #                   uselib=libs,
    #                   uselib_local='limbo',
    #                   variants = ['SIMU'])
