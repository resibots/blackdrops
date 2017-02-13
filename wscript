#! /usr/bin/env python
import sys
sys.path.insert(0, sys.path[0]+'/waf_tools')

import os
import limbo

def options(opt):
    opt.load('sdl')
    opt.load('robot_dart')
    opt.load('libdynamixel')

def configure(conf):
    conf.get_env()['BUILD_ROBOT'] = False
    conf.load('sdl')
    conf.load('robot_dart')
    conf.load('libdynamixel')
    conf.check_sdl()
    conf.check_robot_dart()
    conf.check_libdynamixel()


def build(bld):
    libs = 'TBB EIGEN BOOST LIMBO LIBCMAES NLOPT SFERES2 BOOST_CHRONO RT SDL '
    arm_libs = 'TBB EIGEN BOOST LIMBO LIBCMAES NLOPT SFERES2 BOOST_CHRONO RT ROBOT_DART DART BOOST_DART '
    arm_libs_graphic = arm_libs + 'DART_GRAPHIC'
    if bld.get_env()['BUILD_ROBOT'] == True:
        arm_libs = arm_libs + 'LIBDYNAMIXEL'
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
                      variants = ['SIMU', 'SIMU INTACT', 'SIMU MEDIAN', 'SIMU SPGPS', 'SIMU INTACT SPGPS', 'SIMU MEDIAN SPGPS', 'SIMU GPPOLICY', 'SIMU GPPOLICY INTACT', 'SIMU GPPOLICY MEDIAN', 'SIMU GPPOLICY SPGPS', 'SIMU GPPOLICY INTACT SPGPS', 'SIMU GPPOLICY MEDIAN SPGPS'])

    limbo.create_variants(bld,
                      source='test_arm.cpp',
                      includes='. ../../src ../ ./include',
                      target='test_arm',
                      uselib=arm_libs,
                      uselib_local='limbo',
                      variants = ['SIMU', 'SIMU SPGPS', 'SIMU GPPOLICY', 'SIMU GPPOLICY SPGPS'])

    if bld.get_env()['BUILD_ROBOT'] == True:
        limbo.create_variants(bld,
                          source='robot_arm.cpp',
                          includes='. ../../src ../ ./include',
                          target='robot_arm',
                          uselib=arm_libs,
                          uselib_local='limbo',
                          variants = ['SIMU'])

    if bld.get_env()['BUILD_GRAPHIC'] == True:
        limbo.create_variants(bld,
                          source='test_arm.cpp',
                          includes='. ../../src ../ ./include',
                          target='test_arm',
                          uselib=arm_libs_graphic,
                          uselib_local='limbo',
                          variants = ['GRAPHIC', 'GRAPHIC GPPOLICY'])
        if bld.get_env()['BUILD_ROBOT'] == True:
            limbo.create_variants(bld,
                              source='robot_arm.cpp',
                              includes='. ../../src ../ ./include',
                              target='robot_arm',
                              uselib=arm_libs_graphic,
                              uselib_local='limbo',
                              variants = ['GRAPHIC'])

    limbo.create_variants(bld,
                      source='test.cpp',
                      includes='. ../../src ../ ./include',
                      target='test',
                      uselib=libs,
                      uselib_local='limbo',
                      variants = ['SIMU', 'SIMU INTACT', 'SIMU GPPOLICY','SIMU GPPOLICY INTACT'])

    # limbo.create_variants(bld,
    #                   source='ode_test.cpp',
    #                   includes='. ../../src ../ ./include',
    #                   target='ode_test',
    #                   uselib=libs,
    #                   uselib_local='limbo',
    #                   variants = ['SIMU'])
