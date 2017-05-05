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
    robot_libs = libs + 'LIBDYNAMIXEL'
    cxxflags = bld.get_env()['CXXFLAGS']
    cxxflags += ['-D NODSP']

    limbo.create_variants(bld,
                      source='cartpole.cpp',
                      includes='. ../../src ../ ./include',
                      target='cartpole',
                      uselib=libs,
                      uselib_local='limbo',
                      variants = ['SIMU', 'SIMU SPGPS', 'SIMU GPPOLICY', 'SIMU GPPOLICY SPGPS'])

    limbo.create_variants(bld,
                      source='simu_arm.cpp',
                      includes='. ../../src ../ ./include',
                      target='simu_arm',
                      uselib=arm_libs,
                      uselib_local='limbo',
                      variants = ['SIMU', 'SIMU GPPOLICY'])

    limbo.create_variants(bld,
                      source='half_cheetah.cpp',
                      includes='. ../../src ../ ./include',
                      target='half_cheetah',
                      uselib=arm_libs,
                      uselib_local='limbo',
                      variants = ['SIMU', 'SIMU SPGPS'])

    if bld.get_env()['BUILD_ROBOT'] == True:
        limbo.create_variants(bld,
                          source='robot_arm.cpp',
                          includes='. ../../src ../ ./include',
                          target='robot_arm',
                          uselib=robot_libs,
                          uselib_local='limbo',
                          variants = ['SIMU'])

        limbo.create_variants(bld,
                          source='robot_replay.cpp',
                          includes='. ../../src ../ ./include',
                          target='robot_replay',
                          uselib=robot_libs,
                          uselib_local='limbo',
                          variants = ['SIMU'])

    if bld.get_env()['BUILD_GRAPHIC'] == True:
        limbo.create_variants(bld,
                          source='simu_arm.cpp',
                          includes='. ../../src ../ ./include',
                          target='simu_arm',
                          uselib=arm_libs_graphic,
                          uselib_local='limbo',
                          variants = ['GRAPHIC', 'GRAPHIC GPPOLICY'])

        limbo.create_variants(bld,
                          source='half_cheetah.cpp',
                          includes='. ../../src ../ ./include',
                          target='half_cheetah',
                          uselib=arm_libs_graphic,
                          uselib_local='limbo',
                          variants = ['GRAPHIC', 'GRAPHIC SPGPS'])

    limbo.create_variants(bld,
                      source='pendulum.cpp',
                      includes='. ../../src ../ ./include',
                      target='pendulum',
                      uselib=libs,
                      uselib_local='limbo',
                      variants = ['SIMU', 'SIMU GPPOLICY', 'SIMU LINEAR'])
