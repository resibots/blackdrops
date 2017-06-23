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
    libs = 'TBB EIGEN BOOST LIMBO LIBCMAES NLOPT SFERES2 BOOST_CHRONO RT SDL '
    arm_libs = 'TBB EIGEN BOOST LIMBO LIBCMAES NLOPT SFERES2 BOOST_CHRONO RT ROBOT_DART DART BOOST_DART THREADS KDL TRAC_IK '
    arm_libs_graphic = arm_libs + 'DART_GRAPHIC'
    robot_libs = libs + ' LIBDYNAMIXEL'
    # TO-DO: Change
    robot_arm_libs = 'ROS ' + arm_libs_graphic + ' LIBDYNAMIXEL'
    cxxflags = bld.get_env()['CXXFLAGS']
    cxxflags += ['-D NODSP']

    limbo.create_variants(bld,
                      source='multi_gp.cpp',
                      includes='. ../../src ../ ./include',
                      target='multi_gp',
                      uselib=libs,
                      uselib_local='limbo',
                      variants = ['SIMU'])

    limbo.create_variants(bld,
                      source='cartpole.cpp',
                      includes='. ../../src ../ ./include',
                      target='cartpole',
                      uselib=libs,
                      uselib_local='limbo',
                      variants = ['SIMU', 'SIMU MEAN', 'SIMU MEAN MODELIDENT', 'SIMU MEAN MODELIDENT ONLYMI', 'SIMU SPGPS', 'SIMU GPPOLICY', 'SIMU GPPOLICY SPGPS'])

    limbo.create_variants(bld,
                      source='simu_arm.cpp',
                      includes='. ../../src ../ ./include',
                      target='simu_arm',
                      uselib=arm_libs,
                      uselib_local='limbo',
                      variants = ['SIMU', 'SIMU GPPOLICY'])

    limbo.create_variants(bld,
                      source='door_opening.cpp',
                      includes='. ../../src ../ ./include',
                      target='door_opening',
                      uselib=arm_libs,
                      uselib_local='limbo',
                      variants = ['SIMU SPGPS', 'SIMU SPGPS MEAN'])

    limbo.create_variants(bld,
                      source='reacher.cpp',
                      includes='. ../../src ../ ./include',
                      target='reacher',
                      uselib=arm_libs,
                      uselib_local='limbo',
                      variants = ['SIMU', 'SIMU MEAN'])

    limbo.create_variants(bld,
                      source='half_cheetah.cpp',
                      includes='. ../../src ../ ./include',
                      target='half_cheetah',
                      uselib=arm_libs,
                      uselib_local='limbo',
                      variants = ['SIMU SPGPS'])

    # if bld.get_env()['BUILD_ROBOT'] == True:
        # limbo.create_variants(bld,
        #                   source='robot_arm.cpp',
        #                   includes='. ../../src ../ ./include',
        #                   target='robot_arm',
        #                   uselib=robot_libs,
        #                   uselib_local='limbo',
        #                   variants = ['SIMU'])
        #
        # limbo.create_variants(bld,
        #                   source='robot_replay.cpp',
        #                   includes='. ../../src ../ ./include',
        #                   target='robot_replay',
        #                   uselib=robot_libs,
        #                   uselib_local='limbo',
        #                   variants = ['SIMU'])

        # limbo.create_variants(bld,
        #                   source='door_robot.cpp',
        #                   includes='. ../../src ../ ./include',
        #                   target='door_robot',
        #                   uselib=robot_arm_libs,
        #                   uselib_local='limbo',
        #                   variants = ['SPGPS', 'SPGPS MEAN'])

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
                          variants = ['GRAPHIC SPGPS'])

        limbo.create_variants(bld,
                          source='door_opening.cpp',
                          includes='. ../../src ../ ./include',
                          target='door_opening',
                          uselib=arm_libs_graphic,
                          uselib_local='limbo',
                          variants = ['GRAPHIC SPGPS', 'GRAPHIC SPGPS MEAN'])

        limbo.create_variants(bld,
                          source='reacher.cpp',
                          includes='. ../../src ../ ./include',
                          target='reacher',
                          uselib=arm_libs_graphic,
                          uselib_local='limbo',
                          variants = ['GRAPHIC', 'GRAPHIC MEAN', 'GRAPHIC MEAN MODELIDENT', 'GRAPHIC MEAN MODELIDENT ONLYMI'])

    limbo.create_variants(bld,
                      source='pendulum.cpp',
                      includes='. ../../src ../ ./include',
                      target='pendulum',
                      uselib=libs,
                      uselib_local='limbo',
                      variants = ['SIMU', 'SIMU GPPOLICY', 'SIMU LINEAR'])
