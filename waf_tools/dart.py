#!/usr/bin/env python
# encoding: utf-8
#| Copyright Inria July 2017
#| This project has received funding from the European Research Council (ERC) under
#| the European Union's Horizon 2020 research and innovation programme (grant
#| agreement No 637972) - see http://www.resibots.eu
#|
#| Contributor(s):
#|   - Konstantinos Chatzilygeroudis (konstantinos.chatzilygeroudis@inria.fr)
#|   - Roberto Rama (bertoski@gmail.com)
#|
#| This software is a computer library whose purpose is to optimize continuous,
#| black-box functions. It mainly implements Gaussian processes and Bayesian
#| optimization.
#| Main repository: http://github.com/resibots/blackdrops
#|
#| This software is governed by the CeCILL-C license under French law and
#| abiding by the rules of distribution of free software.  You can  use,
#| modify and/ or redistribute the software under the terms of the CeCILL-C
#| license as circulated by CEA, CNRS and INRIA at the following URL
#| "http://www.cecill.info".
#|
#| As a counterpart to the access to the source code and  rights to copy,
#| modify and redistribute granted by the license, users are provided only
#| with a limited warranty  and the software's author,  the holder of the
#| economic rights,  and the successive licensors  have only  limited
#| liability.
#|
#| In this respect, the user's attention is drawn to the risks associated
#| with loading,  using,  modifying and/or developing or reproducing the
#| software by the user in light of its specific status of free software,
#| that may mean  that it is complicated to manipulate,  and  that  also
#| therefore means  that it is reserved for developers  and  experienced
#| professionals having in-depth computer knowledge. Users are therefore
#| encouraged to load and test the software's suitability as regards their
#| requirements in conditions enabling the security of their systems and/or
#| data to be ensured and,  more generally, to use and operate it in the
#| same conditions as regards security.
#|
#| The fact that you are presently reading this means that you have had
#| knowledge of the CeCILL-C license and that you accept its terms.
#|

"""
Quick n dirty DART detection
"""

import os
from waflib.Configure import conf


def options(opt):
	opt.add_option('--dart', type='string', help='path to DART physics engine/sim', dest='dart')

@conf
def check_dart(conf):
	if conf.options.dart:
		includes_check = [conf.options.dart + '/include']
		libs_check = [conf.options.dart + '/lib']
	else:
		includes_check = ['/usr/local/include', '/usr/include']
		libs_check = ['/usr/local/lib', '/usr/lib']

		if 'RESIBOTS_DIR' in os.environ:
			includes_check = [os.environ['RESIBOTS_DIR'] + '/include'] + includes_check
			libs_check = [os.environ['RESIBOTS_DIR'] + '/lib'] + libs_check

	# DART requires some of bullets includes (if installed with bullet enabled)
	bullet_check = ['/usr/local/include/bullet', '/usr/include/bullet']
	bullet_found = False
	try:
		bullet_found = conf.find_file('btBulletCollisionCommon.h', bullet_check)
	except:
		bullet_found = False

	# DART requires assimp library
	assimp_check = ['/usr/local/include', '/usr/include']
	assimp_libs = ['/usr/local/lib', '/usr/lib', '/usr/lib/x86_64-linux-gnu/']
	assimp_found = False
	try:
		assimp_found = conf.find_file('assimp/scene.h', assimp_check)
		assimp_found = assimp_found and conf.find_file('libassimp.so', assimp_libs)
	except:
		assimp_found = False

	# DART requires OSG library for their graphic version
	osg_check = ['/usr/local/include', '/usr/include']
	osg_libs = ['/usr/local/lib', '/usr/lib', '/usr/lib/x86_64-linux-gnu']
	osg_found = False
	osg_comp = ['osg', 'osgViewer', 'osgManipulator', 'osgGA', 'osgDB']
	try:
		osg_found = True
		for f in osg_comp:
			osg_found = osg_found and conf.find_file(f + '/Version', osg_check)
			osg_found = osg_found and conf.find_file('lib' + f + '.so', osg_libs)
	except:
		osg_found = False

	try:
		conf.start_msg('Checking for DART includes (including utils/urdf)')
		res = conf.find_file('dart/dart.hpp', includes_check)
		res = res and conf.find_file('dart/utils/utils.hpp', includes_check)
		res = res and conf.find_file('dart/utils/urdf/urdf.hpp', includes_check)
		conf.end_msg('ok')
		try:
			conf.start_msg('Checking for DART gui includes')
			res = res and conf.find_file('dart/gui/gui.hpp', includes_check)
			res = res and conf.find_file('dart/gui/osg/osg.hpp', includes_check)
			conf.end_msg('ok')
		except:
			conf.end_msg('Not found', 'RED')
		conf.start_msg('DART: Checking for optional Bullet includes')
		more_includes = []
		if bullet_found:
			more_includes += bullet_check
			conf.end_msg('ok')
		else:
			conf.end_msg('Not found - be sure that your DART installation is without Bullet enabled', 'RED')
		if assimp_found:
			more_includes += assimp_check
		conf.start_msg('Checking for DART libs (including utils/urdf)')
		res = res and conf.find_file('libdart.so', libs_check)
		res = res and conf.find_file('libdart-utils.so', libs_check)
		res = res and conf.find_file('libdart-utils-urdf.so', libs_check)
		conf.end_msg('ok')
		conf.env.INCLUDES_DART = includes_check + more_includes
		conf.env.LIBPATH_DART = libs_check
		conf.env.LIB_DART = ['dart', 'dart-utils', 'dart-utils-urdf']
		conf.start_msg('DART: Checking for Assimp')
		if assimp_found:
			conf.end_msg('ok')
			conf.env.LIBPATH_DART = conf.env.LIBPATH_DART + assimp_libs
			conf.env.LIB_DART.append('assimp')
		else:
			conf.end_msg('Not found - Your programs may not compile', 'RED')
		if bullet_found:
			conf.env.LIB_DART.append('BulletCollision')
			conf.env.LIB_DART.append('LinearMath')
		try:
			conf.start_msg('Checking for DART gui libs')
			res = res and conf.find_file('libdart-gui.so', libs_check)
			res = res and conf.find_file('libdart-gui-osg.so', libs_check)
			conf.end_msg('ok')
			conf.env.INCLUDES_DART_GRAPHIC = conf.env.INCLUDES_DART
			conf.env.LIBPATH_DART_GRAPHIC = conf.env.LIBPATH_DART
			conf.env.LIB_DART_GRAPHIC = conf.env.LIB_DART + ['dart-gui', 'dart-gui-osg']
			conf.start_msg('DART: Checking for OSG (optional)')
			if osg_found:
				conf.env.INCLUDES_DART_GRAPHIC += osg_check
				conf.env.LIBPATH_DART_GRAPHIC += osg_libs
				conf.env.LIB_DART_GRAPHIC += osg_comp
				conf.end_msg('ok')
			else:
				conf.end_msg('Not found - Your graphical programs may not compile/link', 'RED')
			conf.get_env()['BUILD_GRAPHIC'] = True
		except:
			conf.end_msg('Not found', 'RED')
	except:
		conf.end_msg('Not found', 'RED')
		return
	return 1
