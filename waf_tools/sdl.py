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
Quick n dirty sdl 2 detection
"""

import os, glob, types
from waflib.Configure import conf


def options(opt):
	opt.add_option('--sdl', type='string', help='path to nlopt', dest='sdl')


@conf
def check_sdl(conf):
	if conf.options.sdl:
		includes_check = [conf.options.sdl + '/include']
		libs_check = [conf.options.sdl + '/lib']
	else:
		includes_check = ['/usr/local/include', '/usr/include']
		libs_check = ['/usr/local/lib', '/usr/lib', '/usr/lib/x86_64-linux-gnu/']
		if 'RESIBOTS_DIR' in os.environ:
			includes_check = [os.environ['RESIBOTS_DIR'] + '/include'] + includes_check
			libs_check = [os.environ['RESIBOTS_DIR'] + '/lib'] + libs_check

	try:
		conf.start_msg('Checking for SDL 2 C++ includes')
		res = conf.find_file('SDL2/SDL.h', includes_check)
		conf.end_msg('ok')
	except:
		conf.end_msg('Not found', 'RED')
		return 1
	conf.start_msg('Checking for SDL 2 C++ libs')
	found = False
	for lib in ['libSDL2.so', 'libSDL2_image.so']:
		try:
			found = found or conf.find_file(lib, libs_check)
		except:
			continue
	if not found:
		conf.end_msg('Not found', 'RED')
		return 1
	else:
		conf.end_msg('ok')
		conf.env.INCLUDES_SDL = includes_check
		conf.env.LIBPATH_SDL = libs_check
		conf.env.DEFINES_SDL = ['USE_SDL']
		conf.env.LIB_SDL = ['SDL2', 'SDL2_image']
	return 1
