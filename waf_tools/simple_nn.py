from waflib.Configure import conf


def options(opt):
  opt.add_option('--simple_nn', type='string', help='path to simple_nn', dest='simple_nn')

@conf
def check_simple_nn(conf, *k, **kw):
    def fail(msg, required):
        if required:
            conf.fatal(msg)
        conf.end_msg(msg, 'RED')
    def get_directory(filename, dirs):
        res = conf.find_file(filename, dirs)
        return res[:-len(filename)-1]

    required = kw.get('required', False)
    includes_check = ['/usr/local/include', '/usr/include']

    if conf.options.simple_nn:
        includes_check = [conf.options.simple_nn + '/include']
    try:
        conf.start_msg('Checking for simple_nn includes')
        incl = get_directory('simple_nn/neural_net.hpp', includes_check)
        incl = get_directory('simple_nn/layer.hpp', includes_check)
        conf.end_msg(incl)
        conf.env.INCLUDES_SIMPLE_NN = [incl]
    except:
        fail('Not found', required)
        return
    return 1