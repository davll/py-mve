import sysconfig, sys, os
from os.path import join, dirname, abspath

def distutils_dir_name(dname = 'lib'):
    """Returns the name of a distutils build directory"""
    fmt = "{dirname}.{platform}-{version[0]}.{version[1]}"
    return fmt.format(dirname = dname,
                      platform = sysconfig.get_platform(),
                      version = sys.version_info)

def distutils_build_path():
    root = dirname(abspath(__file__))
    return join(root, 'build', distutils_dir_name())

sys.path.insert(0, distutils_build_path())
print(sys.path)
