import glob
import os


def __finder(dir, name):
    path = glob.glob(dir + '**/' + name, recursive=True)
    if path is None:
        return None
    elif isinstance(path, (list, tuple)):
        if len(path) == 0:
            return None
        else:
            return path[0]
    return str(path)


def find_file(dir, name, exts=None):
    if dir[-1] != os.path.sep:
        dir += '/'
    if exts is None:
        return __finder(dir, name)
    for ext in exts:
        if ext[0] != '.':
            ext = '.' + ext
        path = __finder(dir, name + ext)
        if path is not None:
            return path
    return None


if __name__ == '__main__':
    print(find_file('./reportdata', '13-_8897', ['tiff', 'tif']))
