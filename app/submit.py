"""submit result by fixed format
"""
import os
import zipfile

BASE_DIR = r'E:'
ROOT_DIR = '第2组-编号1'
filename_pattern = 'det_test_{0}.txt'

label_map = ['None', 'plane', 'storage', 'bridge', 'ship', 'harbor']


def _select_dst_filename(category):
    return filename_pattern.format(label_map[category])


class Submit(object):
    def __init__(self):
        self._filenames = []
        self._confidences = []
        # a tuple list denoting [(xmin,ymin,xmax,ymax),...]
        self._coords = []
        self._categorys = []
        self._f = None

    def open_files(self):
        if self._f is not None:
            base = os.path.join(BASE_DIR, ROOT_DIR)
            self._f = [open(filename_pattern.format(name)) for name in label_map if name != 'None']

    def close_files(self):
        for f in self._f:
            f.close()

    def build_file_structure(self):
        """build a basic file structure
        -ROOT_DIR
        --det_test_{0}.txt

        :return:None
        """
        _dir = os.path.join(BASE_DIR, ROOT_DIR)
        if not os.path.exists(_dir):
            os.makedirs(_dir)

    def append(self, filename, confidence, coord, category):
        self._filenames.append(filename)
        self._confidences.append(confidence)
        self._coords.append(coord)
        self._categorys.append(category)

    def write_to_file(self, delimiter='\t'):
        for filename, conf, coord, category in zip(self._filenames, self._confidences,
                                                   self._coords, self._categorys):
            xmin, ymin, xmax, ymax = coord
            f = self._f[int(category)]
            f.write('{1}{0}{2}{0}{3}{0}{4}{0}{5}{0}{6}\n'.format(delimiter,
                                                                 filename,
                                                                 conf,
                                                                 xmin, ymin, xmax, ymax))


def create_zip():
    _dir = os.path.join(BASE_DIR, ROOT_DIR)
    _zipdir = _dir + '.zip'
    f = zipfile.ZipFile(_zipdir, 'w', zipfile.ZIP_DEFLATED)
    for dirpath, dirnames, filenames in os.walk(_dir):
        for filename in filenames:
            f.write(os.path.join(dirpath, filename))
    f.close()
