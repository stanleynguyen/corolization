from os import walk, remove
from os.path import join
import sys
from skimage.io import imread
import warnings

warnings.filterwarnings("error")

if len(sys.argv) < 2:
    raise(TypeError)


for r, _, files in walk(sys.argv[1]):
    for f in files:
        if f.endswith('.jpg'):
            try:
                img = imread(join(r, f))
                if img.shape[2] != 3:
                    remove(join(r, f))
                    print('removed {}'.format(join(r, f)))
            except (UserWarning, IndexError) as e:

                remove(join(r, f))
                print('removed {}'.format(join(r, f)))
