import glob
import sys

from wand.image import Image

WIDTH = 80
HEIGHT = 80


for path in glob.glob("{}/**/*.png".format(sys.argv[1]), recursive=True):
    f = open(path, 'rb')
    with Image(file=f) as img:
        img.resize(WIDTH, HEIGHT)
        img.save(filename="{}.resized.png".format(path))
    f.close()