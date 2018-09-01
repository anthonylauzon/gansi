import os
import sys
import cv2
import numpy as np

from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

img = load_img(sys.argv[1]) 

mimg = img_to_array(img)

img = np.ndarray(shape=(128, 
                        128, 
                        3),
                 dtype=np.float32)

rt = '.'.join(sys.argv[1].split('.')[:-1])
os.makedirs("./{}".format(rt), exist_ok=True)

for num_y in range(5):
    offset_y = num_y * 129
    for num_x in range(7):
        offset_x = num_x * 129

        for y in range(128):
            for x in range(128):
                img[y][x] = mimg[offset_y+y][offset_x+x]

        im2 = img.copy()

        f = "./{}/{}-{}.png".format(rt, num_y, num_x)
        cv2.imwrite(f, im2)
