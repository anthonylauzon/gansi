import math
import sys
from wand.image import Image

pixels = []
width = 0
height = 0
blob = None

GRADIENT_VALS = [
    0, 
    10, 
    20, 
    30
]

GLYPHS= [
    b'\xdb',
    b'\xb2',
    b'\xb1',
    b'\xb0'
]

COLOR_TABLE = {
    30: (0,0,0),
    31: (170,0,0),
    32: (0,170,0),
    33: (170,85,0),
    34: (0,0,170),
    35: (170,0,170),
    36: (0,170,170),
    37: (170,170,170),
    90: (85,85,85),
    91: (255,85,85),
    92: (85,255,85),
    93: (255,255,85),
    94: (95,85,255),
    95: (255,85,255),
    96: (85,255,255),
    97: (255,255,255)
}

# Load image
with Image(filename=sys.argv[1]) as image:
    # Enforce pixels quantum between 0 & 255
    image.depth = 8
    # Save width & height for later use
    width, height = image.width, image.height
    # Copy raw image data into blob string
    blob = image.make_blob(format='RGB')

for cursor in range(0, width * height * 3, 3):
    pixels.append((blob[cursor],     
                   blob[cursor + 1],  
                   blob[cursor + 2])) 

def v2ci(v):
    if v < 48:
        return 0
    elif v < 115:
        return 1
    return int((v - 35) / 40)

def dist_square(a1,b1,c1,a2,b2,c2):
    return ((a1-a2)*(a1-a2) + (b1-b2)*(b1-b2) + (c1-c2)*(c1-c2))

def rgb2ansi256(r, g, b):    
    ir = v2ci(r)
    ig = v2ci(g)
    ib = v2ci(b)

    color_index = (36 * ir + 6 * ig + ib) 

    average = (r + g + b) / 3;
    
    gray_index = (average - 3) / 10 
    if average > 238:
        gray_index = 23

    i2cv = [0, 0x5f, 0x87, 0xaf, 0xd7, 0xff]

    cr = i2cv[ir]
    cg = i2cv[ig]
    cb = i2cv[ib]  

    gv = 8 + 10 * gray_index 

    color_err = dist_square(cr, cg, cb, r, g, b)
    gray_err  = dist_square(gv, gv, gv, r, g, b)

    if color_err <= gray_err:
        return 16 + color_index

    return 232 + gray_index

def rgb2ansi(r, g, b):
    best_color_distance = sys.maxsize
    best_glyph = GLYPHS[0]
    best_color = COLOR_TABLE[30]

    for k, v in COLOR_TABLE.items():

        for i, j in enumerate(GRADIENT_VALS):

            v0 = v[0] - j
            v1 = v[1] - j
            v2 = v[2] - j

            v0 = 0 if v0 < 0 else v0
            v1 = 0 if v1 < 0 else v1
            v2 = 0 if v2 < 0 else v2

            sr = math.sqrt(
                (r-v0)**2 + 
                (g-v1)**2 + 
                (b-v2)**2
            )

            if sr < best_color_distance:
                best_color_distance = sr
                best_color = k
                best_glyph = GLYPHS[i]
    return best_color, best_glyph


out_ansi = open("{}.ans".format(sys.argv[1]), 'wb')

for i, pixel in enumerate(pixels):
    r, g, b = pixel
    best_color, best_glyph = rgb2ansi(r, g, b)
    ansi_color_bytes = str(best_color).encode('ascii')

    out_ansi.write(b'\x1b[' + ansi_color_bytes + b'm' + best_glyph)



