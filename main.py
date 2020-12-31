from PIL import Image, ImageDraw
import numpy as np
import cmath
import os
import sys
from tqdm import tqdm
import cv2

# dimensions of frame
DIM = (3840, 2160)

FPS = 60
SECS_PER_SHAPE = 200
FRAMES_PER_SHAPE = FPS * SECS_PER_SHAPE

# everything is drawn on top of this
BASE_FRAME = './resources/stanford/base.jpg'

# number of frames to store in memory at a time
# creates new video in disk every time # of frames pass threshold
MEMORY_THRESHOLD = 600

# defaults to saving videos here
TEST_DIR = '/volumes/nathanbackup/fourier/vids'

# thumbnail attributes
THUMB_POS = (2800, 500)
THUMB_SIZE_ORIG = (600, 600)
THUMB_SIZE = (800, 800)
THUMB_CENTER = (THUMB_SIZE[0]//2 + THUMB_POS[0], THUMB_SIZE[1]//2 + THUMB_POS[1])
THUMB_SCALE = (DIM[0]*DIM[1]) / (THUMB_SIZE_ORIG[0]*THUMB_SIZE_ORIG[1])
MASK_DIM = (700, 500)

TEST_NO = 13

# colors
STANFORD_GREEN = (22, 103, 57)
LIGHT_ORANGE = (255, 116, 26)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# globals
vid_no = 0
endpoints = []
thumb_endpoints = []

# calculated constants
thumb_dim_shift = complex(THUMB_SIZE_ORIG[0] // 2, THUMB_SIZE_ORIG[0] // 2)


def render(dft, image, shift):
    '''Renders one frame.
    - circles, each of which represent a one term of the summation
    - point that shows the result of the sum
    - path that shows the trace
    Also renders a thumbnail with with clearer, zoomed in circles.
    '''
    draw = ImageDraw.Draw(image)
    total_sum = sum(dft)


    total = total_sum.conjugate() + shift[0] + 1j*shift[1]
    thumbnail_dim1 = total - thumb_dim_shift
    thumbnail_dim2 = total + thumb_dim_shift

    # traced path
    endpoints.append(ctuple(total))
    draw.line(endpoints, fill=LIGHT_ORANGE, width=3)

    # thumbnail base (includes image and path)
    zoomed_image = image.crop(ctuple(thumbnail_dim1) + ctuple(thumbnail_dim2))
    zoomed_image = zoomed_image.resize(THUMB_SIZE)
    image.paste(zoomed_image, THUMB_POS)

    # find max number of circles that fit in thumbnail
    acc = 0
    THUMB_CIRCLES = 1

    bounds = ((b1 := complex(*THUMB_POS)), b1 + complex(*THUMB_SIZE))
    while in_bounds(complex(*THUMB_CENTER) - acc * THUMB_SCALE, *bounds):
        acc += dft[-THUMB_CIRCLES]
        THUMB_CIRCLES += 1

    acc += dft[-THUMB_CIRCLES]


    # draw thumbnail circles
    thumb_shift = complex(*THUMB_CENTER) - acc.conjugate() * THUMB_SCALE
    acc = 0
    for num in dft[-THUMB_CIRCLES:]:
        draw_circle(num, acc, draw, scale=THUMB_SCALE, shift=ctuple(thumb_shift))
        acc += num

    # tracing point in thumbnail
    acc = complex(*THUMB_CENTER)
    r = 10
    r_c = r * (1 + 1j)
    draw.arc([ctuple(acc - r_c), ctuple(acc + r_c)], 0, 360, width=r+1, fill=BLACK)



    # mask off circles that escaped thumbnail
    pos = complex(*THUMB_POS)
    w = complex(*MASK_DIM)
    size = complex(*THUMB_SIZE)
    dim = complex(*DIM)
    draw.polygon([
        THUMB_POS, # inner TL
        ctuple(pos + size.real), # inner TR
        ctuple(pos + size), # inner BR
        ctuple(pos + 1j*size.imag), # inner BL
        (dim.real//2, dim.imag), # outer BL
        (dim.real//2, 0), # outer TL
        (dim.real, 0), # outer TR
        DIM, # outer BR
        (dim.real//2, dim.imag), # outer BL
        ctuple(pos + 1j*size.imag) # inner BL
     ], fill=BLACK)

    # draw main circles
    acc = 0
    for num in dft:
        draw_circle(num, acc, draw, shift=shift)
        acc += num

    # tracing point
    r = 5
    acc = total_sum.conjugate() + complex(*shift)
    r_c = r*(1 + 1j)
    draw.arc([ctuple(acc - r_c), ctuple(acc + r_c)], 0, 360, width=r+1, fill=BLACK)



def in_bounds(z: complex, b1: complex, b2: complex) -> bool:
    '''Check whether z is inside the rectangle formed by b1 and b2.'''
    return z.real > b1.real and z.real < b2.real and z.imag > b1.imag and z.imag < b2.imag


def ctuple(z: complex) -> tuple:
    '''Converts complex into tuple'''

    return (z.real, z.imag)

def draw_circle(num: complex, acc: complex, draw, scale=1, shift=(0, 0)):
    if scale > 1:
        num *= scale
        acc *= scale

    # radius, start, and end points
    r = abs(num)
    z1 = acc.conjugate() + complex(*shift)
    z2 = z1 + num.conjugate()

    # circle
    draw.arc([(z1.real - r, z1.imag - r), (z1.real + r, z1.imag + r)], 0, 360, width=2 if r > 100 else 1, fill=STANFORD_GREEN)

    # line part of the arrow
    draw.line((z1.real, z1.imag, z2.real, z2.imag), width=2, fill=STANFORD_GREEN)
    # arrowhead
    ah_points = []
    ah_points.append(z2)
    ah_points.append(z2 - (num.conjugate() * 0.1) * (1 + 0.5j))
    ah_points.append(z2 - (num.conjugate() * 0.1) * (1 - 0.5j))
    draw.polygon([ctuple(z) for z in ah_points], fill=STANFORD_GREEN)


def connect(shape, im):
    logo = Image.open('./resources/stanford/stanford_logo.png')
    m = 1.7
    logo = logo.resize((int(m*logo.size[0]), int(m*logo.size[1])));
    logo_shift = (-500, 200)
    im.paste(logo, (logo_shift[0], logo_shift[1], logo.size[0] + logo_shift[0], logo.size[1] + logo_shift[1]))

    draw = ImageDraw.Draw(im)
    i = 0
    for coords in shape:
        new_coords = []
        for c in coords:
            c = c.conjugate() + settings[i]['X_SHIFT'] + 1j*settings[i]['Y_SHIFT']
            new_coords.append(ctuple(c))
        draw.polygon(new_coords, fill=(255, 0, 0))
        i += 1

# fourier series coefficients
def fourier(x, X):
    # n element of Z
    terms = []
    N = len(dft)
    for k in range(-N//2 + 1, N//2 - 1):
        terms.append((1/N) * X[k] * cmath.exp(1j * 2*cmath.pi * k * x / N))

    return terms

def get_zoom(t: int) -> None:
    global THUMB_SIZE_ORIG
    if 0.4*t % 1 == 0:
        THUMB_SIZE_ORIG = (int(THUMB_SIZE_ORIG[0] - 0.4*t), int(THUMB_SIZE_ORIG[1] - 0.4*t))

def save_frames(frames, test_dir=TEST_DIR):
    if frames == []:
        return

    global vid_no
    global shape

    test_subdir = f'{test_dir}/test_{TEST_NO}'
    if not os.path.exists(test_subdir):
        os.makedirs(test_subdir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'avc1')

    print(f'\n\nMaking video {vid_no + 1}/{max(FRAMES_PER_SHAPE * len(shape) // MEMORY_THRESHOLD, len(shape))}...\n')
    vid = cv2.VideoWriter(f'{test_subdir}/vid_{vid_no:02d}.mp4', fourcc, 60, frames[0].size)

    for frame in tqdm(frames, unit='f'):
        cvim = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        vid.write(cvim)

    cv2.destroyAllWindows()
    vid.release()
    del vid

    frames = []
    vid_no += 1



#data = open('./uchicago_left.csv').read().split('\n')
#data.remove('')
paths = ['./data/stanford/letter.npy', './data/stanford/tree.npy']
settings = [
    {
        'X_SHIFT': 1130,
        'Y_SHIFT': 1150,
        'm': 1,
        'a': 336,
        'b': 465,
    },
    {
        'X_SHIFT': 1140,
        'Y_SHIFT': 1132,
        'm': 1,
        'a': 159,
        'b': 396,
    }
]

m = 1
#xdata = [c.real for c in coords]
#ydata = [c.imag for c in coords]

totals = []
counter = 1

shape = []
for i, path in enumerate(paths):
    b = settings[i]['b']
    a = settings[i]['a']
    m = settings[i]['m']
    coords = [m*complex(d[0]*a, d[1]*b) for d in np.load(path)]
    shape.append(coords)


frame_number = 0
# counts which shape is currently being rendered
frames = []
for i, coords in enumerate(shape):
    dft = np.fft.fft(coords)
    print(f'\n\nRendering frames for shape {i + 1}/{len(shape)}...\n')
    for t in tqdm(np.linspace(0, len(coords), FRAMES_PER_SHAPE), unit='f'):
        im = Image.open(BASE_FRAME)
        #connect(shape, im)
        f = fourier(t, dft)
        key = lambda c: abs(c)
        f.sort(key=key, reverse=True)
        render(f, im, (settings[i]['X_SHIFT'], settings[i]['Y_SHIFT']))
        frames.append(im)
        get_zoom(t)

        if frame_number > MEMORY_THRESHOLD:
            save_frames(frames)
            frames = []
            frame_number = 0
            print(f'\n\nRendering frames for shape {i + 1}/{len(shape) + 1}...\n')

        frame_number += 1
    save_frames(frames)
    frames = []
    # for debugging
    #sys.exit('debugging session ended... exiting')


concat_vid.concat(f'{TEST_DIR}/test_{TEST_NO}')


