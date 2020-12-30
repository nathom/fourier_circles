import cv2
import numpy as np
from time import sleep
from PIL import Image, ImageDraw

x_shift = 500
y_shift = 500
scale = 200

def find_contour(path: str, i: int) -> str:
    filename = path.split('/')[-1].replace('.png', '')
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    contours, heirarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    print(f'{len(contours)=}')
# use second for bottom
    x = contours[i][:, 0, 0]*1.0
    y = contours[i][:, 0, 1]*1.0
    x -= np.mean(x)
    y -= np.mean(y)
    x /= np.std(x)
    y /= np.std(y)
    y = -y

    points = np.stack([x, y], axis=1)

    data_path = f"./data/stanford/{filename}.npy"
    np.save(data_path, points)
    return data_path

def display_contour(*paths, image:str=None) -> None:
    if image is None:
        im = Image.new('RGB', (3840, 2160))
    else:
        im = Image.open(image)

    # add logo to image for reference
    logo = Image.open('./resources/stanford/stanford_logo.png')
    m = 1.7
    logo = logo.resize((int(m*logo.size[0]), int(m*logo.size[1])));
    logo_shift = (-500, 200)
    im.paste(logo, (logo_shift[0], logo_shift[1], logo.size[0] + logo_shift[0], logo.size[1] + logo_shift[1]))

    draw = ImageDraw.Draw(im)
    for path in paths:
        points = np.load(path['path'])
        coords = map(lambda p: (p[0]*path['scale'][0] + path['xshift'], -p[1]*path['scale'][1] - path['yshift']), points)
        #draw.polygon(list(coords))

    im.save("./tests/display_contour.png" if image is None else image, 'PNG')


m = 3
# tree is i=2
# S is i=3
# inner S is i=4
tree = {
    'path': find_contour('./resources/stanford/tree.png', 2),
    'xshift': 1140,
    'yshift': -1132,
    'scale': (m*53, m*132)
}


letter = {
    'path': find_contour('./resources/stanford/letter_S.png', 0),
    'xshift': 1130,
    'yshift': -1150,
    'scale': (m*112, m*155)
}
display_contour(letter)







