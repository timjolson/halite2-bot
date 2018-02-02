import os
import numpy as np
import types
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import imread
from scipy.misc import imsave
from hlt.constants import *
import logging


class Struct(types.SimpleNamespace):
    """A structure to put fields into. Has .keys(), .values(), .items(), .pop(),
    .update() that all perform same as a dict(). Also has .setAll(value) that
    sets all fields in the object to 'value' and .clear() that removes all fields
    from the object.
    Struct is also subcriptable, and supports basic +/-, +=/-=, ==, !=.

    also supported:
        Struct() == dict()
        Struct() != dict()
        dict() == Struct()
        dict != Struct()
        Struct() +/- dict()
        Struct() +=/-= dict()

    obj = Struct(a=2, b=[3,4], c='hello')
    *OR
    obj = Struct({'a':2, 'b':[3,4], 'c':'hello'})

    Example usage after construction:
    obj.a += 1
    obj.d = 'world'
    obj.setItem(b, 42)
    obj['c'] = 3
    obj.update({'a':2})
    del obj['a']
    del obj.c
    obj.clear()

    Iterators
    for k, v in obj.items()
    for k in obj.keys()
    for v in obj.values()
    """

    def __init__(self, *arg, **kwargs):
        if arg and not kwargs:
            if len(arg) == 1 and type(arg[0]) == dict().__class__:
                self.update(*arg)
            else:
                raise Exception('Struct() takes either a dict or keyword arguments. {} provided'.format(type(arg[0])))
        elif kwargs and not arg:
            super().__init__(**kwargs)
        elif not arg and not kwargs:
            pass
        else:
            raise ValueError

    def __str__(self):
        if self.__dict__:
            out = [type(self).__name__ + '(']
            repr = []
            for k, v in self.__dict__.items():
                repr.append('{}=\'{}\''.format(k, v) if type(v) == type('') else '{}={}'.format(k, v))
            for field in repr[0:len(repr) - 1]:
                out += field
                out += ','
            out += repr[-1]
            out += ')'
            return ''.join(out)
        return type(self).__name__ + '()'

    def __repr__(self):
        return self.__str__()

    def setItem(self, key, value):
        self.__dict__.update({key:value})

    def setAll(self, value):
        self.__dict__.update({k:value for k in self.__dict__.keys()})

    def update(self, d):
        self.__dict__.update(d)

    def __add__(self, other):
        return dict(self.__dict__, **other)

    def __iadd__(self, other):
        self.__dict__.update(other)
        return self

    def __sub__(self, other):
        temp = dict(self.__dict__)
        for k in list(other.keys()):
            try:
                temp.pop(k)
            except KeyError:
                pass
        return Struct(temp)

    def __isub__(self, other):
        keys = list(other.keys())
        for k in list(self.__dict__.keys()):
            if k in keys:
                self.__dict__.pop(k)
        return self

    def __getitem__(self, key, failure=None):
        try:
            return self.__dict__.__getitem__(key)
        except KeyError:
            return failure

    def __setitem__(self, key, value):
        self.update({key:value})

    def __setattr__(self, key, value):
        self.update({key:value})

    def __delitem__(self, key):
        self.__dict__.__delitem__(key)

    def keys(self):
        for k in self.__dict__.keys():
            yield k

    def values(self):
        for v in self.__dic__.values():
            yield v

    def items(self):
        for k, v in self.__dict__.items():
            yield k,v

    def __eq__(self, other):
        keys = list(other.keys())
        for k, v in self.__dict__.items():
            if k not in keys or v != other[k]:
                return False
        return True

    def __ne__(self, other):
        return not self == other

    def pop(self, key, *args):
        if args:
            return self.__dict__.pop(key, *args)
        return self.__dict__.pop(key)

    def clear(self):
        self.__dict__.clear()


def circle(img, center, radius, color, thickness=None):
    height, width = img.shape
    xc, yc = center

    xc, yc, radius, thickness = int(xc), int(yc), int(radius), int(thickness) if thickness is not None else 0

    def fill_line(x0, x1, y):
        if y>=0 and y < height:
            minx, maxx = max(min(x0,width), 0), min(max(x1,0), width)
            img[y, minx:maxx] = color
    def color_pixel(x, y):
        if x >= 0 and y >= 0 and x < width and y < height:
            img[y][x] = color

    thickness = thickness if (thickness is not None and thickness != 0) else 1

    if thickness>1:
        circle(img, center, radius+thickness, color, -1)
        circle(img, center, radius, 0, -1)
    else:
        dx, dy = radius, 0
        err = 1 - dx

        if thickness == 1:
            while (dx >= dy):
                color_pixel(dx + xc, dy + yc)  # 1
                color_pixel(dy + xc, dx + yc)  # 2
                color_pixel(-dy + xc, dx + yc)  # 3
                color_pixel(-dx + xc, dy + yc)  # 4
                color_pixel(-dx + xc, -dy + yc)  # 5
                color_pixel(-dy + xc, -dx + yc)  # 6
                color_pixel(dx + xc, -dy + yc)  # 7
                color_pixel(dy + xc, -dx + yc)  # 8

                dy += 1
                if (err < 0):
                    err += 2 * dy + 1
                else:
                    dx -= 1
                    err += 2 * (dy - dx + 1)
        else:
            while (dx >= dy):
                fill_line(xc - dx, xc + dx, yc + dy)  # bottom big
                fill_line(xc - dx, xc + dx, yc - dy)  # top big
                fill_line(xc - dy, xc + dy, yc + dx)  # bottom small
                fill_line(xc - dy, xc + dy, yc - dx)  # top small

                dy += 1
                if (err < 0):
                    err += 2 * dy + 1
                else:
                    dx -= 1
                    err += 2 * (dy - dx + 1)
    return img


def blur(img, sigma):
    ret = gaussian_filter(img.astype(np.float), sigma)
    ret /= np.ptp(ret)
    return ret


def plot3d(x, y, z, fig=None, zlim=(-10,10)):
    '''obj = plot3d(...) ; obj.show()'''
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    x, y = np.meshgrid(x, y)
    ax.plot_surface(x, y, z)
    ax.set_zlim(*zlim)
    return fig


def flatten(base, *layers):
    for mask in layers[::-1]:
        startx, starty = np.int16(mask.x*PIXELS_PER_UNIT+mask.offset), np.int16(mask.y*PIXELS_PER_UNIT+mask.offset)
        endy = min(base.shape[0] - starty, mask.shape[0])
        endx = min(base.shape[1] - startx, mask.shape[1])
        # print('{},{}'.format(starty, startx))
        # print('{},{}'.format(endy, endx))

        # for r, row in enumerate(mask):
        #     R = r + starty
        #     # print('r {}'.format(R))
        #     if R>=0 and R<base.shape[0]:
        #         # print('row {}'.format(row))
        #         for c, val in enumerate(row):
        #             C = c + startx
        #             # print('c {}'.format(C))
        #             if C>=0 and C<base.shape[1]:
        #                 # print('val {}'.format(val))
        #                 # print('*** {} {} {}'.format(r+starty, c+startx, base[r+starty][c+startx] +val))
        #                 base[R].__setitem__(C, base[R][C] + val)

        [[base[r+starty].__setitem__(c+startx, base[r+starty][c+startx] + val)
           for c,val in enumerate(row) if c+startx>=0 and c+startx<base.shape[1]]
            for r, row in enumerate(mask) if r+starty>=0 and r+starty<base.shape[0]]

        # [print(row) for row in mask]
        # [[base[r+starty].__setitem__(c+startx, base[r+starty][c+startx]+val) for c, val in enumerate(row)] for r, row in enumerate(mask)]
        # [[[print(item) for item in row] for row in col] for col in mask]

        # for y in np.arange(0, min(base.shape[0] - starty, mask.shape[0]), 1):
        #     for x in np.arange(0, min(base.shape[1] - startx, mask.shape[1]), 1):
        #         if y + starty >= 0 and x + startx >= 0:
        #             base[y + starty][x + startx] += mask[y][x]

        # for y in np.arange(0, min(base.shape[0] - starty, mask.shape[0]), 1):
        #     y += starty
        #     if y >=0 :
        #         for x in np.arange(0, min(base.shape[1] - startx, mask.shape[1]), 1):
        #             x += startx
        #             if x >= 0:
        #                 base[y][x] += mask[y][x]

    return base


def check_file(name):
    return os.path.isfile(name)


def get_centroid(entities, width, height):
    cx, cy = 0.0, 0.0
    
    if entities:
        for entity in entities:
            cx += entity.x
            cy += entity.y
        cx /= len(entities)
        cy /= len(entities)
        if width:
            cx /= width
        if height:
            cy /= height
    
    return (cx, cy)
    