# Functions to create image channels from entities
import os
import numpy as np
from copy import deepcopy
import logging
from hlt.utils import imsave, imread, circle, blur, check_file
from hlt.constants import *


class Images():
    @staticmethod
    def blank_map(shape):
        return Images.Layer(shape)

    class Layer(list):
        def __init__(self, shape, x=0.0, y=0.0, val=0.0, data=None):
            if data is None:
                # print('data = None')
                super(Images.Layer, self).__init__()
                cols, rows = shape[1], shape[0]
                self.extend([[val for _ in range(cols)] for _ in range(rows)])
            else:
                # print('data != None')
                super(Images.Layer, self).__init__(data)

            self.x, self.y = x, y
            self.shape = shape
            self.offset = -1 * np.int16(shape[0] / 2)

        def __mul__(self, scalar):
            # try:
            return Images.Layer(
                self.shape,
                x=self.x, y=self.y,
                data= [[self[x][y] * scalar for y in range(self.shape[1])] for x in range(self.shape[0])]
                )
            # except:
            #     logging.exception('')
            #     raise()
            # return Images.Layer(self.shape, data=np.zeros(self.shape).tolist())

        def pos(self, x, y):
            self.x, self.y = x, y
            return self

        @staticmethod
        def from_np(data, x, y, val=None):
            assert isinstance(data, np.ndarray), "provide numpy array"
            # print(data.shape)
            if val is None:
                # print('val = None')
                return Images.Layer(list(data.shape), x, y, data=data)
            else:
                # print('val != None')
                return Images.Layer(list(data.shape), x, y, val)

    class Ship():
        back_img, armed_img, unarmed_img, friendly_img = None, None, None, None

        @classmethod
        def back(cls):
            if cls.back_img is None:
                logging.debug('made back')
                cls.back_img = np.zeros(SHIP_SHAPE, np.float)
                Images.save_ship(cls.back_img, 'b')
            return cls.back_img

        @classmethod
        def armed(cls, x=0, y=0):
            if cls.armed_img is None:
                r = np.uint16(WEAPON_RADIUS * PIXELS_PER_UNIT * .8)
                cls.armed_img = deepcopy(cls.back())
                circle(cls.armed_img, SHIP_CENTER, r, 1.0, -1)
                cls.armed_img = blur(cls.armed_img, r*1.1)
                Images.save_ship(cls.armed_img, 'a')
                logging.debug('made armed')
            return Images.Layer.from_np(cls.armed_img, x, y)

        @classmethod
        def _create_unarmed(cls):
            if cls.unarmed_img is None:
                r = np.uint16(SHIP_RADIUS * PIXELS_PER_UNIT)
                cls.unarmed_img = deepcopy(cls.back())
                circle(cls.unarmed_img, SHIP_CENTER, max(1, r), 1.0, -1)
                cls.unarmed_img = blur(cls.unarmed_img, r + 6)
                Images.save_ship(cls.unarmed_img, 'u')
                logging.debug('made unarmed')
            return cls.unarmed_img

        @classmethod
        def unarmed(cls, x=0, y=0):
            return Images.Layer.from_np(cls._create_unarmed(), x, y)

        @classmethod
        def friendly(cls, x=0, y=0):
            if cls.friendly_img is None:
                r = SHIP_RADIUS * PIXELS_PER_UNIT
                img = deepcopy(cls._create_unarmed())
                circle(img, SHIP_CENTER, np.uint16(r*3.0), 1.0, -1)
                img = blur(img, r + 5)
                cls.friendly_img = circle(img, SHIP_CENTER, np.uint16(r*2.5), 0.0, -1)
                Images.save_ship(cls.friendly_img, 'f')
                logging.debug('made friendly')
            return Images.Layer.from_np(cls.friendly_img, x, y)

    class Planet():
        back_img, full_img, dock_img = dict(), dict(), dict()
        shape, center = dict(), dict()

        @classmethod
        def get_shapes(cls, r):
            # max shape of a planet
            rr = round(r,1)
            if rr not in cls.shape.keys():
                logging.debug('made shape[{}]'.format(rr))
                cls.shape.update({rr:[int(r*3 +
                         DOCK_RADIUS*4)*PIXELS_PER_UNIT] * 2})
                cls.center.update({rr:tuple([int(cls.shape[rr][0] / 2)] * 2)})
            return rr

        @classmethod
        def back(cls, r):
            rr = cls.get_shapes(r)
            if rr not in cls.back_img.keys():
                logging.debug('made back[{}]'.format(rr))
                cls.back_img.update({rr: np.zeros(cls.shape[rr], np.float)})
                Images.save_planet(cls.back_img[rr],'b',rr)
            return cls.back_img[rr]

        @classmethod
        def full(cls, r, x=0, y=0):
            rr = cls.get_shapes(r)
            if rr not in cls.full_img.keys():
                logging.debug('made full[{}]'.format(rr))
                img = deepcopy(cls.back(rr))
                R = np.uint16(r * PIXELS_PER_UNIT)
                circle(img, cls.center[rr], R, 1.0, -1)
                cls.full_img.update({rr:blur(img, r*1.3 + 6)})
                Images.save_planet(cls.full_img[rr], 'f', rr)
            return Images.Layer.from_np(cls.full_img[rr], x, y)

        @classmethod
        def dockable(cls, r, x=0, y=0):
            rr = cls.get_shapes(r)
            if rr not in cls.dock_img.keys():
                logging.debug('made dockable[{}]'.format(rr))
                img = deepcopy(cls.back(rr))
                R = np.uint16((r + DOCK_RADIUS) * PIXELS_PER_UNIT)
                circle(img, cls.center[rr], R, 1.0, -1)
                img = blur(img, r*1.2 + 6)
                circle(img, cls.center[rr],
                       np.uint16(max(r - DOCK_RADIUS/2,2) * PIXELS_PER_UNIT),
                       0.0, -1)
                cls.dock_img.update({rr: img})
                Images.save_planet(cls.dock_img[rr], 'd', rr)
            return Images.Layer.from_np(cls.dock_img[rr], x, y)


    @classmethod
    def get_path(cls):
        if __name__ != '__main__':
            path = "hlt/img/{}/".format(PIXELS_PER_UNIT)
        else:
            path = "img/{}/".format(PIXELS_PER_UNIT)
        directory = os.path.dirname(path)
        try:
            os.stat(directory)
        except:
            os.mkdir(directory)

        return path

    @classmethod
    def load(cls):
        logging.debug('loading images')

        for t in cls.ship_dict.keys():
            cls.generate_ship(t)

        for t in cls.planet_dict.keys():
            for r in list(range(40, 200)):
                cls.generate_planet(t,r/10)

    ship_dict = dict(
        {'b': Ship.back, 'u': Ship.unarmed, 'a': Ship.armed, 'f': Ship.friendly})
    planet_dict = dict({'b': Planet.back, 'f': Planet.full, 'd': Planet.dockable})

    @classmethod
    def generate_ship(cls, type):
        path = cls.get_path()
        name = '{}S{}.bmp'.format(path, type)

        # make and save or read-in image
        if not check_file(name):
            img = np.array(cls.ship_dict[type]())
            logging.debug('created image {}'.format(name))
            imsave(name, img * 255)
        else:
            logging.debug('read image {}'.format(name))
            img = imread(name, 0)

        # save image into class attributes
        if type == 'a':
            cls.Ship.armed_img = img
        elif type == 'u':
            cls.Ship.unarmed_img = img
        elif type == 'b':
            cls.Ship.back_img = img
        elif type == 'f':
            cls.Ship.friendly_img = img

    @classmethod
    def generate_planet(cls, type, r):
        path = cls.get_path()
        name = '{}P{}{}.bmp'.format(path, type, r)

        # make and save or read-in image
        if not check_file(name):
            img = np.array(cls.planet_dict[type](r))
            logging.debug('created image {}'.format(name))
            imsave(name, img * 255)
        else:
            logging.debug('read image {}'.format(name))
            img = imread(name, 0)

        # save image into class attributes
        if type == 'f':
            cls.Planet.full_img[r] = img
        elif type == 'd':
            cls.Planet.dock_img[r] = img
        elif type == 'b':
            cls.Planet.back_img[r] = img

    @classmethod
    def save_ship(cls, img, type):
        logging.debug('saving ship')

        if __name__ != '__main__':
            path = "hlt/img/{}/".format(PIXELS_PER_UNIT)
        else:
            path = "img/{}/".format(PIXELS_PER_UNIT)
        directory = os.path.dirname(path)
        try:
            os.stat(directory)
        except:
            os.mkdir(directory)

        name = '{}S{}.bmp'.format(path, type)

        # make and save or read-in image
        if not check_file(name):
            logging.debug('saving ship image {}'.format(name))
            imsave(name, img * 255)
        else:
            logging.debug('ship {} exists'.format(name))

    @classmethod
    def save_planet(cls, img, type, r):
        logging.debug('saving planet')

        if __name__ != '__main__':
            path = "hlt/img/{}/".format(PIXELS_PER_UNIT)
        else:
            path = "img/{}/".format(PIXELS_PER_UNIT)
        directory = os.path.dirname(path)
        try:
            os.stat(directory)
        except:
            os.mkdir(directory)

        name = '{}P{}{}.bmp'.format(path, type, r)

        # make and save or read-in image
        if not check_file(name):
            logging.debug('saving ship image {}'.format(name))
            imsave(name, img * 255)
        else:
            logging.debug('ship {} exists'.format(name))

if __name__ == '__main__':
    Images.load()
