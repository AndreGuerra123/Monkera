"""Utilities for real-time data augmentation on image data.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import re
from six.moves import range
import os
import threading
import warnings
import multiprocessing.pool
from functools import partial

from keras import backend
from keras import utils as keras_utils
from keras.preprocessing.image import Iterator

from PIL import ImageEnhance
from PIL import Image as pil_image

import io
import pymongo
from PIL import Image

import pydash as p_
import collections

import keras
import random

_PIL_INTERPOLATION_METHODS = {
    'nearest': pil_image.NEAREST,
    'bilinear': pil_image.BILINEAR,
    'bicubic': pil_image.BICUBIC,
    'hamming': pil_image.HAMMING,
    'box': pil_image.BOX,
    'lanczos': pil_image.LANCZOS}

color_formats = {
            'L':1,
            'RGB':3,
            'RGBA':4
        }

def random_rotation(x, rg, row_axis=1, col_axis=2, channel_axis=0,
                    fill_mode='nearest', cval=0.):
    """Performs a random rotation of a Numpy image tensor.

    # Arguments
        x: Input tensor. Must be 3D.
        rg: Rotation range, in degrees.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.

    # Returns
        Rotated Numpy image tensor.
    """
    theta = np.random.uniform(-rg, rg)
    x = apply_affine_transform(x, theta=theta, channel_axis=channel_axis,
                               fill_mode=fill_mode, cval=cval)
    return x


def random_shift(x, wrg, hrg, row_axis=1, col_axis=2, channel_axis=0,
                 fill_mode='nearest', cval=0.):
    """Performs a random spatial shift of a Numpy image tensor.

    # Arguments
        x: Input tensor. Must be 3D.
        wrg: Width shift range, as a float fraction of the width.
        hrg: Height shift range, as a float fraction of the height.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.

    # Returns
        Shifted Numpy image tensor.
    """
    h, w = x.shape[row_axis], x.shape[col_axis]
    tx = np.random.uniform(-hrg, hrg) * h
    ty = np.random.uniform(-wrg, wrg) * w
    x = apply_affine_transform(x, tx=tx, ty=ty, channel_axis=channel_axis,
                               fill_mode=fill_mode, cval=cval)
    return x


def random_shear(x, intensity, row_axis=1, col_axis=2, channel_axis=0,
                 fill_mode='nearest', cval=0.):
    """Performs a random spatial shear of a Numpy image tensor.

    # Arguments
        x: Input tensor. Must be 3D.
        intensity: Transformation intensity in degrees.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.

    # Returns
        Sheared Numpy image tensor.
    """
    shear = np.random.uniform(-intensity, intensity)
    x = apply_affine_transform(x, shear=shear, channel_axis=channel_axis,
                               fill_mode=fill_mode, cval=cval)
    return x


def random_zoom(x, zoom_range, row_axis=1, col_axis=2, channel_axis=0,
                fill_mode='nearest', cval=0.):
    """Performs a random spatial zoom of a Numpy image tensor.

    # Arguments
        x: Input tensor. Must be 3D.
        zoom_range: Tuple of floats; zoom range for width and height.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.

    # Returns
        Zoomed Numpy image tensor.

    # Raises
        ValueError: if `zoom_range` isn't a tuple.
    """
    if len(zoom_range) != 2:
        raise ValueError('`zoom_range` should be a tuple or list of two'
                         ' floats. Received: ', (zoom_range,))

    if zoom_range[0] == 1 and zoom_range[1] == 1:
        zx, zy = 1, 1
    else:
        zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
    x = apply_affine_transform(x, zx=zx, zy=zy, channel_axis=channel_axis,
                               fill_mode=fill_mode, cval=cval)
    return x


def apply_channel_shift(x, intensity, channel_axis=0):
    """Performs a channel shift.

    # Arguments
        x: Input tensor. Must be 3D.
        intensity: Transformation intensity.
        channel_axis: Index of axis for channels in the input tensor.

    # Returns
        Numpy image tensor.

    """
    x = np.rollaxis(x, channel_axis, 0)
    min_x, max_x = np.min(x), np.max(x)
    channel_images = [
        np.clip(x_channel + intensity,
                min_x,
                max_x)
        for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x


def random_channel_shift(x, intensity_range, channel_axis=0):
    """Performs a random channel shift.

    # Arguments
        x: Input tensor. Must be 3D.
        intensity_range: Transformation intensity.
        channel_axis: Index of axis for channels in the input tensor.

    # Returns
        Numpy image tensor.
    """
    intensity = np.random.uniform(-intensity_range, intensity_range)
    return apply_channel_shift(x, intensity, channel_axis=channel_axis)


def apply_brightness_shift(x, brightness):
    """Performs a brightness shift.

    # Arguments
        x: Input tensor. Must be 3D.
        brightness: Float. The new brightness value.
        channel_axis: Index of axis for channels in the input tensor.

    # Returns
        Numpy image tensor.

    # Raises
        ValueError if `brightness_range` isn't a tuple.
    """

    x = array_to_img(x)
    x = imgenhancer_Brightness = ImageEnhance.Brightness(x)
    x = imgenhancer_Brightness.enhance(brightness)
    x = img_to_array(x)
    return x


def random_brightness(x, brightness_range):
    """Performs a random brightness shift.

    # Arguments
        x: Input tensor. Must be 3D.
        brightness_range: Tuple of floats; brightness range.
        channel_axis: Index of axis for channels in the input tensor.

    # Returns
        Numpy image tensor.

    # Raises
        ValueError if `brightness_range` isn't a tuple.
    """
    if len(brightness_range) != 2:
        raise ValueError(
            '`brightness_range should be tuple or list of two floats. '
            'Received: %s' % (brightness_range,))

    u = np.random.uniform(brightness_range[0], brightness_range[1])
    return apply_brightness_shift(x, u)


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_affine_transform(x, theta=0, tx=0, ty=0, shear=0, zx=1, zy=1,
                           row_axis=0, col_axis=1, channel_axis=2,
                           fill_mode='nearest', cval=0.):
    """Applies an affine transformation specified by the parameters given.

    # Arguments
        x: 2D numpy array, single image.
        theta: Rotation angle in degrees.
        tx: Width shift.
        ty: Heigh shift.
        shear: Shear angle in degrees.
        zx: Zoom in x direction.
        zy: Zoom in y direction
        row_axis: Index of axis for rows in the input image.
        col_axis: Index of axis for columns in the input image.
        channel_axis: Index of axis for channels in the input image.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.

    # Returns
        The transformed version of the input.
    """

    transform_matrix = None
    if theta != 0:
        theta = np.deg2rad(theta)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        transform_matrix = rotation_matrix

    if tx != 0 or ty != 0:
        shift_matrix = np.array([[1, 0, tx],
                                 [0, 1, ty],
                                 [0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = shift_matrix
        else:
            transform_matrix = np.dot(transform_matrix, shift_matrix)

    if shear != 0:
        shear = np.deg2rad(shear)
        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = shear_matrix
        else:
            transform_matrix = np.dot(transform_matrix, shear_matrix)

    if zx != 1 or zy != 1:
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = zoom_matrix
        else:
            transform_matrix = np.dot(transform_matrix, zoom_matrix)

    if transform_matrix is not None:
        h, w = x.shape[row_axis], x.shape[col_axis]
        transform_matrix = transform_matrix_offset_center(
            transform_matrix, h, w)
        x = np.rollaxis(x, channel_axis, 0)
        final_affine_matrix = transform_matrix[:2, :2]
        final_offset = transform_matrix[:2, 2]

        channel_images = [scipy.ndimage.interpolation.affine_transform(
            x_channel,
            final_affine_matrix,
            final_offset,
            order=1,
            mode=fill_mode,
            cval=cval) for x_channel in x]
        x = np.stack(channel_images, axis=0)
        x = np.rollaxis(x, 0, channel_axis + 1)
    return x


def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


def array_to_img(x, data_format=None, scale=True):
    """Converts a 3D Numpy array to a PIL Image instance.

    # Arguments
        x: Input Numpy array.
        data_format: Image data format.
            either "channels_first" or "channels_last".
        scale: Whether to rescale image values
            to be within `[0, 255]`.

    # Returns
        A PIL Image instance.

    # Raises
        ImportError: if PIL is not available.
        ValueError: if invalid `x` or `data_format` is passed.
    """
    x = np.asarray(x, dtype=backend.floatx())
    if x.ndim != 3:
        raise ValueError('Expected image array to have rank 3 (single image). '
                         'Got array with shape:', x.shape)

    if data_format is None:
        data_format = backend.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Invalid data_format:', data_format)

    # Original Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but target PIL image has format (width, height, channel)
    if data_format == 'channels_first':
        x = x.transpose(1, 2, 0)
    if scale:
        x = x + max(-np.min(x), 0)
        x_max = np.max(x)
        if x_max != 0:
            x /= x_max
        x *= 255
    if x.shape[2] == 4:
        # RGBA
        return pil_image.fromarray(x.astype('uint8'), 'RGBA')
    elif x.shape[2] == 3:
        # RGB
        return pil_image.fromarray(x.astype('uint8'), 'RGB')
    elif x.shape[2] == 1:
        # grayscale
        return pil_image.fromarray(x[:, :, 0].astype('uint8'), 'L')
    else:
        raise ValueError('Unsupported channel number: ', x.shape[2])


def img_to_array(img, data_format=None):
    """Converts a PIL Image instance to a Numpy array.

    # Arguments
        img: PIL Image instance.
        data_format: Image data format,
            either "channels_first" or "channels_last".

    # Returns
        A 3D Numpy array.

    # Raises
        ValueError: if invalid `img` or `data_format` is passed.
    """
    if data_format is None:
        data_format = backend.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: ', data_format)
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = np.asarray(img, dtype=backend.floatx())
    if len(x.shape) == 3:
        if data_format == 'channels_first':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if data_format == 'channels_first':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError('Unsupported image shape: ', x.shape)
    return x


def _g(obj, loc):
    return p_.get(obj, loc)


def _g_d(obj, loc, default):
    return p_.get(obj, loc, default=default)


def _g_d_a_t(obj, loc, default, typ, msg):
    pro = _g_d(obj, loc, default)
    if pro != default:
        assert type(pro) is typ, msg
    return pro

def _a_dic(object, msg):
    assert (type(object) is dict), msg


def _a_g(obj,loc,msg):
    pro = _g(obj,loc)
    assert pro != None, msg
    return pro

class MongoImageDataGenerator(object):
    def __init__(self,
                 connection={'host': "localhost",
                             'port': 12721,
                             'database': "database",
                             'collection': "collection"},
                 query={},
                 location={'image': "image", 'label': "label"},
                 config={
                     'batch_size': 1,
                     'shuffle': True,
                     'seed': None,
                     'target_size': (32, 32),
                     'data_format': 'channels_last',
                     'color_format': 'RGB',
                     'validation_split': 0.},
                 stand={
                     'center': False,
                     'normalize': False,
                     'rescale': 1/255,
                     'preprocessing_function': None,
                 },
                 affine={
                     'rounds': 0,
                     'transform': False,
                     'random': False,
                     'keep_original': False,
                     'rotation': 0.,
                     'width_shift': 0.,
                     'height_shift': 0.,
                     'shear': 0.,
                     'channel_shift': 0.,
                     'brightness': 1.,
                     'zoom': 0.,
                     'horizontal_flip': False,
                     'vertical_flip': False,
                     'fill_mode': 'nearest',
                     'cval': 0.
                 }
                 ):
        # Check if inputs are dictionary objects
        _a_dic(connection, "Please select a valid connection dictionary")
        _a_dic(query, "Please select a valid query dictionary")
        _a_dic(location, "Please select a valid location dictionary")
        _a_dic(config, "Please select a valid config dictionary")
        _a_dic(stand, "Please select a valid stand dictionary")
        _a_dic(affine, "Please select a valida affine dictionary")
        # Check for type an load
        self.host = _g_d_a_t(connection, 'host', 'localhost', str,
                             "Please provide a valid string for mongodb hostname.")
        self.port = _g_d_a_t(connection, 'port', 12721, int,
                             "Please provide a valid integer for mongodb port.")
        self.database = _g_d_a_t(connection, 'database',"database", str,
                               "Please provide a valid string for mongodb database.")
        self.collection = _g_d_a_t(connection, 'collection',"collection", str,
                                 "Please provide a valid string for mongodb collection.")

        self.query = query

        self.img_location = _g_d_a_t(
            location, 'image',"image", str, "Please provide a valid location for the image binary field in the selected mongodb collection.")
        self.lbl_location = _g_d_a_t(
            location, 'label',"label", str, "Please provide a valid location for the label field in the selected mongodb collection.")

        self.batch_size = _g_d_a_t(
            config, 'batch_size', 1 , int, "Please select a valid integer value for the batchsize parameter.")
        assert (self.batch_size >= 1), "Batch size must be a positive integer (at least 1)."

        self.shuffle = _g_d_a_t(config, 'shuffle', True, bool,
                                "Please select a valid boolean value for the shuffle parameter.")

        self.seed = _g_d_a_t(config, 'seed', None, int,
                             "Please select a valid integer value for the seed parameter.")
        
        self.target_size = _g_d_a_t(config, 'target_size', (32,32), tuple, 'Please select a valid tuple value (height,width) for the target size parameter.')
        self.height = self.target_size[0]
        self.width = self.target_size[1]
        assert (self.height > 0), "Height must be positive integer, check target size."
        assert (self.width > 0), "Width must be positive integer, check target size."

        self.validation_split = _g_d_a_t(config, 'validation_split', 0., float,
                                         "Please select a valid float value for the validation split parameter.")
        assert (self.validation_split >= 0 and self.validation_split <1), "Validation split parameter must be between 0 and 1"

        self.data_format = _g_d_a_t(config, 'data_format', backend.image_data_format(
        ), str, "Please select a valid string for the data_format parameter.")
        assert (self.data_format in ['channels_first', 'channels_last']
                ), 'Data format parameter should be `"channels_last"` (channel after row and column) or `"channels_first"` (channel before row and column). Received: %s' % self.data_format

        if self.data_format == 'channels_first':
            self.channel_axis = 1
            self.row_axis = 2
            self.col_axis = 3
        if self.data_format == 'channels_last':
            self.channel_axis = 3
            self.row_axis = 1
            self.col_axis = 2

        self.color_format = _g_d_a_t(config, 'color_format', 'RGB', str,"Please select a valid float value for the validation split parameter.")
        assert self.color_format in color_formats, "Please select a valid color format string: L, RGB, or RGBA"
        self.color_shape = _g(color_formats,self.color_format)

        self.center = _g_d_a_t(stand, 'center', False, bool,
                                          "Please select a valid boolean value for the samplewise_center parameter.")
        self.normalize = _g_d_a_t(stand, 'normalize', False,
                                                     bool, "Please select a valid boolean value for the normalize parameter.")

        if self.normalize:
            if not self.center:
                self.center = True
                warnings.warn('This MongoDataGenerator specifies '
                              '`normalization`, '
                              'which overrides setting of '
                              '`center`.')


        self.rescale = _g_d_a_t(stand, 'rescale', 1/255., float,
                                "Please select a valid float value for the rescale parameter.")
        assert (self.rescale != 0), "Rescaling with factor 0 will reset all data. Please update rescaling parameter to 1 if no rescaling is required."
        self.preprocessing_function = _g_d(
            stand, 'preprocessing_function', None)
        assert ((self.preprocessing_function == None) | (type(self.preprocessing_function)
                is 'function')), "Preprocessing function is not a valid parameter."

       
        self.rounds = _g_d_a_t(
            affine, 'rounds', 0, int, "Please select a valid integer value for the rounds parameter.")
        self.transform = _g_d_a_t(affine, 'transform', False, bool,
                                  "Please select a valid boolean value for the transform parameter.")
        self.random = _g_d_a_t(affine, 'random', False, bool,
                                  "Please select a valid boolean value for the random parameter.")
        self.keep_original = _g_d_a_t(affine, 'keep_original', False, bool,
                                      "Please select a valid boolean value for the keep_original parameter.")

        self.rotation = _g_d_a_t(
            affine, 'rotation', 0., float, "Please select a valid float value for the rotation parameter.")
        self.width_shift = _g_d_a_t(
            affine, 'width_shift', 0., float, "Please select a valid float value for the width_shift parameter.")
        self.height_shift = _g_d_a_t(
            affine, 'height_shift', 0., float, "Please select a valid float value for the height_shift parameter.")
        self.shear = _g_d_a_t(
            affine, 'shear', 0., float, "Please select a valid float value for the shear parameter.")
        self.channel_shift = _g_d_a_t(
            affine, 'channel_shift', 0., float, "Please select a valid float value for the channel_shift parameter.")
        self.brightness = _g_d_a_t(affine, 'brightness', 1., float,
                                   "Please select a valid float value for the brightness parameter.")
        assert (0 < self.brightness <= 1) , "Please select a value superior to zero (black image) for the brightness parameter"

        self.zoom = _g_d_a_t(affine, 'zoom', 0.,float,"Please select a valid folat value for the zoom parameter")
        assert (0 <= self.zoom <= 2) , "Please select a value superior to zero (black image) for the brightness parameter"

        

        self.horizontal_flip = _g_d_a_t(affine, 'horizontal_flip', False, bool,
                                        "Please select a valid boolean value for the horizontal_flip parameter.")
        self.vertical_flip = _g_d_a_t(affine, 'vertical_flip', False, bool,
                                      "Please select a valid boolean value for the vertical_flip parameter.")
        self.fill_mode = _g_d_a_t(affine, 'fill_mode', 'nearest', str,
                                  "Please select a valid str value for the fill_mode parameter.")
        self.cval = _g_d_a_t(affine, 'cval', 0., float,
                             "Please select a valid float value for the cval parameter.")

        self.dtype = backend.floatx()

    def flows_from_mongo(self):

        # Initial Connection to retrieve samples and OBIDS

        self.object_ids = self.__getOBIDS(self.query)
        assert (len(self.object_ids) > 0), "The resulted query returned zero(0) samples."

        self.dictionary, self.classes = self.__getDictionary()
        assert (self.classes > 1), "The resulted query return insufficient distinct classes."

        # Split Obids into train and validation
        self.train_obids, self.test_obids = self.partitioning()

        self.train_samples = len(self.train_obids)
        self.test_samples = len(self.test_obids)

        return (MongoTrainFlowGenerator(self), MongoTestFlowGenerator(self))

    def partitioning(self):


        many = int(round(self.validation_split*len(self.object_ids)))

        if self.shuffle:
            random.shuffle(self.object_ids)
    
        return self.object_ids[many:], self.object_ids[:many]

    def standardize(self, x):
        """Applies the normalization configuration to a batch of inputs.

        # Arguments
            x: Batch of inputs to be normalized.

        # Returns
            The inputs, normalized.
        """
        if self.preprocessing_function:
            x = self.preprocessing_function(x)
        if self.rescale:
            x *= self.rescale
        if self.center:
            x -= np.mean(x, keepdims=True)
        if self.normalize:
            x /= (np.std(x, keepdims=True) + backend.epsilon())

        return x

    def get_random_transform(self, img_shape, seed=None):
        """Generates random parameters for a transformation.

        # Arguments
            seed: Random seed.
            img_shape: Tuple of integers.
                Shape of the image that is transformed.

        # Returns
            A dictionary containing randomly chosen parameters describing the
            transformation.
        """
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1

        if seed is not None:
            np.random.seed(seed)

        if self.rotation != 0:
            theta = np.random.uniform(-self.rotation,self.rotation)
        else:
            theta = 0

        if self.height_shift != 0:
            tx = np.random.uniform(-self.height_shift, self.height_shift)
            if np.max(self.height_shift) < 1:
                tx *= img_shape[img_row_axis]
        else:
            tx = 0

        if self.width_shift != 0:
            ty = np.random.uniform(-self.width_shift,self.width_shift)
            if np.max(self.width_shift) < 1:
                ty *= img_shape[img_col_axis]
        else:
            ty = 0

        if self.shear != 0:
            shear = np.random.uniform(
                -self.shear,
                self.shear)
        else:
            shear = 0

        if self.zoom == 0.:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(1-self.zoom,1+self.zoom,2)

        flip_horizontal = (np.random.random() < 0.5) * self.horizontal_flip

        flip_vertical = (np.random.random() < 0.5) * self.vertical_flip

        channel_shift_intensity = None
        if self.channel_shift != 0:
            channel_shift_intensity = np.random.uniform(-self.channel_shift, self.channel_shift)

        brightness = None
        if self.brightness != 1:
            brightness = np.random.uniform(1-self.brightness, self.brightness)

        transform_parameters = {'theta': theta,
                                'tx': tx,
                                'ty': ty,
                                'shear': shear,
                                'zx': zx,
                                'zy': zy,
                                'flip_horizontal': flip_horizontal,
                                'flip_vertical': flip_vertical,
                                'channel_shift_intensity': channel_shift_intensity,
                                'brightness': brightness}

        return transform_parameters

    def get_exact_transform(self, img_shape, seed=None):
        """Generates random parameters for a transformation.

        # Arguments
            seed: Random seed.
            img_shape: Tuple of integers.
                Shape of the image that is transformed.

        # Returns
            A dictionary containing randomly chosen parameters describing the
            transformation.
        """
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1

        if seed is not None:
            np.random.seed(seed)

        theta = self.rotation
        
        if self.height_shift != 0:
            tx = self.height_shift
            if np.max(self.height_shift) < 1:
                tx *= img_shape[img_row_axis]
        else:
            tx = 0

        if self.width_shift != 0:
            ty = self.width_shift
            if np.max(self.width_shift) < 1:
                ty *= img_shape[img_col_axis]
        else:
            ty = 0

        if self.shear != 0:
            shear = self.shear
        else:
            shear = 0

        if self.zoom == 0.:
            zx, zy = 1, 1
        else:
            zx, zy = (1-self.zoom,1+self.zoom)

        flip_horizontal = self.horizontal_flip

        flip_vertical = self.vertical_flip

        channel_shift_intensity = None
        if self.channel_shift != 0:
            channel_shift_intensity = self.channel_shift

        brightness = None
        if self.brightness != 1:
            brightness = self.brightness

        transform_parameters = {'theta': theta,
                                'tx': tx,
                                'ty': ty,
                                'shear': shear,
                                'zx': zx,
                                'zy': zy,
                                'flip_horizontal': flip_horizontal,
                                'flip_vertical': flip_vertical,
                                'channel_shift_intensity': channel_shift_intensity,
                                'brightness': brightness}

        return transform_parameters

    def apply_transform(self, x, transform_parameters):
        """Applies a transformation to an image according to given parameters.

        # Arguments
            x: 3D tensor, single image.
            transform_parameters: Dictionary with string - parameter pairs
                describing the transformation.
                Currently, the following parameters
                from the dictionary are used:
                - `'theta'`: Float. Rotation angle in degrees.
                - `'tx'`: Float. Shift in the x direction.
                - `'ty'`: Float. Shift in the y direction.
                - `'shear'`: Float. Shear angle in degrees.
                - `'zx'`: Float. Zoom in the x direction.
                - `'zy'`: Float. Zoom in the y direction.
                - `'flip_horizontal'`: Boolean. Horizontal flip.
                - `'flip_vertical'`: Boolean. Vertical flip.
                - `'channel_shift_intencity'`: Float. Channel shift intensity.
                - `'brightness'`: Float. Brightness shift intensity.

        # Returns
            A transformed version of the input (same shape).
        """
        # x is a single image, so it doesn't have image number at index 0
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1
        img_channel_axis = self.channel_axis - 1

        x = apply_affine_transform(x, transform_parameters.get('theta', 0),
                                   transform_parameters.get('tx', 0),
                                   transform_parameters.get('ty', 0),
                                   transform_parameters.get('shear', 0),
                                   transform_parameters.get('zx', 1),
                                   transform_parameters.get('zy', 1),
                                   row_axis=img_row_axis, col_axis=img_col_axis,
                                   channel_axis=img_channel_axis,
                                   fill_mode=self.fill_mode, cval=self.cval)

        if transform_parameters.get('channel_shift_intensity') is not None:
            x = apply_channel_shift(x,
                                    transform_parameters['channel_shift_intensity'],
                                    img_channel_axis)

        if transform_parameters.get('flip_horizontal', False):
            x = flip_axis(x, img_col_axis)

        if transform_parameters.get('flip_vertical', False):
            x = flip_axis(x, img_row_axis)

        if transform_parameters.get('brightness') is not None:
            x = apply_brightness_shift(x, transform_parameters['brightness'])

        return x

    def random_transform(self, x, seed=None):
        """Applies a random transformation to an image.

        # Arguments
            x: 3D tensor, single image.
            seed: Random seed.

        # Returns
            A randomly transformed version of the input (same shape).
        """
        params = self.get_random_transform(x.shape, seed)
        return self.apply_transform(x, params)

    def exact_transform(self,x, seed=None):
        """Applies a exact transformation to an image.

        # Arguments
            x: 3D tensor, single image.
            seed: Random seed.

        # Returns
            A randomly transformed version of the input (same shape).
        """
        params = self.get_exact_transform(x.shape, seed)
        return self.apply_transform(x, params)
 

    def transform_test_batch(self,x,y):
        return self.standardize(x),y

    def transform_train_batch(self, x, y):
        
        x, y = self.reform(x,y) #Augmentation, tranformations and replications
        return self.standardize(x), y
        
    def reform(self,x,y):
        xx = np.copy(x)
        yy = np.copy(y)

        if self.rounds > 0:
            if self.transform:
                if self.random:
                    xx,yy = self.atr(xx,yy)
                else:
                    xx,yy = self.atn(xx,yy)
            else:
                return an(xx,yy)
                 
        else:
            if self.transform:
                if self.random:
                    xx,yy = self.ntr(xx,yy)
                else:
                    xx,yy = self.ntn(xx,yy)
            else:
                return xx, yy

        if self.keep_original:

            xx,yy = self.keeper(x,xx,y,yy)

        return xx,yy

    def atr(self,xx,yy):
         size = self.getBatchSize(xx)
         ax = np.asarray([self.random_transform(xx[i]) for i in range(size) for r in range(self.rounds)])
         ay = np.asarray([yy[i] for i in range(size) for r in range(self.rounds)])
         return ax, ay
                    
    def atn(self,xx,yy):
         size = self.getBatchSize(xx)
         ax = np.asarray([self.exact_transform(xx[i]) for i in range(size) for r in range(self.rounds)])
         ay = np.asarray([yy[i] for i in range(size) for r in range(self.rounds)])
         return ax, ay

    def an(self,xx,yy):
         size = self.getBatchSize(xx)
         ax = np.asarray([xx[i] for i in range(size) for r in range(self.rounds)])
         ay = np.asarray([yy[i] for i in range(size) for r in range(self.rounds)])
         return ax, ay

    def ntr(self,xx,yy):
         size = self.getBatchSize(xx)
         ax = np.asarray([self.random_transform(xx[i]) for i in range(size)])
         ay = np.asarray([yy[i] for i in range(size)])
         return ax, ay

    def ntn(self,xx,yy):
         size = self.getBatchSize(xx)
         ax = np.asarray([self.exact_transform(xx[i]) for i in range(size)])
         ay = np.asarray([yy[i] for i in range(size)])
         return ax, ay

    def getBatchSize(self,x):
        return x.shape[0]

    def keeper(x,xx,y,yy):
        return np.concatenate(x,xx,axis=0), np.concatenate(y,yy,axis=0)

    def __getOBIDS(self, query):
        collection = self.__connect()
        object_ids = collection.distinct("_id", query)
        self.__disconnect(collection)
        return object_ids

    def __getDictionary(self):
        collection = self.__connect()
        lbls = collection.distinct(
            self.lbl_location, {'_id': {'$in': self.object_ids}})
        nb = len(lbls)
        # keys as human readable, any type.
        dictionary = {k: self.__hot(v, nb) for v, k in enumerate(lbls)}
        self.__disconnect(collection)
        return dictionary, nb

    def __hot(self, idx, nb):
        hot = np.zeros((nb,))
        hot[idx] = 1
        return hot

    def readMongoSample(self, oid):

        collection = self.__connect()
        sample = collection.find_one({'_id': oid})
        assert sample != None, "Failed to retrieve the sample corresponding to the image ID: " + \
            str(oid)
        return (self.__getImage(sample), self.__getLabel(sample))

    def __connect(self):
        return pymongo.MongoClient(self.host, self.port)[self.database][self.collection]

    def __disconnect(self, collection):
        del collection
        collection = None

    def __getImage(self, sample):
        strg = _a_g(sample, self.img_location, "Failed to retrieve image binary (ID:" +
                   str(_g(sample, '_id'))+") at "+self.img_location+".")
        img = Image.open(io.BytesIO(strg)).resize(
            self.target_size).convert(self.color_format)
        return img_to_array(img,data_format=self.data_format)

    def __getLabel(self, sample):
        label = _a_g(sample, self.lbl_location,
                    "Failed to retrieve image label (ID:"+str(_g(sample, '_id'))+") at "+self.lbl_location+".")
        return self.getEncoded(label)

    def getShape(self):
        return (self.height, self.width, self.color_shape)

    def getClassNumber(self):
        return self.classes

    def getEncoded(self, label):
        return _g(self.dictionary, label)

    def getDecoded(self, np):
        return self.dictionary.keys()[self.dictionary.values().index(np)]

class MongoTrainFlowGenerator(Iterator):

    def __init__(self, mdig):

        self.mdig = mdig
        self.n = self.mdig.train_samples
        self.batch_size = self.mdig.batch_size
        self.shuffle = self.mdig.shuffle
        self.seed = self.mdig.seed
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_array = None
        self.index_generator = self._flow_index()
    
    def _get_batches_of_transformed_samples(self, index_array):

        batch_x = np.empty((len(index_array), self.mdig.height, self.mdig.width, self.mdig.color_shape), dtype=self.mdig.dtype)

        batch_y = np.empty((len(index_array), self.mdig.classes), dtype=self.mdig.dtype)

        for i, j in enumerate(index_array):

                # Get sample data
            (x, y) = self.mdig.readMongoSample(self.mdig.train_obids[j])

            # Add the image and the label to the batch (one-hot encoded).
            batch_x[i] = x
            batch_y[i] = y

        (batchx, batchy) = self.mdig.transform_train_batch(batch_x, batch_y)

        return batchx, batchy

    def next(self):
        return self._get_batches_of_transformed_samples(next(self.index_generator))

class MongoTestFlowGenerator(Iterator):
        
    def __init__(self, mdig):

        self.mdig = mdig
        self.n = self.mdig.test_samples
        self.batch_size = self.mdig.batch_size
        self.shuffle = self.mdig.shuffle
        self.seed = self.mdig.seed
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_array = None
        self.index_generator = self._flow_index()

     
    def _get_batches_of_transformed_samples(self, index_array):

        batch_x = np.empty((len(index_array), self.mdig.height, self.mdig.width, self.mdig.color_shape), dtype=self.mdig.dtype)

        batch_y = np.empty((len(index_array), self.mdig.classes), dtype=self.mdig.dtype)

        for i, j in enumerate(index_array):

                # Get sample data
            (x, y) = self.mdig.readMongoSample(self.mdig.test_obids[j])

            # Add the image and the label to the batch (one-hot encoded).
            batch_x[i] = x
            batch_y[i] = y

        (batchx, batchy) = self.mdig.transform_test_batch(batch_x, batch_y)

        return batchx, batchy

    def next(self):
            return self._get_batches_of_transformed_samples(next(self.index_generator))
 
