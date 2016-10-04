#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import numpy as np
import random

import os
import sys
import tarfile

import numpy
import six.moves.cPickle as pickle

from chainer.dataset import download
from chainer.datasets import tuple_dataset

def EXget_cifar10(withlabel=True, ndim=3, scale=1., mean=[0,0,0], std=[1,1,1]):
    """Gets the CIFAR-10 dataset.

    `CIFAR-10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ is a set of small
    natural images. Each example is an RGB color image of size 32x32,
    classified into 10 groups. In the original images, each component of pixels
    is represented by one-byte unsigned integer. This function scales the
    components to floating point values in the interval ``[0, scale]``.

    This function returns the training set and the test set of the official
    CIFAR-10 dataset. If ``withlabel`` is ``True``, each dataset consists of
    tuples of images and labels, otherwise it only consists of images.

    Args:
        withlabel (bool): If ``True``, it returns datasets with labels. In this
            case, each example is a tuple of an image and a label. Otherwise,
            the datasets only contain images.
        ndim (int): Number of dimensions of each image. The shape of each image
            is determined depending on ndim as follows:

            - ``ndim == 1``: the shape is ``(3072,)``
            - ``ndim == 3``: the shape is ``(3, 32, 32)``

        scale (float): Pixel value scale. If it is 1 (default), pixels are
            scaled to the interval ``[0, 1]``.

    Returns:
        A tuple of two datasets. If ``withlabel`` is ``True``, both datasets
        are :class:`~chainer.datasets.TupleDataset` instances. Otherwise, both
        datasets are arrays of images.

    """
    raw = _retrieve_cifar('cifar-10')
    train = EX_preprocess_cifar(raw['train_x'], raw['train_y'],
                              withlabel, ndim, scale, mean, std)
    test = EX_preprocess_cifar(raw['test_x'], raw['test_y'],
                             withlabel, ndim, scale, mean, std)
    return train, test

def EXget_cifar100(withlabel=True, ndim=3, scale=1., mean=[0,0,0], std=[1,1,1]):
    """Gets the CIFAR-100 dataset.

    `CIFAR-100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ is a set of
    small natural images. Each example is an RGB color image of size 32x32,
    classified into 100 groups. In the original images, each component
    pixels is represented by one-byte unsigned integer. This function scales
    the components to floating point values in the interval ``[0, scale]``.

    This function returns the training set and the test set of the official
    CIFAR-100 dataset. If ``withlabel`` is ``True``, each dataset consists of
    tuples of images and labels, otherwise it only consists of images.

    Args:
        withlabel (bool): If ``True``, it returns datasets with labels. In this
            case, each example is a tuple of an image and a label. Otherwise,
            the datasets only contain images.
        ndim (int): Number of dimensions of each image. The shape of each image
            is determined depending on ndim as follows:

            - ``ndim == 1``: the shape is ``(3072,)``
            - ``ndim == 3``: the shape is ``(3, 32, 32)``

        scale (float): Pixel value scale. If it is 1 (default), pixels are
            scaled to the interval ``[0, 1]``.

    Returns:
        A tuple of two datasets. If ``withlabel`` is ``True``, both
        are :class:`~chainer.datasets.TupleDataset` instances. Otherwise, both
        datasets are arrays of images.

    """
    raw = _retrieve_cifar_100()
    train = EX_preprocess_cifar(raw['train_x'], raw['train_y'],
                              withlabel, ndim, scale, mean, std)
    test = EX_preprocess_cifar(raw['test_x'], raw['test_y'],
                             withlabel, ndim, scale, mean, std)
    return train, test



def EX_preprocess_cifar(images, labels, withlabel, ndim, scale, mean, std):
    if ndim == 1:
        images = images.reshape(-1, 3072)
    elif ndim == 3:
        images = images.reshape(-1, 3, 32, 32)
    else:
        raise ValueError('invalid ndim for CIFAR dataset')
    images = images.astype(numpy.float32)
    images *= scale / 255.
    images[:,0] -= mean[0]
    images[:,1] -= mean[1]
    images[:,2] -= mean[2]
    images[:,0] /= std[0]
    images[:,1] /= std[1]
    images[:,2] /= std[2]
    pad_images=np.zeros((images.shape[0],images.shape[1],images.shape[2]+8,images.shape[3]+8),dtype=np.float32)
    pad_images[:,:,4:images.shape[2]+4,4:images.shape[3]+4] = images
    images = pad_images

    if withlabel:
        labels = labels.astype(numpy.int32)
        return tuple_dataset.TupleDataset(images, labels)
    else:
        return images


def _retrieve_cifar_100():
    root = download.get_dataset_directory('pfnet/chainer/cifar')
    path = os.path.join(root, 'cifar-100.npz')
    url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'

    def creator(path):

        def load(archive, file_name):
            d = _pickle_load(archive.extractfile(file_name))
            x = d['data'].reshape((-1, 3072))
            y = numpy.array(d['fine_labels'], dtype=numpy.uint8)
            return x, y

        archive_path = download.cached_download(url)
        with tarfile.open(archive_path, 'r:gz') as archive:
            train_x, train_y = load(archive, 'cifar-100-python/train')
            test_x, test_y = load(archive, 'cifar-100-python/test')

        numpy.savez_compressed(path, train_x=train_x, train_y=train_y,
                               test_x=test_x, test_y=test_y)
        return {'train_x': train_x, 'train_y': train_y,
                'test_x': test_x, 'test_y': test_y}

    return download.cache_or_load_file(path, creator, numpy.load)


def _retrieve_cifar(name):
    root = download.get_dataset_directory('pfnet/chainer/cifar')
    path = os.path.join(root, '{}.npz'.format(name))
    url = 'https://www.cs.toronto.edu/~kriz/{}-python.tar.gz'.format(name)

    def creator(path):
        archive_path = download.cached_download(url)

        train_x = numpy.empty((5, 10000, 3072), dtype=numpy.uint8)
        train_y = numpy.empty((5, 10000), dtype=numpy.uint8)
        test_y = numpy.empty(10000, dtype=numpy.uint8)

        dir_name = '{}-batches-py'.format(name)

        with tarfile.open(archive_path, 'r:gz') as archive:
            # training set
            for i in range(5):
                file_name = '{}/data_batch_{}'.format(dir_name, i + 1)
                d = _pickle_load(archive.extractfile(file_name))
                train_x[i] = d['data']
                train_y[i] = d['labels']

            # test set
            file_name = '{}/test_batch'.format(dir_name)
            d = _pickle_load(archive.extractfile(file_name))
            test_x = d['data']
            test_y[...] = d['labels']  # copy to array

        train_x = train_x.reshape(50000, 3072)
        train_y = train_y.reshape(50000)

        numpy.savez_compressed(path, train_x=train_x, train_y=train_y,
                               test_x=test_x, test_y=test_y)
        return {'train_x': train_x, 'train_y': train_y,
                'test_x': test_x, 'test_y': test_y}

    return download.cache_or_load_file(path, creator, numpy.load)


def _pickle_load(f):
    if sys.version_info > (3, ):
        # python3
        return pickle.load(f, encoding='latin-1')
    else:
        # python2
        return pickle.load(f)

class PreprocessedDataset(chainer.dataset.DatasetMixin):
    def __init__(self, pairs, random=False):
        self._pairs = pairs
        self._random = random

    def __len__(self):
        return len(self._pairs)

    def get_example(self, i):
        image, label = self._pairs[i]

        # load label data
        label = np.array(label, dtype=np.int32)

        # data augmentation
        if self._random:
            if random.randint(0, 1):
                image = image[:, :, ::-1]
            top = random.randint(0, 8)
            left = random.randint(0, 8)
            image = image[:, top:top+32, left:left+32]
        else:
            top = 4
            left = 4
            image = image[:, top:top+32, left:left+32]

        return image, label
