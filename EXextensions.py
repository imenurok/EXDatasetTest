from __future__ import division

from chainer.training import extension


class cifarShift(extension.Extension):

    """
    It's imenurok's original extension (2016/09/16)

    Args:
        attr (str): Name of the optimizer attribute to adjust.
        value_range (tuple of float): The first, second and the last values of the
            attribute.
        time_range (tuple of ints): The first and second counts of calls in which
            the attribute is adjusted.
        optimizer (~chainer.Optimizer): Target optimizer object. If it is None,
            the main optimizer of the trainer is used.

    """
    invoke_before_training = True

    def __init__(self, attr, value_range, time_range, optimizer=None):
        self._attr = attr
        self._value_range = value_range
        self._time_range = time_range
        self._optimizer = optimizer
        self._t = 0

    def __call__(self, trainer):
        optimizer = self._optimizer or trainer.updater.get_optimizer('main')
        t1, t2 = self._time_range
        v1, v2, v3 = self._value_range

        if self._t <= t1-1: #if 1 <= epoch <= t1, lr = v1
            value = v1
        elif self._t <= t2-1: #if t1 < epoch <= t2, lr = v2
            value = v2
        else: #if t2 < epoch, lr = v3
            value = v3
        setattr(optimizer, self._attr, value)

        self._t += 1

    def serialize(self, serializer):
        self._t = serializer('_t', self._t)

