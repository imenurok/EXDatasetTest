#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
#print chainer.__version__
from chainer.datasets import cifar
from chainer import serializers
from chainer import training
from chainer.training import extensions

import math
import numpy
import time

import cmd_options
import dataset
from dataset import PreprocessedDataset
from densenet import DenseNet
from graph import create_fig

def main(args):

    assert((args.depth - args.block - 1) % args.block == 0)
    n_layer = (args.depth - args.block - 1) / args.block
    
    if args.dataset == 'cifar10':
        mean = numpy.asarray((125.3,123.0,113.9))#from fb.resnet.torch
        std = numpy.asarray((63.0, 62.1, 66.7))# Did the std data computed from 0 padding images?
        train, test = dataset.EXget_cifar10(scale=255,mean=mean,std=std)
        
        n_class = 10
    elif args.dataset == 'cifar100':
        mean = numpy.asarray((129.3,124.1,112.4))#from fb.resnet.torch
        std = numpy.asarray((68.2, 65.4, 70.4))
        train, test = dataset.EXget_cifar100(scale=255,mean=mean,std=std)
        
        n_class = 100
    elif args.dataset == 'SVHN':
        raise NotImplementedError()

    train = PreprocessedDataset(train, random=True)
    test = PreprocessedDataset(test)

    train_iter = chainer.iterators.MultiprocessIterator(train, args.batchsize)
    test_iter = chainer.iterators.MultiprocessIterator(
        test, args.batchsize, repeat=False, shuffle=False)

    model = chainer.links.Classifier(DenseNet(
        n_layer, args.growth_rate, n_class, args.drop_ratio, 16, args.block))
    if args.init_model:
        serializers.load_npz(args.init_model, model)

    import EXoptimizers
    optimizer = EXoptimizers.EXNesterovAG(lr=args.lr / len(args.gpus), momentum=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))

    devices = {'main': args.gpus[0]}
    if len(args.gpus) > 2:
        for gid in args.gpus[1:]:
            devices['gpu%d' % gid] = gid
    updater = training.ParallelUpdater(train_iter, optimizer, devices=devices)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.dir)

    val_interval = (1, 'epoch')
    log_interval = (1, 'epoch')

    eval_model = model.copy()
    eval_model.train = False

    trainer.extend(extensions.Evaluator(
        test_iter, eval_model, device=args.gpus[0]), trigger=val_interval)

    import EXextensions
    trainer.extend(EXextensions.cifarShift('lr', (0.1,0.01,0.001), (150,225)), trigger=val_interval)#default epoch<=150 lr = 0.1, 150<epoch<=225 lr = 0.01, else lr = 0.001
    
    trainer.extend(extensions.snapshot_object(
        model, 'epoch_{.updater.epoch}.model'), trigger=val_interval)
    trainer.extend(extensions.snapshot_object(
        optimizer, 'epoch_{.updater.epoch}.state'), trigger=val_interval)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    start_time = time.time()
    trainer.extend(extensions.observe_value(
        'time', lambda _: time.time() - start_time), trigger=log_interval)
    trainer.extend(extensions.PrintReport([
        'time', 'epoch', 'iteration', 'main/loss', 'validation/main/loss',
        'main/accuracy', 'validation/main/accuracy', 'lr',
    ]), trigger=log_interval)
    trainer.extend(extensions.observe_value(
        'graph', lambda _: create_fig(args.dir)), trigger=(2, 'epoch'))
    trainer.extend(extensions.ProgressBar(update_interval=10))

    trainer.run()

if __name__ == '__main__':
    args = cmd_options.get_arguments()
    main(args)
