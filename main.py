#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: main.py
# Created Date: Tuesday October 1st 2019
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Saturday, 5th October 2019 9:17:18 am
# Modified By: Chen Xuanhong
# Copyright (c) 2019 Shanghai Jiao Tong University
#############################################################




from    parameter import *
from    trainer import Trainer
# from    tester import Tester
from    dataTool.dataLoader import DataLoader
from    torch.backends import cudnn
from    utilities.Utilities import makeFolder
import  torch

def main(config):
    # For fast training
    cudnn.benchmark = True


    # Data loader
    data_loader = DataLoader(config.train, config.dataset, config.image_path, config.imsize,
                             config.batch_size, shuf=config.train)

    # Create directories if not exist
    makeFolder(config.model_save_path, config.version)
    makeFolder(config.sample_path, config.version)
    makeFolder(config.log_path, config.version)


    if config.train:
        trainer = Trainer(data_loader.loader(), config)
        trainer.train()
    else:
        tester = Tester(data_loader.loader(), config)
        tester.test()

if __name__ == '__main__':
    config = getParameters()
    main(config)