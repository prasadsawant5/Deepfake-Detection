import os
import argparse
from mxNet.train import MxNetTrainer
from tensorFlow.train import TfTrain

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--framework", required=True, help="Framework to be used for training, e.g. tf (for TensorFlow) or mx (for mxnet)", default='tf')
    ap.add_argument('-m', '--mode', required=True, help='Specify whether you want to perform training or inference. e.g. -m inference', default='training')

    args = vars(ap.parse_args())

    if args['framework'] == 'tf':
        tfTrain = TfTrain()
        tfTrain.train()
    elif args['framework'] == 'mx':
        mxNetTrainer = MxNetTrainer()
        mxNetTrainer.train()