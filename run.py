import os
import argparse
from mxNet.train import MxNetTrainer
from tensorFlow.train import TfTrain

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--framework", required=False, help="Framework to be used for training, e.g. tf (for TensorFlow) or mx (for mxnet)", default='tf', choices=('tf', 'mx'))
    ap.add_argument('-m', '--mode', required=True, help='Specify whether you want to perform training or inference. e.g. -m inference', default='training')
    ap.add_argument('-s', '--squeeze', required=False, help='Flag for building a SqueezeNet', default=True, type=bool, choices=(True, False))
    ap.add_argument('-r', '--resume', required=False, help='Resume training or start a new training', default=False, type=bool, choices=(True, False))

    args = vars(ap.parse_args())

    if args['framework'] == 'tf':
        tfTrain = TfTrain(args['resume'])
        if args['squeeze']:
            tfTrain.set_squeeze(True)
        tfTrain.train()
    elif args['framework'] == 'mx':
        mxNetTrainer = MxNetTrainer()
        mxNetTrainer.train()