import os
import mxnet as mx
from mxnet import nd, autograd, gluon
from mxnet.io import ImageRecordIter

REC = 'rec'
REAL_REC = 'real_rec.rec'
REAL_IDX = 'real_rec.idx'

class MxNetTrainer:
    def __init__(self):
        self.batch_size = 32
        self.learning_rate = 1e-3
        self.epochs = 100
        self.num_classes = 2

        self.ctx = mx.gpu()

        self.net = gluon.nn.Sequential()

        if os.path.exists(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', REC))):
            self.rec_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', REC))
        else:
            return

        if not os.path.exists(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'mxnet'))):
            os.makedirs(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'mxnet')))

        self.save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'mxnet'))

        if self.rec_path != None:
            self.train_data = ImageRecordIter(
                path_imgrec = os.path.join(self.rec_path, REAL_REC),
                path_imgidx = os.path.join(self.rec_path, REAL_IDX),
                data_shape = (3, 384, 384),
                shuffle = True,
                batch_size = self.batch_size
            )


    def model(self):
        with self.net.name_scope():
            self.net.add(gluon.nn.Conv2D(channels=32, kernel_size=5, activation='relu'))
            self.net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))

            self.net.add(gluon.nn.Conv2D(channels=64, kernel_size=5, activation='relu'))
            self.net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))

            self.net.add(gluon.nn.Conv2D(channels=128, kernel_size=5, activation='relu'))
            self.net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))

            self.net.add(gluon.nn.Flatten())

            self.net.add(gluon.nn.Dense(256, 'relu'))
            self.net.add(gluon.nn.Dense(64, 'relu'))
            self.net.add(gluon.nn.Dense(32, 'relu'))
            
            self.net.add(gluon.nn.Dense(self.num_classes))


    def evaluate_accuracy(self, data, label):
        acc = mx.metric.Accuracy()
        output = self.net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)

        return acc.get()[1]


    def train(self):
        self.model()

        self.net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=self.ctx)
        
        smoothing_constant = .01

        softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()    

        trainer = gluon.Trainer(self.net.collect_params(), 'adam', {'learning_rate': self.learning_rate})

        for e in range(self.epochs):
            i = 0
            self.train_data.reset()
            while self.train_data.iter_next():
                d = self.train_data.getdata() / 255.
                l = self.train_data.getlabel()

                data = d.as_in_context(self.ctx)
                label = l.as_in_context(self.ctx)

                step = data.shape[0]
                with autograd.record():
                    output = self.net(data)
                    loss = softmax_cross_entropy(output, label)
                
                loss.backward()

                trainer.step(step)
                ##########################
                #  Keep a moving average of the losses
                ##########################
                curr_loss = nd.mean(loss).asscalar()
                moving_loss = (curr_loss if ((i == 0) and (e == 0))
                    else (1 - smoothing_constant) * moving_loss + smoothing_constant * curr_loss)
                

                acc = self.evaluate_accuracy(data, label)
                print("Epoch {:03d} ... Dataset {:03d} ... ".format(e+1, i), "Loss = {:.4f}".format(curr_loss), " Moving Loss = {:.4f}".format(moving_loss), " Accuracy = {:.4f}".format(acc))

                # self.summary_writer.add_histogram(tag='accuracy', values=acc, global_step=e)

                i += 1

            # self.summary_writer.add_scalar(tag='moving_loss', value=moving_loss, global_step=e)

        self.save_path = os.path.join(self.save_path, 'model.params')
        self.net.save_parameters(self.save_path)
