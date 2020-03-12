# DeepFake Detection
This repository is an attempt to develop a DeepLearning model which can discriminate between fake and original videos. It is based on the [Deepfake Detection Challenge](https://www.kaggle.com/c/deepfake-detection-challenge) organized by Kaggle. 


### Setup:
Before jumping into training a model, you need to make sure that your environment is ready. Please make sure that the following dependencies are installed.
1. TensorFlow, MxNet, OpenCV
2. Pandas (For CSV file processing)
3. tqdm (Progress bar, check out pip tqdm package)


### Dataset:
On Kaggle you will see that the entire dataset is compressed into one single zip file called `all.zip` which is 471 GB in size and the same zip file is split into 50 chunks of separate zip files, each 10 GB in size. Please download those chunks instead of downloading the single entire blob and extract those chunks into a directory called `data` in the root of the repository. After extraction your `data` directory should look something like this, `./data/dfdc_train_part_0, ./data/dfdc_train_part_1, ./data/dfdc_train_part_2`.

Once you extract the dataset in the appropriate directory, we are ready to preprocess the dataset.

### Preprocessing
We'll be training our models using both TensorFlow and MxNet. Also, we'll be exploiting MxNet's `ImageRecordIter` API to store and preprocess the dataset. At times while training a model, your disk I/O becomes a bottleneck which impacts the time it takes to train a model. To avoid such bottlenecks, we'll use `ImageRecordIter` in both TensorFlow and MxNet based training.

```
python mxnet_preprocess.py
```

This script will take care of preprocessing the dataset. It creates raw `JPEG` images from the videos and also, creates the `.rec`, `.idx`, `.lst` files which are used by `ImageRecordIter`.


### Training:
In order to train the dataset, simply run the following script

```
python run.py -f mx -m training
```

#### Command Line Arguments
Please provided the following arguments to the `run.py` script.
1. `-f` : Specifies the framework to be used. `mx` for MxNet, `tf` for TensorFlow.
2. `-m` : Specifies the operation mode. `training` to train a model and `inference` to run inference.

### TODO
- [ ] Finish the TensorFlow based training pipeline
- [ ] Finish inference pipeline for MxNet
- [ ] Finish inference pipeline for TensorFlow
- [ ] Train TensorFlow model using quantization so that it can be deployed on Android and iOS devices.