# Intel-DevCloud-vs-Google-Colab-Which-is-Faster-for-Machine-Learning-
This repository contains code for training a convolutional neural network on the Chest X-Ray dataset to classify images as normal or pneumonia. The same code was run on two different platforms: Intel DevCloud and Google Colab. The runtime of the first two epochs was recorded on both platforms, and the results are presented below.


## Dataset

The Chest X-Ray dataset contains 5,856 images of chest x-rays, labeled as either normal or pneumonia. The dataset is split into training, validation, and test sets. The training set contains 5,216 images, the validation set contains 16 images, and the test set contains 624 images.

## Model

The model used for training is a convolutional neural network with three convolutional layers, each followed by a max pooling layer, and two fully connected layers. The model is trained using binary cross-entropy loss and the Adam optimizer.

## Hardware Specifications

The two platforms used for training have different hardware specifications, which could potentially impact the runtime of the model. Here are the specifications of each platform:

### Intel DevCloud

The Intel DevCloud is a cloud-based platform for developing and testing AI solutions. It provides access to a variety of hardware accelerators, including CPUs, GPUs, and FPGAs. For this experiment, we used an Intel Xeon Platinum 8268 CPU with 48 cores and 96 threads, and 192 GB of RAM.

### Google Colab

Google Colab is a free, cloud-based platform for running Jupyter notebooks. It provides access to a GPU and 12 GB of RAM. For this experiment, we used a Tesla T4 GPU with 16 GB of VRAM.

## Results

The model was trained for 2 epochs on both Intel DevCloud and Google Colab. The table below shows the runtime of the first two epochs on each platform.

| Platform        | Epoch 1 | Epoch 2 |
| ---------------|---------|---------|
| Intel DevCloud  | 99.59 s | 98.80 s |
| Google Colab    | 382.39 s| 373.95 s|

To visualize the differences in runtime more clearly, we have also included a barplot:

![barplot runtime](https://user-images.githubusercontent.com/111365771/235187635-eb6add6f-e8ab-4b3a-8801-8d2188455fcc.png)


As you can see, the runtime of the first two epochs on Intel DevCloud is consistently around 99 seconds, while on Google Colab it is consistently around 382 seconds. This suggests that Intel DevCloud is significantly faster for training machine learning models, at least for this particular dataset and model architecture.

### Screenshot of Colab model Runtime

![colab run time](https://user-images.githubusercontent.com/111365771/235187893-0d9844ba-c45d-4b72-81a8-9bb23dca621d.png)


### Screenshot of Devcloud model Runtime
![oneAPI runtime](https://user-images.githubusercontent.com/111365771/235186925-a7fb6dfc-9dd7-487b-ad9c-45c12f1a4a03.png)

## Code

To reproduce these results, you can run the `train.py` script on both Intel DevCloud and Google Colab. The script requires the Chest X-Ray dataset, which can be downloaded from Kaggle [here](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia). The script should be run with Python 3 and the following dependencies:

- TensorFlow 2.4.1
- NumPy 1.19.5
- OpenCV 4.1.2.30
- Gradio 1.6.4

## Live Implementation

https://user-images.githubusercontent.com/111365771/235189170-93984d8a-e188-4b2b-997a-c3c45d43b015.mp4

