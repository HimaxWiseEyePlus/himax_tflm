# HIMAX TensorFlow Lite for Microcontrollers
It is a modified version of the [TensorFlow Lite for Microcontrollers](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/micro) for use with HIMAX WE-I Plus Boards. Each example in the package has been tested in Ubuntu 20 environment.

Following examples are included :
- magic wand example
- micro speech example
- person detection INT8 example
- handwriting example
  
## Table of contents
  - [Prerequisites](#prerequisites)
  - [Deploy to Himax WE1 EVB](#deploy-to-himax-we1-evb)
  - [Training your own model](#training-your-own-model)
  - [Convert model from PyTorch to TensorFlow Lite for Microcontrollers](#convert-model-from-pytorch-to-tensorflow-lite-for-microcontrollers)
   
## Prerequisites
- Make Tool version
  
  A `make` tool is required for deploying Tensorflow Lite Micro applications, See
[Check make tool version](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/tools/make/targets/arc/README.md#make-tool)
section for proper environment.
- MetaWare Development Toolkit

  See
[Install the Synopsys DesignWare ARC MetaWare Development Toolkit](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/tools/make/targets/arc/README.md#install-the-synopsys-designware-arc-metaware-development-toolkit)
section for instructions on toolchain installation.
- curl command
  
  Installing curl for Ubuntu Linux.
  ```
  sudo apt update
  sudo apt upgrade
  sudo apt install curl
  ```
- Serial Terminal Emulation Application

  There are 2 main purposes for HIMAX WE1 EVB Debug UART port, print application output and burn application to flash by using xmodem send application binary.

## Deploy to Himax WE1 EVB

The example project for HIMAX WE1 EVB platform can be generated with following command:

Download related third party data and model setting (only need to download one time)

```
make download
```

Build magic wand example and flash image, flash image name will be `magic_wand.img`

```
make magic_wand
make flash example=magic_wand
```

Build micro speech example and flash image, flash image name will be `micro_speech.img`

```
make micro_speech
make flash example=micro_speech
```

Build person detection INT8 example and flash image, flash image name will be `person_detection_int8.img`

```
make person_detection_int8
make flash example=person_detection_int8
```

Build handwriting example and flash image, flash image name will be `handwriting.img`. please check [here](tensorflow/lite/micro/examples/handwriting/README.md#handwriting-example) to know more about handwriting detail. 

```
make handwriting
make flash example=handwriting
```

After flash image generated, please download the flash image file to HIMAX WE1 EVB by UART, details are described [here](https://github.com/HimaxWiseEyePlus/bsp_tflu/tree/master/HIMAX_WE1_EVB_user_guide#flash-image-update-at-linux-environment)

## Training your own model

Model used by handwriting example is training with MNIST dataset, please take a look [here](tensorflow/lite/micro/examples/handwriting/training_a_model.md#train-handwriting-model-on-MNIST-dataset) about training flow if you are interested in.


## Convert model from PyTorch to TensorFlow Lite for Microcontrollers

Whenever there is a PyTorch model in your hand, [here](tensorflow/lite/micro/examples/handwriting/pytorch_onnx_tflite/README.md#convert-model-from-pytorch-to-tensorflow-lite-for-microcontrollers) is a toturial to switch it to tflite model and deploy on HIMAX WE1 EVB.

