/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "../../handwriting/image_provider.h"

#include "../../handwriting/model_settings.h"
#include "hx_drv_tflm.h"

namespace {
hx_drv_sensor_image_config_t g_pimg_config;
}

TfLiteStatus GetImage(tflite::ErrorReporter* error_reporter, int image_width,
                      int image_height, int channels, int8_t* image_data) {
  static bool is_initialized = false;

  if (!is_initialized) {
    g_pimg_config.sensor_type = HX_DRV_SENSOR_TYPE_HM0360_MONO;
    g_pimg_config.format = HX_DRV_VIDEO_FORMAT_YUV400;
    g_pimg_config.img_width = 640;
    g_pimg_config.img_height = 480;
    if (hx_drv_sensor_initial(&g_pimg_config) != HX_DRV_LIB_PASS) {
      return kTfLiteError;
    }

    if (hx_drv_spim_init() != HX_DRV_LIB_PASS) {
      return kTfLiteError;
    }

    is_initialized = true;
  }

  //capture image by sensor
  hx_drv_sensor_capture(&g_pimg_config);

  //send jpeg image data out through SPI
  hx_drv_spim_send(g_pimg_config.jpeg_address, g_pimg_config.jpeg_size,
                   SPI_TYPE_JPG);

  hx_drv_image_rescale((uint8_t*)g_pimg_config.raw_address,
                       g_pimg_config.img_width, g_pimg_config.img_height,
                       image_data, image_width, image_height);

  return kTfLiteOk;
}
