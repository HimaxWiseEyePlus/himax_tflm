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

#include "../../handwriting/detection_responder.h"
#include "../../handwriting/model_settings.h"
#include "hx_drv_tflm.h"

namespace {
uint8_t score_output[kCategoryCount];
}
void RespondToDetection(tflite::ErrorReporter* error_reporter, int8_t* score) {

  int maxIndex = -1;
  int maxScore = -255;

  TF_LITE_REPORT_ERROR(error_reporter, "number");
  for (int i = 0; i < kCategoryCount; i++) {
    TF_LITE_REPORT_ERROR(error_reporter, "[%d]:%d,",i,score[i]);
    if (score[i] > 0 && maxScore < score[i] ) {
      maxScore = score[i];
      maxIndex = i;
    }

    score_output[i] = score[i] + 128;
  }

  if(maxIndex!= -1)
    TF_LITE_REPORT_ERROR(error_reporter, "result:%d",maxIndex);
  else
    TF_LITE_REPORT_ERROR(error_reporter, "result:unknown");

  //send result data out through SPI
  hx_drv_spim_send((uint32_t)score_output, sizeof(int8_t) * kCategoryCount,
                   SPI_TYPE_META_DATA);
}
