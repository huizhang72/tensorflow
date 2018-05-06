/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

package com.example.android.tflitecamerademo;

import static android.graphics.Color.blue;
import static android.graphics.Color.green;
import static android.graphics.Color.red;

import android.app.Activity;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.util.Log;
import android.widget.Toast;
import com.huawei.hiaidemo.ModelManager;
import java.io.IOException;
import java.io.InputStream;

/**
 * This classifier works with the Inception-v3 slim model.
 * It applies floating point inference rather than using a quantized model.
 */
public class ImageClassifierNpuInception extends ImageClassifier {

  private static final String TAG = "ImageClassifierNpuInception";

  public static final Integer AI_OK = 0;
  public static final int GALLERY_REQUEST_CODE = 0;
  public static final int IMAGE_CAPTURE_REQUEST_CODE = 1;
  public static final int RESIZED_WIDTH = 299;
  public static final int RESIZED_HEIGHT = 299;
  public static final double meanValueOfBlue = 103.939;
  public static final double meanValueOfGreen = 116.779;
  public static final double meanValueOfRed = 123.68;

  private AssetManager assetManager;
  
  /**
   * The inception net requires additional normalization of the used input.
   */
  private static final int IMAGE_MEAN = 128;
  private static final float IMAGE_STD = 128.0f;

  /**
   * An array to hold inference results, to be feed into Tensorflow Lite as outputs.
   * This isn't part of the super class, because we need a primitive array here.
   */
  private float[][] labelProbArray = null;

  private String[] labels;

  /**
   * Initializes an {@code ImageClassifier}.
   *
   * @param activity
   */
  ImageClassifierNpuInception(Activity activity) throws IOException {
    super(activity);

    assetManager = activity.getAssets();
    
    labelProbArray = new float[1][getNumLabels()];

    boolean isSoLoadSuccess = ModelManager.init();
    if (isSoLoadSuccess) {
      Log.d(TAG, "ModelManager.init ok.");
    } else {
      Log.e(TAG, "ModelManager.init failed !!!.");
    }
    
    // init classify labels.
    initLabels(activity);

    // InceptionV3.cambricon.
    int ret = ModelManager.loadModelSync("InceptionV3", assetManager);
    Log.d(TAG, "loadModelSync " + ret);
  }

  @Override
  protected String getModelPath() {
    // you can download this file from
    return "inceptionv3_slim_2016.tflite";
  }

  @Override
  protected String getLabelPath() {
    return "labels_inceptionV3_cambricon.txt";
  }

  @Override
  protected int getImageSizeX() {
    return 299;
  }

  @Override
  protected int getImageSizeY() {
    return 299;
  }

  @Override
  protected int getNumBytesPerChannel() {
    // a 32bit float value requires 4 bytes
    return 4;
  }

  @Override
  protected void addPixelValue(int pixelValue) {
    imgData.putFloat((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
    imgData.putFloat((((pixelValue >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
    imgData.putFloat(((pixelValue & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
  }

  @Override
  protected float getProbability(int labelIndex) {
    return labelProbArray[0][labelIndex];
  }

  @Override
  protected void setProbability(int labelIndex, Number value) {
    labelProbArray[0][labelIndex] = value.floatValue();
  }

  @Override
  protected float getNormalizedProbability(int labelIndex) {
    // TODO the following value isn't in [0,1] yet, but may be greater. Why?
    return getProbability(labelIndex);
  }

  @Override
  protected void runInference(Bitmap bitmap) {
    float[] buffer = getPixel(bitmap, RESIZED_WIDTH, RESIZED_HEIGHT);
    labels = ModelManager.runModelSync("InceptionV3", buffer);
  }

  @Override
  protected String getModelName() {
    return "NPU";
  }
  /*
  @Override
  protected String printTopKLabels() {
    if (labels == null) {
      return "";
    }
    // Log.d(TAG, "0: " + labels[0] +  " 1: " + labels[1] +  " 2: " + labels[2]);
    return String.format("\n%s%s", labels[0], labels[1]);
    }*/

  /** Closes tflite to release resources. */
  @Override
  public void close() {
    ModelManager.unloadModelSync();
  }

  private float[] getPixel(Bitmap bitmap, int resizedWidth, int resizedHeight) {
    int channel = 3;
    float[] buff = new float[channel * resizedWidth * resizedHeight];

    int rIndex, gIndex, bIndex;
    int k = 0;
    for (int i = 0; i < resizedHeight; i++) {
      for (int j = 0; j < resizedWidth; j++) {
        bIndex = i * resizedWidth + j;
        gIndex = bIndex + resizedWidth * resizedHeight;
        rIndex = gIndex + resizedWidth * resizedHeight;

        int color = bitmap.getPixel(j, i);

        buff[rIndex] = (float) ((red(color) - meanValueOfRed))/255;
        buff[gIndex] = (float) ((green(color) - meanValueOfGreen))/255;
        buff[bIndex] = (float) ((blue(color) - meanValueOfBlue))/255;

      }
    }
    return buff;
  }

  private void initLabels(Activity activity) {
    Log.d(TAG, "initLabels ");
    byte[] labels;
    try {
      InputStream assetsInputStream = activity.getAssets().open(getLabelPath());
      int available = assetsInputStream.available();
      labels = new byte[available];
      assetsInputStream.read(labels);
      assetsInputStream.close();
      ModelManager.initLabels(labels);
    } catch (IOException e) {
      Log.e(TAG, "Read label failed. " + e);
    }
  }
}
