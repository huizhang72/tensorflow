package(default_visibility = ["//visibility:private"])

licenses(["notice"])  # Apache 2.0

android_binary(
    name = "TfLiteCameraDemo",
    srcs = glob(["java/**/*.java"]),
    assets = [
        "//tensorflow/contrib/lite/java/demo/app/src/main/assets:labels_mobilenet_quant_v1_224.txt",
        "//tensorflow/contrib/lite/java/demo/app/src/main/assets:inceptionv3_slim_2016.tflite",
        "//tensorflow/contrib/lite/java/demo/app/src/main/assets:imagenet_slim_labels.txt",
        "//tensorflow/contrib/lite/java/demo/app/src/main/assets:labels_imagenet_slim.txt",
        "//tensorflow/contrib/lite/java/demo/app/src/main/assets:InceptionV3.cambricon",
        "//tensorflow/contrib/lite/java/demo/app/src/main/assets:labels_inceptionV3_cambricon.txt",
        "@tflite_mobilenet//:mobilenet_quant_v1_224.tflite",
    ],
    assets_dir = "",
    custom_package = "com.example.android.tflitecamerademo",
    manifest = "AndroidManifest.xml",
    nocompress_extensions = [
        ".tflite",
    ],
    resource_files = glob(["res/**"]),
    # In some platforms we don't have an Android SDK/NDK and this target
    # can't be built. We need to prevent the build system from trying to
    # use the target in that case.
    tags = ["manual"],
    deps = [
        "//tensorflow/contrib/lite/java:tensorflowlite",
        "//tensorflow/contrib/lite/java/src/main/libs:libhiai",
        "//tensorflow/contrib/lite/java/src/testhelper/java/org/tensorflow/lite:testhelper",
        "@androidsdk//com.android.support:support-v13-25.2.0",
        "@androidsdk//com.android.support:support-v4-25.2.0",
    ],
)
