# Native detector

`fruit_detector.cpp` loads the deployed YOLOv8n NCNN graph once, converts RGBA to
RGB, resizes with aspect ratio, pads114, divides by255 and reads13x3549 output.
The first4rows are center/size boxes; the next9rows are class scores. External
class-aware NMS uses0.5IoU; score cutoff0.4; up to5regions. These settings are not
a calibrated accuracy guarantee. Coordinates are mapped back and clamped before
Kotlin crops the photo. CPU2threads; Vulkan and FP16arithmetic disabled. FP16
weight storage is enabled. JNI names retain source namespace com.fruitlens.app.

The CMake file reuses the project NCNN20261020 installation. No OpenCV or GPU
dependency is needed on the phone.
