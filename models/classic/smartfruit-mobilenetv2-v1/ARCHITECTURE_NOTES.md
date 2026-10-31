# Measured architecture details

Read architecture.txt where present for every nested layer/module. Framework modules, Keras layers and optimized graph operations are different counts.

inspection.json:
{
  "executable_operator_count": 66,
  "operator_types": {
    "CONV_2D": 35,
    "DEPTHWISE_CONV_2D": 17,
    "ADD": 10,
    "MEAN": 1,
    "FULLY_CONNECTED": 2,
    "SOFTMAX": 1
  },
  "kernel_and_bias_tensor_elements": 2371945,
  "note": "Optimized graph counts. Folded constants and removed training layers mean this is not the original trainable parameter count."
}