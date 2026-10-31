# Measured architecture details

Read architecture.txt where present for every nested layer/module. Framework modules, Keras layers and optimized graph operations are different counts.

inspection.json:
{
  "executable_operator_count": 106,
  "operator_types": {
    "CONV_2D": 41,
    "HARD_SWISH": 18,
    "DEPTHWISE_CONV_2D": 11,
    "MEAN": 10,
    "MUL": 18,
    "ADD": 6,
    "FULLY_CONNECTED": 1,
    "SOFTMAX": 1
  },
  "kernel_and_bias_tensor_elements": 924414,
  "note": "Optimized graph counts. Folded constants and removed training layers mean this is not the original trainable parameter count."
}