# Measured architecture details

Read architecture.txt where present for every nested layer/module. Framework modules, Keras layers and optimized graph operations are different counts.

inspection.json:
{
  "executable_graph_nodes": 240,
  "node_types": {
    "Input": 1,
    "Convolution": 67,
    "Pooling": 1,
    "Split": 33,
    "ConvolutionDepthWise": 43,
    "Concat": 33,
    "ShuffleChannel": 16,
    "Crop": 26,
    "Interp": 2,
    "BinaryOp": 5,
    "Slice": 4,
    "Sigmoid": 4,
    "Reshape": 4,
    "Permute": 1
  },
  "original_trainable_parameters": null,
  "note": "NCNN optimized graph node counts; original trainable parameter count not independently established for this export."
}