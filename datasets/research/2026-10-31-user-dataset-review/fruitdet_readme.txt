---
license: cc-by-4.0
task_categories:
- object-detection
language:
- en
---

# FruitDet – Object Detection Dataset

A community-driven fruit image dataset annotated for object detection, [original repo](https://huggingface.co/datasets/sirunchained/fruit-dataset).  
The images are real-world fruit photos collected from various environments. All images have been resized to **920×1080** pixels, and each fruit instance is labeled with a bounding box in **YOLO format**.

## Features

- Real-world images from diverse sources
- 19 fruit/vegetable categories, each containing 30 images
- All images resized to **920×1080** pixels
- **YOLO‑format annotations**: each image has a corresponding `.txt` file with bounding box coordinates
- Dataset split into **train** (20 images), **val** (5 images), and **test** (5 images) per class
- Each class folder includes a `classes.txt` file listing the category names (one per split)
- Standardized naming convention (`{class}_0000.jpg` and `{class}_0000.txt`)
- Clean folder structure, ready for YOLO-based frameworks (YOLOv5, YOLOv8, YOLOv9, etc.)

## Annotation Format

Bounding boxes are stored in the standard YOLO format:

class_id x_center y_center width height


- `class_id` – integer index of the class (0‑based)
- `x_center`, `y_center` – normalized coordinates of the box center (0–1)
- `width`, `height` – normalized dimensions of the box (0–1)

Each image file (e.g., `apple_0000.jpg`) has a corresponding label file with the same name (`apple_0000.txt`) placed in the same directory.  
Each class folder also contains a `classes.txt` file that maps class indices to class names.

## Classes

- Apple  
- Avocado  
- Banana  
- Blackberry  
- Carrot  
- Cherry  
- Grape  
- Kiwi  
- Lemon  
- Onion  
- Orange  
- Papaya  
- Peach  
- Pear  
- Pepper  
- Raspberry  
- Strawberry  
- Tomato  

## Repository Structure

The repository is organized with separate `images/` and `labels/` directories for each class, further split into `train/`, `val/`, and `test/` subsets. Each split contains its own `classes.txt` file.

```text
data/
├── apple/
│   ├── images/
│   │   ├── train/
│   │   │   ├── apple_0000.jpg
│   │   │   ├── apple_0001.jpg
│   │   │   └── ...
│   │   ├── val/
│   │   │   ├── apple_0025.jpg
│   │   │   └── ...
│   │   └── test/
│   │       ├── apple_0020.jpg
│   │       └── ...
│   └── labels/
│       ├── train/
│       │   ├── apple_0000.txt
│       │   ├── apple_0001.txt
│       │   ├── ...
│       │   └── classes.txt
│       ├── val/
│       │   ├── apple_0025.txt
│       │   ├── ...
│       │   └── classes.txt
│       └── test/
│           ├── apple_0020.txt
│           ├── ...
│           └── classes.txt
├── avocado/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
├── banana/
│   └── ...
├── blackberry/
├── carrot/
├── cherry/
├── grape/
├── kiwi/
├── lemon/
├── onion/
├── orange/
├── papaya/
├── peach/
├── pear/
├── pepper/
├── raspberry/
├── strawberry/
└── tomato/
```

## License

This dataset is licensed under **CC BY 4.0**.

You are free to:
- Use
- Modify
- Redistribute
- Use commercially

As long as proper attribution is provided.

## Citation

If this dataset contributes to your research or project, please cite this repository.
