# Fruit detection inventories

boxed_images.csv inventories the GitHub source photos. publisher_boxes.csv preserves
its original pixel boxes and condition names. openimages_images.csv records local
photos and attribution. openimages_boxes.csv stores normalized box endpoints and
source flags. No missing-fruit boxes have been fabricated.

All paths are relative to fruit13/. These CSV files are staging tables, not YOLO
training labels. Preserve every target box in selected images. Confirm rotation,
coordinate conventions, group policy, box validity and source rights before export.
The sources do not provide comprehensive freshness labels for these natural scenes.
