# Pill Anomaly Detection Using YOLOv8n

An automated computer-vision project for detecting and localizing tablets, capsules, and visible pill-related anomalies in pharmaceutical quality-control images. The project uses the Ultralytics **YOLOv8n** object-detection model, fine-tuned on an annotated YOLO-format dataset, and evaluates the trained model before running inference on unseen test images.

## 1. Project Overview

Manual visual inspection of pharmaceutical products can be time-consuming when large batches of medicines must be checked. This project explores an object-detection pipeline that can identify pill-related objects and highlight their locations in images. The model predicts bounding boxes and class labels, allowing a quality-control operator to review the results.

**Main objectives**
- Prepare and load an annotated pill-image dataset in YOLO format.
- Initialize a pretrained YOLOv8n model and train it for the detection task.
- Evaluate the best saved model using Precision, Recall, mAP@0.50, and mAP@0.50:0.95.
- Run predictions on the unseen test-image collection.
- Visualize training and validation box/classification losses.

The project is an academic laboratory experiment; its results are not a certification for clinical use or manufacturing release decisions.

## 2. Dataset

The notebook uses the `Labeled_full_pill.v2i.yolov8.zip` dataset archive. It extracts the archive into `/content/Pill_anomaly_Detection_dataset` and reads the dataset configuration from `data.yaml`.

The dataset follows the YOLO object-detection structure, with images and corresponding label files divided into training, validation, and test splits. The test directory contains **339 images**.

The UML reference document identifies four target categories:
- `capsule`
- `damaged-pill`
- `foreign-matter`
- `tablet`

The notebook obtains the class configuration from `data.yaml`; check that file if you need to confirm the exact class names and class-ID mapping for your dataset copy.

## 3. Model and Training Configuration

| Parameter | Configuration |
|---|---|
| Detection model | YOLOv8n (pretrained) |
| Framework | Ultralytics |
| Training epochs | 50 |
| Input image size | 640 × 640 |
| Batch size | 16 |
| Compute device | CUDA GPU (`device=0`) |
| Random seed | 42 |
| Early-stopping patience | 15 |
| Data-loader workers | 2 |
| Checkpoint saving | Enabled; every 10 epochs |
| Validation | Enabled |

The notebook uses Ultralytics' automatic optimizer selection; an optimizer name is not explicitly specified in the training call.

## 4. Workflow

1. **Environment setup:** Install the `ultralytics` package and import the required libraries.
2. **Dataset preparation:** Locate the uploaded ZIP archive, extract it, and set the path to `data.yaml`.
3. **Model initialization:** Load the pretrained `yolov8n.pt` weights.
4. **Training:** Train the detector using the configuration shown above.
5. **Checkpoint loading:** Load the best checkpoint from `runs/Pill_anomaly_Detection/weights/best.pt`.
6. **Validation:** Evaluate the checkpoint and report Precision, Recall, mAP@0.50, and mAP@0.50:0.95.
7. **Test inference:** Predict on the 339 test images using confidence threshold `0.25` and image size `640`.
8. **Loss visualization:** Read `results.csv` and plot training/validation box loss and classification loss.

## 5. Validation Results

The supplied project report records the following validation metrics for the best checkpoint:

| Metric | Result |
|---|---:|
| Precision | **95.09%** |
| Recall | **66.35%** |
| mAP@0.50 | **83.48%** |
| mAP@0.50:0.95 | **58.99%** |

The reported model summary contains **3,006,428 parameters** and approximately **8.1 GFLOPs**. Precision is comparatively high, while the lower recall indicates that some ground-truth objects were not detected. Performance also decreases under the stricter IoU thresholds included in mAP@0.50:0.95.

The validation scan reported 678 images, including 79 background images. It also reported 30 ignored image/label pairs due to corruption or mixed segmentation/detection label rows. These annotation issues should be reviewed and corrected before drawing stronger conclusions from the experiment.

## 6. Outputs

The notebook is configured to save:
- Best trained weights: `runs/Pill_anomaly_Detection/weights/best.pt`
- Training results: `runs/Pill_anomaly_Detection/results.csv`
- Test predictions: `runs/test_predictions/`
- Training and validation loss plot: generated from `results.csv`

## 7. Running the Notebook

1. Open `Pill_anomaly_Detection.ipynb` in Google Colab or a compatible Jupyter environment.
2. Upload the YOLO dataset ZIP archive to the runtime's `/content/` directory (for the supplied Colab paths).
3. Run the notebook cells in order.
4. Confirm that the extracted dataset contains `data.yaml`, the expected image splits, and matching YOLO label files.
5. Review the validation metrics, loss curves, saved checkpoint, and test prediction images.

**Requirements:** Python, Jupyter/Google Colab, `ultralytics`, PyTorch, pandas, and matplotlib. A CUDA-enabled GPU is recommended; the supplied experiment was documented on an NVIDIA Tesla T4 environment.

---

**Project:** Pill Anomaly Detection Using YOLOv8n  
**Course:** Deep Learning Lab — Semester 7, Assignment 9  
**Date in project report:** 30 September 2026
