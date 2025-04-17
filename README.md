# Anatomical Landmark Detector (Testing)

This project provides a deep learning-based testing pipeline using U-Net to evaluate anatomical landmark predictions on medical images (e.g., AP hip X-rays).  

- For **annotating** landmarks: [Anatomical-Landmark-Annotator](https://github.com/yehyunsuh/Anatomical-Landmark-Annotator)
- For **training** the model: [Anatomical-Landmark-Detector-Training](https://github.com/yehyunsuh/Anatomical-Landmark-Detector-Training)

---

## 📂 Directory Structure
```
Anatomical-Landmark-Detector-Testing/
│
├── data/
│   ├── test_images/               # Directory with test input images
│   └── labels/
│       └── test_annotation.csv   # CSV with landmark coordinates (optional, used if –-labels y)
│
├── test_results/                 # CSV outputs of predicted coordinates
├── visualization_w_label/        # Visualized images with predicted & ground-truth landmarks (if labels exist)
├── visualization_wo_label/       # Visualized images with only predictions (if no labels)
├── weight/
│   └── best_model.pth            # Trained model weights
│
├── main.py                       # Main entry point to run testing
├── model.py                      # U-Net model definition
├── data_loader.py                # Dataset and DataLoader for test data
├── test.py                       # Core testing logic and prediction pipeline
├── utils.py                      # Utility functions (e.g., seeding)
├── visualization.py              # Functions for drawing predicted/ground-truth landmarks
├── requirements.txt              # Python dependencies
└── README.md                     # You’re here!
```

---

## 🚀 Getting Started

### 1. Install Dependencies

We recommend using a virtual environment:

```bash
git clone https://github.com/yehyunsuh/Anatomical-Landmark-Detector-Testing.git
cd Anatomical-Landmark-Detector-Testing
conda create -n detector python=3.10 -y
conda activate detector
pip3 install -r requirements.txt
```

### 2. Prepare Your Data

Place your training images under:
```
data/test_images/
```

If you have ground-truth landmarks, place your annotation CSV file under:
```
data/labels/test_annotation.csv
```

Format of the CSV:
```
image_name,image_width,image_height,n_landmarks,landmark_1_x,landmark_1_y,...
image1.jpg,1098,1120,3,123,145,...
image2.jpg,1400,1210,3,108,132,...
```

Make sure the number of (x, y) coordinate pairs matches `--n_landmarks`.

### 3. Run Training
With labels (evaluation enabled):
```bash
python main.py --n_landmarks 2 --labels y
```

Without labels (prediction only):
```bash
python main.py --n_landmarks 2 --labels n
```

You can also customize:
```bash
python main.py \
    --test_image_dir ./data/test_images \
    --label_dir ./data/labels \
    --test_csv_file test_annotation.csv \
    --image_resize 512 \
    --n_landmarks 2 \
    --labels y \
    --weight_dir ./weight \
    --weight_name best_model.pth
```

### 🧩 Argument Reference

| Argument           | Description                                                     | Default                     |
|--------------------|-----------------------------------------------------------------|-----------------------------|
| `--test_image_dir` | Path to the test image directory                                | `./data/test_images`        |
| `--label_dir`      | Path to the directory containing annotation CSV (optional)      | `./data/labels`             |
| `--test_csv_file`  | CSV filename containing image metadata and landmarks            | `test_annotation.csv`       |
| `--image_resize`   | Resize images to this square dimension (must be /32)            | `512`                       |
| `--n_landmarks`    | Number of landmarks per image *(REQUIRED)*                      |                             |
| `--labels`         | `y` if ground-truth labels exist, `n` otherwise *(REQUIRED)*             |                             |
| `--weight_dir`     | Path to the directory containing the model weight               | `./weight`                  |
| `--weight_name`    | Filename of the model weight to load                            | `best_model.pth`            |
| `--seed`           | Random seed for reproducibility                                 | `42`                        |

## 📊 Visualization
- Blue circles: Ground truth landmarks
- Red circles: Predicted landmarks
- Visualization example 

<img src="https://github.com/user-attachments/assets/47d56ed5-637b-431a-bec5-9260d9762539" width="250" height="250">
<img src="https://github.com/user-attachments/assets/55b3c460-906d-4156-8ffe-b299c3112df0" width="250" height="250">
<img src="https://github.com/user-attachments/assets/bcadf422-0fc6-4575-b26e-dfbf0e89ba9d" width="250" height="250">
<img src="https://github.com/user-attachments/assets/a7095213-b050-4983-a929-529cddc9f507" width="250" height="250">
<img src="https://github.com/user-attachments/assets/d07d11b1-eef5-44d9-92c8-ace64199f723" width="250" height="250">
<img src="https://github.com/user-attachments/assets/e66f5dd5-d98a-4a29-b5d2-ef1235b7c983" width="250" height="250">   

## Citation
If you find this helpful, please cite this [paper](https://openreview.net/forum?id=bVC9bi_-t7Y):
```
@inproceedings{
suh2023dilationerosion,
title={Dilation-Erosion Methods for Radiograph Annotation in Total Knee Replacement},
author={Yehyun Suh and Aleksander Mika and J. Ryan Martin and Daniel Moyer},
booktitle={Medical Imaging with Deep Learning, short paper track},
year={2023},
url={https://openreview.net/forum?id=bVC9bi_-t7Y}
}
```
Also, if you conduct research in Total Knee Arthroplasty or Total Hip Arthroplasty, check these papers:
- Mika, A. P., Suh, Y., Elrod, R. W., Faschingbauer, M., Moyer, D. C., & Martin, J. R. (2024). Novel Dilation-erosion Labeling Technique Allows for Rapid, Accurate and Adjustable Alignment Measurements in Primary TKA. Computers in Biology and Medicine, 185, 109571. https://doi.org/10.1016/j.compbiomed.2024.109571.  
- Mohsin, M., Suh, Y., Chandrashekar, A., Martin, J., & Moyer, D. (2025). Landmark prediction in large radiographs using RoI-based label augmentation. Proc. SPIE 13410, Medical Imaging 2025: Clinical and Biomedical Imaging, 134100E, 14. https://doi.org/10.1117/12.3047290
- Chan, P. Y., Baker, C. E., Suh, Y., Moyer, D., & Martin, J. R. (2025). Development of a deep learning model for automating implant position in total hip arthroplasty. The Journal of Arthroplasty. https://doi.org/10.1016/j.arth.2025.01.032
