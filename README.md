# Clustering and Classification Project

This project features custom implementations of **KMeans**, **DBSCAN**, and **KNN** algorithms. It supports image segmentation, multi-dimensional synthetic data analysis, and scientific classification benchmarks.

## 📂 Project Structure

- **`main.py`**: Central CLI entry point for all tasks  
- **`algorithms.py`**: Mathematical logic for KMeans, DBSCAN, and KNN  
- **`image.py`**: Handles image downloading and local file pathing  
- **`output/`**: Directory for all results (segmented images, original copies, and 3D plots)  
- **`utils/`**: Helper scripts for 3D RGB visualization  
- **`test_data_task_1/`**: Contains algorithm verification scripts  

## Task 1: Image Segmentation (Clustering)

Task 1 treats image pixels as **3D vectors (Red, Green, Blue)** to perform color-based segmentation.

### Running with a Specific Local Image
To use an image already existing in your folder or absolute path (e.g., `my_image.jpg` or `C:\Users\XYZ\Desktop\test-folder\test_image.jpg`), use the `--image` flag.

**Note:** The script will automatically look for this file, process it, and save the result into the `output/` folder.

```bash
# KMeans with a local image
python main.py --task 1 --algo kmeans --k 5 --dist euclidean --image my_image.jpg

# DBSCAN with a local image
python main.py --task 1 --algo dbscan --eps 12.0 --min_samples 20 --image my_image.jpg
```

### Running with a Random Image
If you do not have a local image, the script will download a random sample from a cloud repository automatically.

```bash
python main.py --task 1 --algo kmeans --k 4 --dist euclidean
```

### Running Universal Dimensionality Tests

Tests the algorithms on hardcoded **2D, 3D, and 5D datasets** to demonstrate mathematical universality.

```bash
python main.py --task 1 --algo kmeans --test_file test_data_task_1/test_kmeans.py
```

# Classification Analysis - Task 2

This task implements and evaluates Supervised Learning models to classify structured scientific data. It compares a scratch-built KNN implementation against industry-standard library models.

## How to Run

Task 2 runs a comparative benchmark across two datasets: the **Iris** dataset (simple/multi-class) and the **Breast Cancer** dataset (complex/binary).

### Run Full Benchmark Suite
Execute the following command to train and test all classifiers:
```bash
python main.py --task 2 