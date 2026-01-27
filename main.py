import argparse
import numpy as np
from PIL import Image
from random import  randint

# Local imports from your files
from algorithms import KMeans, DBSCAN
from utils.random_image_picker import download_test_image
from utils.visualization import plot_3d_pixels
from algorithms import run_experiment
from sklearn.datasets import load_iris, load_breast_cancer

def main():
    parser = argparse.ArgumentParser(description="Data Mining Assignment: Tasks 1 & 2")
    
    # Universal Arguments
    parser.add_argument("--task", choices=['1', '2'], required=True, help="1 for Clustering, 2 for Classification")
    parser.add_argument("--dist", default='euclidean', choices=['euclidean', 'manhattan', 'maximum'], help="Distance metric")
    parser.add_argument("--k", type=int, default=3, help="k-value for KMeans or KNN")
    
    # Task 1 Specific Arguments
    parser.add_argument("--algo", choices=['kmeans', 'dbscan'], help="Algorithm for Task 1")
    parser.add_argument("--image", help="Path to custom image for Task 1")
    parser.add_argument("--eps", type=float, default=5.0, help="DBSCAN epsilon")
    parser.add_argument("--min_samples", type=int, default=30, help="DBSCAN min_samples")
    parser.add_argument("--test_file", choices=['test_data_task_1/test_kmeans.py', 'test_data_task_1/test_dbscan.py'], help="Test file for Task 1")
    
    
    args = parser.parse_args()

    # Task 1: Clustering
    if args.task == '1':
        if not args.algo:
            print("Error: --algo (kmeans or dbscan) is required for Task 1.")
            return
        
        if args.test_file:
            if args.test_file == 'test_data_task_1/test_kmeans.py' and args.algo == 'kmeans':
                from test_data_task_1.test_kmeans import run_kmeans_universal_tests
                run_kmeans_universal_tests()
                return
            else:
                from test_data_task_1.test_dbscan import run_dbscan_universal_tests
                run_dbscan_universal_tests()
                return
            

        # Image Handling
        if args.image:
            img = Image.open(args.image).convert('RGB')
        else:
            random_image_no = f"{randint(0, 99):02d}"
            download_test_image(random_image_no)
            img = Image.open("output/real_test.jpg").convert('RGB')   
                 
        pixels = np.array(img)
        w, h, d = pixels.shape
        flat_pixels = pixels.reshape(-1, 3).astype(float)

        print(f"Running Task 1: {args.algo} clustering...")
        
        if args.algo == 'kmeans':
            model = KMeans(k=args.k, distance_metric=args.dist)
            labels = model.fit(flat_pixels)
            centers = model.centroids
            new_pixels = centers[labels]
        else:
            model = DBSCAN(eps=args.eps, min_samples=args.min_samples, distance_metric=args.dist)
            labels = model.fit(flat_pixels)
            unique_labels = np.unique(labels)
            
            # Calculate average colors for DBSCAN clusters
            centers = {}
            for l in unique_labels:
                if l == -1:
                    centers[l] = np.array([0, 0, 0]) # Noise is black
                else:
                    centers[l] = flat_pixels[labels == l].mean(axis=0)
            new_pixels = np.array([centers[l] for l in labels])

        # 3D Visualization
        plot_3d_pixels(flat_pixels, labels, centers, title=f"3D Space: {args.algo} (k={args.k})")
            
        # Save Output
        segmented = new_pixels.reshape(w, h, d).astype('uint8')
        Image.fromarray(segmented).save("output/output.png")
        print(f"Task 1 Complete.")
    
    # Task 2: Classification
    elif args.task == '2':
            
        # Iris dataset
        iris = load_iris()
        run_experiment(
            iris.data,
            iris.target,
            "Iris",
            iris.feature_names,
            iris.target_names
        )

        # Breast Cancer dataset
        cancer = load_breast_cancer()
        run_experiment(
            cancer.data,
            cancer.target,
            "Breast Cancer Wisconsin",
            cancer.feature_names,
            cancer.target_names
        )
      

if __name__ == "__main__":
    main()