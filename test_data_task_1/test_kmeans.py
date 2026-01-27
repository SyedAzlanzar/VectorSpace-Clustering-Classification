import numpy as np
from algorithms import KMeans

def run_kmeans_universal_tests():
    test_cases = [
        {
            "name": "2D Simple Clusters (Best Case)",
            "data": np.array([[1,1], [1.2,1], [10,10], [10.2,10]]),
            "k": 2,
            "dist": "euclidean"
        },
        {
            "name": "3D Color Points (Average Case)",
            "data": np.random.randint(0, 255, (50, 3)),
            "k": 3,
            "dist": "manhattan"
        },
        {
            "name": "5D Sensor Data (Universal Case)",
            "data": np.random.rand(30, 5) * 100,
            "k": 4,
            "dist": "maximum"
        }
    ]

    for case in test_cases:
        print(f"\nTesting: {case['name']}")
        model = KMeans(k=case['k'], distance_metric=case['dist'])
        labels = model.fit(case['data'])
        
        print(f"Data Shape: {case['data'].shape} (Dimensions: {case['data'].shape[1]})")
        print(f"Requested k: {case['k']}")
        print(f"Centroids Found: {len(model.centroids)}")
        print(f"Labels assigned to first 5 points: {labels[:5]}")