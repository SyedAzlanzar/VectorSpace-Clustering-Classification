import numpy as np
from algorithms import DBSCAN

def run_dbscan_universal_tests():
    test_cases = [
        {
            "name": "2D Simple Clusters (Best Case)",
            "data": np.array([[1,1], [1.1,1], [5,5], [5.1,5]]),
            "eps": 1.0,
            "min_samples": 2
        },
        {
            "name": "3D Random Points (Average Case)",
            "data": np.random.rand(50, 3) * 10, # 3D data
            "eps": 2.0,
            "min_samples": 3
        },
        {
            "name": "5D High Dimensionality (Universal Case)",
            "data": np.random.rand(20, 5), # 5D data
            "eps": 0.5,
            "min_samples": 2
        }
    ]

    for case in test_cases:
        print(f"\nTesting: {case['name']}")
        db = DBSCAN(eps=case['eps'], min_samples=case['min_samples'], distance_metric='euclidean')
        labels = db.fit(case['data'])
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        print(f"Data Shape: {case['data'].shape}")
        print(f"Clusters found: {n_clusters}")
        print(f"Noise points: {n_noise}")
        print(f"Labels: {labels}")