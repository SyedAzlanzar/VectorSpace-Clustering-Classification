import matplotlib.pyplot as plt
import numpy as np

def plot_3d_pixels(pixels, labels, centers=None, title="3D Pixel Space (RGB)"):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Subsample for visualization if too many points
    if len(pixels) > 2000:
        idx = np.random.choice(len(pixels), 2000, replace=False)
        p_sample = pixels[idx]
        l_sample = labels[idx]
    else:
        p_sample, l_sample = pixels, labels

    # 3D Scatter Plot
    ax.scatter(p_sample[:, 0], p_sample[:, 1], p_sample[:, 2], c=l_sample, cmap='viridis', s=5, alpha=0.6)
    
    # Plot Centroids if they exist (for KMeans)
    if centers is not None:
        if isinstance(centers, dict): # For DBSCAN dict
            centers = np.array([v for k, v in centers.items() if k != -1])
        ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], 
                  c='red', marker='X', s=200, label='Centroids', edgecolors='black')

    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    ax.set_title(title)
    plt.savefig("output/3d_pixel_space.png")
    plt.close()