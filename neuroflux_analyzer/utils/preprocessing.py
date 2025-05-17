import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from PIL import Image
import cv2
from skimage import exposure, restoration

class MRIPreprocessor:
    """
    A comprehensive MRI preprocessing class with various methods for 
    normalization, artifact detection, and quality improvement.
    """
    
    def __init__(self, verbose=True):
        """Initialize the preprocessor"""
        self.verbose = verbose
        self.outliers = []
        self.stats = {}
        self.output_path = 'dataset_analysis'
    
    def analyze_dataset(self, image_paths):
        """Analyze the dataset and compute statistics"""
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
            
        intensities = []
        contrasts = []
        snrs = []
        entropies = []
        
        for i, path in enumerate(image_paths):
            try:
                img = self.load_image(path)
                img_array = np.array(img)
                
                # Collect statistics
                intensities.append(np.mean(img_array))
                contrasts.append(np.std(img_array))
                snrs.append(self._calculate_snr(img_array))
                entropies.append(self._calculate_entropy(img_array))
                
                if i % 100 == 0 and self.verbose:
                    print(f"Processed {i}/{len(image_paths)} images")
                    
            except Exception as e:
                print(f"Error processing {path}: {e}")
        
        # Store statistics
        self.stats = {
            'intensity': {
                'mean': np.mean(intensities),
                'std': np.std(intensities),
                'min': np.min(intensities),
                'max': np.max(intensities),
                'values': intensities
            },
            'contrast': {
                'mean': np.mean(contrasts),
                'std': np.std(contrasts),
                'min': np.min(contrasts),
                'max': np.max(contrasts),
                'values': contrasts
            },
            'snr': {
                'mean': np.mean(snrs),
                'std': np.std(snrs),
                'min': np.min(snrs),
                'max': np.max(snrs),
                'values': snrs
            },
            'entropy': {
                'mean': np.mean(entropies),
                'std': np.std(entropies),
                'min': np.min(entropies),
                'max': np.max(entropies),
                'values': entropies
            }
        }
        
        # Plot histograms
        self._plot_histograms(self.output_path)
        
        return self.stats
    
    def detect_outliers(self, image_paths, labels, threshold=2.0, save_dir='outliers'):
        """
        Detect outlier images using statistical methods
        Returns: list of indices of potential outliers
        """
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
            
        if not self.stats:
            print("Running analysis first...")
            self.analyze_dataset(image_paths)
            
        # Extract features for outlier detection
        features = np.column_stack((
            self.stats['intensity']['values'],
            self.stats['contrast']['values'],
            self.stats['snr']['values'],
            self.stats['entropy']['values']
        ))
        
        # Normalize features
        features_mean = np.mean(features, axis=0)
        features_std = np.std(features, axis=0)
        features_norm = (features - features_mean) / (features_std + 1e-10)
        
        # Calculate z-scores
        z_scores = np.abs(features_norm)
        max_z_scores = np.max(z_scores, axis=1)
        
        # Find outliers
        outlier_indices = np.where(max_z_scores > threshold)[0]
        self.outliers = outlier_indices.tolist()
        
        # Plot some outliers
        if len(outlier_indices) > 0:
            fig, axes = plt.subplots(min(10, len(outlier_indices)), 2, figsize=(12, 2*min(10, len(outlier_indices))))
            if len(outlier_indices) == 1:
                axes = np.array([axes])  # Handle single-row case
                
            for i, idx in enumerate(outlier_indices[:10]):
                try:
                    img = self.load_image(image_paths[idx])
                    class_name = labels[idx] if labels else "Unknown"
                    
                    axes[i, 0].imshow(img, cmap='gray')
                    axes[i, 0].set_title(f"Outlier {idx}: Class {class_name}")
                    axes[i, 0].axis('off')
                    
                    # Plot z-scores for this image
                    axes[i, 1].bar(['Intensity', 'Contrast', 'SNR', 'Entropy'], z_scores[idx], color='red')
                    axes[i, 1].axhline(y=threshold, color='black', linestyle='--')
                    axes[i, 1].set_title(f"Z-scores (Max: {max_z_scores[idx]:.2f})")
                except Exception as e:
                    print(f"Error plotting outlier {idx}: {e}")
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_path, 'outliers.png'))
            plt.close()
            
            # Save outlier statistics
            outlier_df = pd.DataFrame({
                'path': [image_paths[i] for i in outlier_indices],
                'index': outlier_indices,
                'intensity': [self.stats['intensity']['values'][i] for i in outlier_indices],
                'contrast': [self.stats['contrast']['values'][i] for i in outlier_indices],
                'snr': [self.stats['snr']['values'][i] for i in outlier_indices],
                'entropy': [self.stats['entropy']['values'][i] for i in outlier_indices],
                'max_z_score': max_z_scores[outlier_indices]
            })
            outlier_df.to_csv(os.path.join(self.output_path, 'outliers.csv'), index=False)
        
        print(f"Detected {len(outlier_indices)} outliers out of {len(image_paths)} images")
        return outlier_indices
    
    def visualize_dataset_tsne(self, image_paths, labels, perplexity=30, save_dir='tsne_results'):
        """
        Visualize the dataset using t-SNE to identify clusters and potential problems
        """
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
            
        if not self.stats:
            print("Running analysis first...")
            self.analyze_dataset(image_paths)
            
        # Extract features
        features = np.column_stack((
            self.stats['intensity']['values'],
            self.stats['contrast']['values'],
            self.stats['snr']['values'],
            self.stats['entropy']['values']
        ))
        
        # Run t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        tsne_result = tsne.fit_transform(features)
        
        # Plot t-SNE result
        plt.figure(figsize=(12, 10))
        
        unique_labels = np.unique(labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            indices = [j for j, l in enumerate(labels) if l == label]
            plt.scatter(tsne_result[indices, 0], tsne_result[indices, 1], 
                        c=[colors[i]], label=label, alpha=0.7)
            
        # Mark outliers if detected
        if self.outliers:
            plt.scatter(tsne_result[self.outliers, 0], tsne_result[self.outliers, 1],
                        marker='x', c='black', s=100, label='Outliers')
            
        plt.legend()
        plt.title('t-SNE visualization of MRI dataset')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, 'tsne_visualization.png'))
        plt.close()
            
        # Run DBSCAN to find clusters
        dbscan = DBSCAN(eps=3.0, min_samples=5)
        clusters = dbscan.fit_predict(tsne_result)
        
        # Visualize clusters
        plt.figure(figsize=(12, 10))
        unique_clusters = np.unique(clusters)
        
        for cluster in unique_clusters:
            if cluster == -1:  # Noise points
                plt.scatter(tsne_result[clusters == cluster, 0], 
                           tsne_result[clusters == cluster, 1],
                           c='black', marker='x', label='Noise')
            else:
                plt.scatter(tsne_result[clusters == cluster, 0], 
                           tsne_result[clusters == cluster, 1],
                           label=f'Cluster {cluster}')
                
        plt.legend()
        plt.title('DBSCAN clustering of MRI dataset')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, 'dbscan_clustering.png'))
        plt.close()
        
        return tsne_result, clusters
    
    def load_image(self, path):
        return Image.open(path).convert('L')  # Convert to grayscale
    
    def preprocess_image(self, image, methods=None):
        """
        Apply preprocessing methods to an image
        methods: list of preprocessing methods to apply, e.g. ['n4', 'clahe', 'denoise']
        """
        if methods is None:
            methods = ['normalize']
            
        img_array = np.array(image)
        
        for method in methods:
            if method == 'normalize':
                img_array = self._normalize_intensity(img_array)
            elif method == 'clahe':
                img_array = self._apply_clahe(img_array)
            elif method == 'denoise':
                img_array = self._apply_denoising(img_array)
            elif method == 'equalize':
                img_array = self._equalize_histogram(img_array)
                
        return Image.fromarray(img_array)
    
    def _normalize_intensity(self, img_array):
        """Normalize image intensity to [0,1]"""
        min_val = np.min(img_array)
        max_val = np.max(img_array)
        
        # Avoid division by zero
        if max_val > min_val:
            img_array = (img_array - min_val) / (max_val - min_val)
        else:
            img_array = np.zeros_like(img_array)
            
        return (img_array * 255).astype(np.uint8)
    
    def _apply_clahe(self, img_array):
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        if img_array.dtype != np.uint8:
            img_array = (img_array * 255).astype(np.uint8)
        return clahe.apply(img_array)
    
    def _apply_denoising(self, img_array):
        """Apply denoising filter"""
        return restoration.denoise_nl_means(
            img_array.astype(np.float32), 
            h=0.8, 
            fast_mode=True,
            patch_size=5,
            patch_distance=7
        ) * 255
    
    def _equalize_histogram(self, img_array):
        """Apply histogram equalization"""
        return exposure.equalize_hist(img_array) * 255
    
    def _calculate_snr(self, img_array):
        """Calculate signal-to-noise ratio"""
        if np.std(img_array) == 0:
            return 0
        return np.mean(img_array) / np.std(img_array)
    
    def _calculate_entropy(self, img_array):
        """Calculate entropy of the image"""
        # Calculate histogram
        hist = np.histogram(img_array, bins=256, range=(0, 255))[0]
        hist = hist / hist.sum()
        
        # Calculate entropy
        entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
        return entropy
    
    def _plot_histograms(self, save_dir):
        """Plot histograms of image statistics"""
        _, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Intensity histogram
        axes[0, 0].hist(self.stats['intensity']['values'], bins=50, color='blue', alpha=0.7)
        axes[0, 0].axvline(self.stats['intensity']['mean'], color='red', linestyle='--')
        axes[0, 0].set_title(f"Intensity Distribution (Mean: {self.stats['intensity']['mean']:.2f})")
        
        # Contrast histogram
        axes[0, 1].hist(self.stats['contrast']['values'], bins=50, color='green', alpha=0.7)
        axes[0, 1].axvline(self.stats['contrast']['mean'], color='red', linestyle='--')
        axes[0, 1].set_title(f"Contrast Distribution (Mean: {self.stats['contrast']['mean']:.2f})")
        
        # SNR histogram
        axes[1, 0].hist(self.stats['snr']['values'], bins=50, color='purple', alpha=0.7)
        axes[1, 0].axvline(self.stats['snr']['mean'], color='red', linestyle='--')
        axes[1, 0].set_title(f"SNR Distribution (Mean: {self.stats['snr']['mean']:.2f})")
        
        # Entropy histogram
        axes[1, 1].hist(self.stats['entropy']['values'], bins=50, color='orange', alpha=0.7)
        axes[1, 1].axvline(self.stats['entropy']['mean'], color='red', linestyle='--')
        axes[1, 1].set_title(f"Entropy Distribution (Mean: {self.stats['entropy']['mean']:.2f})")
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, 'dataset_statistics.png'))
        plt.close()