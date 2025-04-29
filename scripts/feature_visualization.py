import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import argparse
from pathlib import Path
import json
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize extracted features using dimensionality reduction")
    parser.add_argument("--features_dir", type=str, required=True, help="Directory containing extracted features")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save visualizations")
    parser.add_argument("--perplexity", type=int, default=15, help="Perplexity parameter for t-SNE")
    parser.add_argument("--n_components", type=int, default=2, help="Number of components for dimensionality reduction")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility")
    return parser.parse_args()

def load_data(features_dir):
    """Load features and labels from the features directory"""
    # Load metadata to check feature shapes
    with open(os.path.join(features_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)
    
    # Load labels
    labels = np.load(os.path.join(features_dir, "labels.npy"))
    
    # Load features
    features = {}
    for layer_name in metadata["feature_shapes"].keys():
        feature_path = os.path.join(features_dir, f"{layer_name}_features.npy")
        if os.path.exists(feature_path):
            features[layer_name] = np.load(feature_path)
    
    return features, labels, metadata

def reshape_features(features, layer_name):
    """Reshape high-dimensional features to 2D for dimensionality reduction"""
    if layer_name == "classification":
        # Classification features might already be low-dimensional
        return features
    
    # For multi-dimensional features, flatten all dimensions except the first (sample dimension)
    original_shape = features.shape
    flattened_features = features.reshape(original_shape[0], -1)
    
    print(f"Reshaped {layer_name} features from {original_shape} to {flattened_features.shape}")
    return flattened_features

def apply_pca(features, n_components=2, random_state=42):
    """Apply PCA dimensionality reduction"""
    pca = PCA(n_components=n_components, random_state=random_state)
    reduced_features = pca.fit_transform(features)
    explained_variance = pca.explained_variance_ratio_.sum() * 100
    return reduced_features, explained_variance

def apply_tsne(features, n_components=2, perplexity=15, random_state=42):
    """Apply t-SNE dimensionality reduction"""
    tsne = TSNE(n_components=n_components, perplexity=perplexity, 
                n_iter=2000, random_state=random_state)
    reduced_features = tsne.fit_transform(features)
    return reduced_features

def plot_reduced_features(reduced_features, labels, method, layer_name, output_path, explained_variance=None):
    """Create a scatter plot of the reduced features, colored by label"""
    plt.figure(figsize=(10, 8))
    
    # Convert label indices to strings for better display
    unique_labels = np.unique(labels)
    label_names = [f"Class {int(label)}" for label in unique_labels]
    
    # Create a color palette with distinct colors
    palette = sns.color_palette("tab10", len(unique_labels))
    
    # Scatter plot for each class
    for i, label in enumerate(unique_labels):
        idx = labels == label
        plt.scatter(
            reduced_features[idx, 0], 
            reduced_features[idx, 1], 
            c=[palette[i]], 
            label=label_names[i],
            alpha=0.7,
            edgecolors='w',
            s=100
        )
    
    # Add title and labels
    title = f"{method} visualization of {layer_name} features"
    if explained_variance is not None:
        title += f" (Explained variance: {explained_variance:.2f}%)"
    
    plt.title(title, fontsize=14)
    plt.xlabel(f"{method} Component 1", fontsize=12)
    plt.ylabel(f"{method} Component 2", fontsize=12)
    
    # Add legend and grid
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {output_path}")

def visualize_features(features_dict, labels, args):
    """Visualize each feature set using PCA and t-SNE"""
    os.makedirs(args.output_dir, exist_ok=True)
    
    target_layers = ["fused_features", "before_classification", "f_rad_f_histo", "positional_embeddings"]
    for layer_name in target_layers:
        if layer_name not in features_dict:
            print(f"Warning: {layer_name} not found in features dictionary")
            continue
        
        # Reshape features for dimensionality reduction
        reshaped_features = reshape_features(features_dict[layer_name], layer_name)
        
        # Skip if features are too sparse or invalid
        if np.isnan(reshaped_features).any() or np.isinf(reshaped_features).any():
            print(f"Warning: {layer_name} features contain NaN or Inf values, skipping")
            continue
        
        print(f"Applying dimensionality reduction to {layer_name} features...")
        
        # Apply PCA
        pca_features, explained_variance = apply_pca(
            reshaped_features, 
            n_components=args.n_components,
            random_state=args.random_state
        )
        
        pca_output_path = os.path.join(args.output_dir, f"{layer_name}_pca.png")
        plot_reduced_features(
            pca_features, 
            labels, 
            "PCA", 
            layer_name, 
            pca_output_path,
            explained_variance
        )
        
        # Apply t-SNE (this can be slow for large feature dimensions)
        tsne_features = apply_tsne(
            reshaped_features,
            n_components=args.n_components,
            perplexity=args.perplexity,
            random_state=args.random_state
        )
        
        tsne_output_path = os.path.join(args.output_dir, f"{layer_name}_tsne.png")
        plot_reduced_features(
            tsne_features, 
            labels, 
            "t-SNE", 
            layer_name, 
            tsne_output_path
        )

def main():
    # Parse arguments
    args = parse_args()
    
    # Load data
    features_dict, labels, metadata = load_data(args.features_dir)
    print(f"Loaded features with shapes: {metadata['feature_shapes']}")
    print(f"Label distribution: {metadata['label_distribution']}")
    
    # Visualize features
    visualize_features(features_dict, labels, args)
    print("Visualization completed successfully")

if __name__ == "__main__":
    main()