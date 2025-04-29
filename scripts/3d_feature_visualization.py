import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import argparse
from pathlib import Path
import json
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize extracted features in 3D")
    parser.add_argument("--features_dir", type=str, required=True, help="Directory containing extracted features")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save visualizations")
    parser.add_argument("--perplexity", type=int, default=15, help="Perplexity parameter for t-SNE")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility")
    parser.add_argument("--interactive", action="store_true", help="Generate interactive HTML plots")
    return parser.parse_args()

def load_data(features_dir):
    """Load features and labels from the features directory"""
    # Load metadata to check feature shapes
    with open(os.path.join(features_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)
    
    # Load labels
    labels = np.load(os.path.join(features_dir, "labels.npy"))
    
    # Load modality masks if available
    modality_path = os.path.join(features_dir, "modality_masks.npy")
    modality_masks = np.load(modality_path) if os.path.exists(modality_path) else None
    
    # Load features
    features = {}
    for layer_name in metadata["feature_shapes"].keys():
        feature_path = os.path.join(features_dir, f"{layer_name}_features.npy")
        if os.path.exists(feature_path):
            features[layer_name] = np.load(feature_path)
    
    return features, labels, modality_masks, metadata

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

def apply_pca(features, n_components=3, random_state=42):
    """Apply PCA dimensionality reduction"""
    pca = PCA(n_components=n_components, random_state=random_state)
    reduced_features = pca.fit_transform(features)
    explained_variance = pca.explained_variance_ratio_.sum() * 100
    component_variances = pca.explained_variance_ratio_ * 100
    return reduced_features, explained_variance, component_variances

def apply_tsne(features, n_components=3, perplexity=15, random_state=42):
    """Apply t-SNE dimensionality reduction"""
    tsne = TSNE(n_components=n_components, perplexity=perplexity, 
                n_iter=2000, random_state=random_state)
    reduced_features = tsne.fit_transform(features)
    return reduced_features
def create_static_3d_plot(reduced_features, labels, method, layer_name, output_path, 
                           explained_variance=None, component_variances=None, modality_masks=None):
    """Create a static 3D scatter plot of the reduced features without modality distinctions"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Convert label indices to strings for better display
    unique_labels = np.unique(labels)
    label_names = [f"Class {int(label)}" for label in unique_labels]
    
    # Create a color palette with distinct colors
    palette = sns.color_palette("tab10", len(unique_labels))
    
    # Scatter plot for each class
    for i, label in enumerate(unique_labels):
        idx = labels == label
        ax.scatter(
            reduced_features[idx, 0], 
            reduced_features[idx, 1], 
            reduced_features[idx, 2],
            c=[palette[i]], 
            label=label_names[i],
            alpha=0.8,
            edgecolors='w',
            s=100
        )
    
    # Add component variances to axis labels if available
    if component_variances is not None:
        xlabel = f"{method} Component 1 ({component_variances[0]:.1f}%)"
        ylabel = f"{method} Component 2 ({component_variances[1]:.1f}%)"
        zlabel = f"{method} Component 3 ({component_variances[2]:.1f}%)"
    else:
        xlabel = f"{method} Component 1"
        ylabel = f"{method} Component 2"
        zlabel = f"{method} Component 3"
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_zlabel(zlabel, fontsize=12)
    
    # Add title
    title = f"{method} 3D visualization of {layer_name} features"
    if explained_variance is not None:
        title += f"\nExplained variance: {explained_variance:.2f}%"
    
    plt.title(title, fontsize=14)
    
    # Add legend and grid
    plt.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))
    
    # Adjust the plot for better visualization
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved 3D visualization to {output_path}")
    
def _create_static_3d_plot(reduced_features, labels, method, layer_name, output_path, 
                        explained_variance=None, component_variances=None, modality_masks=None):
    """Create a static 3D scatter plot of the reduced features"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Convert label indices to strings for better display
    unique_labels = np.unique(labels)
    label_names = [f"Class {int(label)}" for label in unique_labels]
    
    # Create a color palette with distinct colors
    palette = sns.color_palette("tab10", len(unique_labels))
    
    # If we have modality information, use different markers for different modalities
    markers = ['o', '^', 's', 'D']  # Circle, triangle, square, diamond
    marker_labels = []
    
    # Scatter plot for each class
    for i, label in enumerate(unique_labels):
        idx = labels == label
        
        if modality_masks is not None:
            # For samples with both modalities
            both_idx = idx & (modality_masks[:, 0] & modality_masks[:, 1])
            if np.any(both_idx):
                ax.scatter(
                    reduced_features[both_idx, 0], 
                    reduced_features[both_idx, 1], 
                    reduced_features[both_idx, 2],
                    c=[palette[i]], 
                    marker='o',
                    label=f"{label_names[i]} (Both)",
                    alpha=0.8,
                    edgecolors='w',
                    s=100
                )
                marker_labels.append(f"Both modalities")
            
            # For samples with only CT
            ct_only_idx = idx & (modality_masks[:, 0] & ~modality_masks[:, 1])
            if np.any(ct_only_idx):
                ax.scatter(
                    reduced_features[ct_only_idx, 0], 
                    reduced_features[ct_only_idx, 1], 
                    reduced_features[ct_only_idx, 2],
                    c=[palette[i]], 
                    marker='^',
                    label=f"{label_names[i]} (CT only)",
                    alpha=0.8,
                    edgecolors='w',
                    s=100
                )
                marker_labels.append(f"CT only")
            
            # For samples with only WSI
            wsi_only_idx = idx & (~modality_masks[:, 0] & modality_masks[:, 1])
            if np.any(wsi_only_idx):
                ax.scatter(
                    reduced_features[wsi_only_idx, 0], 
                    reduced_features[wsi_only_idx, 1], 
                    reduced_features[wsi_only_idx, 2],
                    c=[palette[i]], 
                    marker='s',
                    label=f"{label_names[i]} (WSI only)",
                    alpha=0.8,
                    edgecolors='w',
                    s=100
                )
                marker_labels.append(f"WSI only")
        else:
            # Without modality information, just use the class
            ax.scatter(
                reduced_features[idx, 0], 
                reduced_features[idx, 1], 
                reduced_features[idx, 2],
                c=[palette[i]], 
                label=label_names[i],
                alpha=0.8,
                edgecolors='w',
                s=100
            )
    
    # Add component variances to axis labels if available
    if component_variances is not None:
        xlabel = f"{method} Component 1 ({component_variances[0]:.1f}%)"
        ylabel = f"{method} Component 2 ({component_variances[1]:.1f}%)"
        zlabel = f"{method} Component 3 ({component_variances[2]:.1f}%)"
    else:
        xlabel = f"{method} Component 1"
        ylabel = f"{method} Component 2"
        zlabel = f"{method} Component 3"
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_zlabel(zlabel, fontsize=12)
    
    # Add title
    title = f"{method} 3D visualization of {layer_name} features"
    if explained_variance is not None:
        title += f"\nExplained variance: {explained_variance:.2f}%"
    
    plt.title(title, fontsize=14)
    
    # Add legend and grid
    plt.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))
    
    # Adjust the plot for better visualization
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved 3D visualization to {output_path}")

def create_interactive_3d_plot(reduced_features, labels, method, layer_name, output_path, 
                             explained_variance=None, component_variances=None, modality_masks=None):
    """Create an interactive 3D plot using Plotly"""
    # Convert label indices to strings for better display
    label_array = [f"Class {int(label)}" for label in labels]
    
    # Create a DataFrame for plotly
    df = pd.DataFrame({
        'Component 1': reduced_features[:, 0],
        'Component 2': reduced_features[:, 1],
        'Component 3': reduced_features[:, 2],
        'Label': label_array,
    })
    
    # Add modality information if available
    if modality_masks is not None:
        modality_info = []
        for i in range(len(modality_masks)):
            if modality_masks[i, 0] and modality_masks[i, 1]:
                modality_info.append("Both")
            elif modality_masks[i, 0]:
                modality_info.append("CT only")
            elif modality_masks[i, 1]:
                modality_info.append("WSI only")
            else:
                modality_info.append("None")
        df['Modality'] = modality_info
    
    # Update component labels with variance explained
    axis_labels = {}
    if component_variances is not None:
        axis_labels = {
            'Component 1': f"Component 1 ({component_variances[0]:.1f}%)",
            'Component 2': f"Component 2 ({component_variances[1]:.1f}%)",
            'Component 3': f"Component 3 ({component_variances[2]:.1f}%)"
        }
    
    # Create the plot
    title = f"{method} 3D visualization of {layer_name} features"
    if explained_variance is not None:
        title += f" (Explained variance: {explained_variance:.2f}%)"
    
    if modality_masks is not None:
        # Plot with modality information
        fig = px.scatter_3d(
            df, x='Component 1', y='Component 2', z='Component 3',
            color='Label', symbol='Modality', 
            labels=axis_labels,
            title=title
        )
    else:
        # Plot without modality information
        fig = px.scatter_3d(
            df, x='Component 1', y='Component 2', z='Component 3',
            color='Label',
            labels=axis_labels,
            title=title
        )
    
    # Customize the plot
    fig.update_traces(marker=dict(size=6, opacity=0.7, line=dict(width=1, color='white')))
    
    # Update layout
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        scene=dict(
            xaxis_title=axis_labels.get('Component 1', 'Component 1'),
            yaxis_title=axis_labels.get('Component 2', 'Component 2'),
            zaxis_title=axis_labels.get('Component 3', 'Component 3'),
        )
    )
    
    # Save the plot as HTML
    fig.write_html(output_path)
    print(f"Saved interactive 3D visualization to {output_path}")

def visualize_features(features_dict, labels, modality_masks, args):
    """Visualize each feature set using PCA and t-SNE in 3D"""
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
        pca_features, explained_variance, component_variances = apply_pca(
            reshaped_features, 
            n_components=3,
            random_state=args.random_state
        )
        
        # Create static PCA plot
        pca_output_path = os.path.join(args.output_dir, f"{layer_name}_pca_3d.png")
        create_static_3d_plot(
            pca_features, 
            labels, 
            "PCA", 
            layer_name, 
            pca_output_path,
            explained_variance,
            component_variances,
            modality_masks
        )
        
        # Create interactive PCA plot if requested
        if args.interactive:
            interactive_pca_path = os.path.join(args.output_dir, f"{layer_name}_pca_3d_interactive.html")
            create_interactive_3d_plot(
                pca_features, 
                labels, 
                "PCA", 
                layer_name, 
                interactive_pca_path,
                explained_variance,
                component_variances,
                modality_masks
            )
        
        # Apply t-SNE
        tsne_features = apply_tsne(
            reshaped_features,
            n_components=3,
            perplexity=args.perplexity,
            random_state=args.random_state
        )
        
        # Create static t-SNE plot
        tsne_output_path = os.path.join(args.output_dir, f"{layer_name}_tsne_3d.png")
        create_static_3d_plot(
            tsne_features, 
            labels, 
            "t-SNE", 
            layer_name, 
            tsne_output_path,
            modality_masks=modality_masks
        )
        
        # Create interactive t-SNE plot if requested
        if args.interactive:
            interactive_tsne_path = os.path.join(args.output_dir, f"{layer_name}_tsne_3d_interactive.html")
            create_interactive_3d_plot(
                tsne_features, 
                labels, 
                "t-SNE", 
                layer_name, 
                interactive_tsne_path,
                modality_masks=modality_masks
            )

def main():
    # Parse arguments
    args = parse_args()
    
    # Load data
    features_dict, labels, modality_masks, metadata = load_data(args.features_dir)
    print(f"Loaded features with shapes: {metadata['feature_shapes']}")
    print(f"Label distribution: {metadata['label_distribution']}")
    
    # Visualize features
    visualize_features(features_dict, labels, modality_masks, args)
    print("Visualization completed successfully")

if __name__ == "__main__":
    main()