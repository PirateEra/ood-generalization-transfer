import os
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

if __name__ == "__main__":
    root_folder = "text_embeddings"

    #---
    # Find all .pt files in the main directory
    #---
    pt_files = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.pt'):
                full_path = os.path.join(root, file)
                pt_files.append(full_path)

    pt_files.sort()

    embeddings = []
    labels = []

    print(f"Found {len(pt_files)} .pt files.")

    # Create a np array of shape pt-files, embedding vector size
    for embedding in pt_files:
        vec = torch.load(embedding)
        vec = vec.view(-1).cpu().numpy()
        embeddings.append(vec)
        labels.append(os.path.basename(embedding))

    X = np.stack(embeddings)
    print(f"X has shape: {X.shape}")

    #---
    # Perform PCA to reduce dimensionality
    #---
    print("Reducing dimensions with PCA...")
    pca = PCA(n_components=5)
    X_pca = pca.fit_transform(X)

    #---
    # Perform t-SNE to reduce to 2d
    #---
    print("Applying t-SNE...")
    tsne = TSNE(n_components=2, perplexity=5, n_iter=1000, random_state=42)
    X_tsne = tsne.fit_transform(X_pca)

    #---
    # (save) Plot the final embeddings
    #---
    datasets = {
        'XED': 'multi-label',
        'WASSA22': 'multi-class',
        'UsVsThem': 'multi-label',
        'TalesEmotions': 'multi-class',
        'SentimentalLIAR': 'multi-variate',
        'Semeval2018': 'multi-variate',
        'GoodNewsEveryone': 'multi-class',
        'EmotionStimulus': 'multi-class',
        'EmoBank': 'multi-variate',
        'CancerEmo': 'multi-label',
    }

    categories = ['multi-label', 'multi-class', 'multi-variate']
    colors = {
            'multi-label': 'green',
            'multi-class': 'blue',
            'multi-variate': 'red',

        }
    markers = {
        'multi-label': 'o',
        'multi-class': 's',
        'multi-variate': '^',
    }

    plt.figure(figsize=(12, 10))

    plotted_categories = set() # For the legend

    # Plot each dataset embedding vector
    for i, label in enumerate(labels):
        dataset_name = os.path.splitext(label)[0]
        category = datasets.get(dataset_name, 'unknown')

        color = colors.get(category, 'gray')
        marker = markers.get(category, 'x')

        # keep track of the legend info
        show_label = category not in plotted_categories
        plotted_categories.add(category)

        plt.scatter(X_tsne[i, 0], X_tsne[i, 1], 
                    color=color, marker=marker, 
                    s=100, 
                    alpha=0.9, 
                    edgecolors='black',
                    linewidths=1.2,
                    label=category if show_label else None)

        plt.text(X_tsne[i, 0]+1, X_tsne[i, 1]+1, dataset_name,
                fontsize=9, weight='bold', color='black')

    # Expand the plot a bit to make it look nicer
    x_min, x_max = X_tsne[:, 0].min(), X_tsne[:, 0].max()
    y_min, y_max = X_tsne[:, 1].min(), X_tsne[:, 1].max()
    plt.xlim(x_min - 5, x_max + 5)
    plt.ylim(y_min - 5, y_max + 5)

    plt.xticks([])
    plt.yticks([])

    # The plot title with a nice background
    plt.title("Text Embedding Space", fontsize=20, weight='bold', color='black',
            pad=20, loc='center', backgroundcolor='#f0f0f0')

    # The legend to show the different type of text
    legend = plt.legend(title='Text Type', fontsize=10, title_fontsize=12,
                        loc='upper right', frameon=True, edgecolor='black')
    legend.get_frame().set_facecolor('#f8f8f8')

    plt.tight_layout()

    output_path = "text_embeddings_self.png"
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")