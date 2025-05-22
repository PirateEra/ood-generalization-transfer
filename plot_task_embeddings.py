import os
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

if __name__ == "__main__":
    root_folder = "task_embeddings"

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

    plt.figure(figsize=(10, 8))

    # Plot each datasets embedding vector
    for i, label in enumerate(labels):
        category = datasets.get(label, 'unknown')
        print(category)

        color = colors.get(category, 'gray')
        marker = markers.get(category, 'x')

        plt.scatter(X_tsne[i, 0], X_tsne[i, 1], 
                    color=color, marker=marker, 
                    s=100, 
                    alpha=0.8, 
                    label=category)

        plt.text(X_tsne[i, 0]+0.5, X_tsne[i, 1]+0.5, label, fontsize=8)
    
    plt.xticks([])
    plt.yticks([])

    plt.title("Task Embedding Space")
    plt.tight_layout()

    output_path = "task_embeddings_self.png"
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")