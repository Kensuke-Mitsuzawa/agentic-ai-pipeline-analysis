import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def render_cka_heatmap(cka_matrix: np.ndarray, labels: list[str], output_path: str = "cka_heatmap.png"):
    """
    Renders a seaborn lower-triangular heatmap for the Centered Kernel Alignment.
    """
    # Create mask for the upper triangle
    mask = np.triu(np.ones_like(cka_matrix, dtype=bool))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Draw heatmap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(
        cka_matrix, 
        mask=mask, 
        cmap=cmap, 
        vmax=1.0, 
        vmin=0.0, 
        center=0.5,
        square=True, 
        linewidths=.5, 
        cbar_kws={"shrink": .5}, 
        annot=True, 
        fmt=".2f",
        xticklabels=labels, 
        yticklabels=labels,
        ax=ax
    )
    
    ax.set_title("Agentic AI Centered Kernel Alignment (CKA) Heatmap", pad=20)
    
    # Rotate tick labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
