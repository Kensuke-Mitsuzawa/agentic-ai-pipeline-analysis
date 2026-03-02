import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def render_cka_heatmap(cka_matrix: np.ndarray, labels: list[str], output_path: str = "cka_heatmap.png", graph_dependency: dict | None = None, original_node_names: list[str] | None = None):
    """
    Renders a directed graph for the Centered Kernel Alignment based on dependencies.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    G = nx.DiGraph()
    
    # Add nodes
    for i, label in enumerate(labels):
        G.add_node(i, label=label)
        
    # Add edges based on graph_dependency if provided
    if graph_dependency and original_node_names:
        name_to_idx = {name: i for i, name in enumerate(original_node_names)}
        for from_node, to_nodes in graph_dependency.items():
            if from_node in name_to_idx:
                u = name_to_idx[from_node]
                for to_node in to_nodes:
                    if to_node in name_to_idx:
                        v = name_to_idx[to_node]
                        # Assume cka_matrix has indices matching original_node_names
                        weight = cka_matrix[u, v]
                        G.add_edge(u, v, weight=weight)
    else:
        # Fallback to fully connected or adjacent if no dependency provided
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                if cka_matrix[i, j] > 0:
                    G.add_edge(i, j, weight=cka_matrix[i, j])
                    
    pos = nx.spring_layout(G, k=1.5, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='skyblue', ax=ax)
    
    # Draw node labels
    node_labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, ax=ax)
    
    # Draw edges and edge labels
    edges = G.edges()
    weights = [G[u][v]['weight'] * 5 for u, v in edges]  # scale width for visibility
    
    nx.draw_networkx_edges(G, pos, edgelist=edges, arrowstyle='->', arrowsize=20, width=weights, edge_color='gray', ax=ax)
    
    edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9, ax=ax)
    
    ax.set_title("Agentic AI Centered Kernel Alignment (CKA) Graph", pad=20)
    ax.axis("off")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
