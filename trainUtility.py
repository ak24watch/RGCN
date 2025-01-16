from LoadDataset import loadDataset

import torch
import dgl

def trainInfo(dataset_name, loop=False):
    """
    Load dataset and prepare graph for training.

    Args:
        loop (bool): Whether to add self-loops to the graph.

    Returns:
        tuple: Prepared graph, category ID, number of classes, number of relations, labels, training indices, testing indices, target indices.
    """

    dataset = loadDataset(dataset_name)
    g = dataset[0]
    category = dataset.predict_category
    num_classes = dataset.num_classes
    num_rels = len(g.canonical_etypes)
    category_id = g.ntypes.index(category)

    train_mask = g.nodes[category].data["train_mask"]
    test_mask = g.nodes[category].data["test_mask"]
    labels = g.nodes[category].data["label"]

    train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze()
    test_idx = torch.nonzero(test_mask, as_tuple=False).squeeze()
    
    for cetype in g.canonical_etypes:
        g.edges[cetype].data["norm"] = dgl.norm_by_dst(g, cetype).unsqueeze(1)
    
    g = dgl.to_homogeneous(g, edata=["norm"])
    node_ids = torch.arange(g.num_nodes())
    target_idx = node_ids[g.ndata[dgl.NTYPE] == category_id]
    return g, category_id, num_classes, num_rels, labels, train_idx, test_idx, target_idx