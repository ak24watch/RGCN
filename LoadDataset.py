from dgl.data.rdf import AIFBDataset, AMDataset, BGSDataset, MUTAGDataset


def loadDataset(name):
    """
    Load and preprocess dataset based on the provided name.

    Parameters:
    name (str): The name of the dataset to load. Options are 'aifb', 'mutag', 'bgs', 'am'.

    Returns:
    data: The loaded dataset object.

    Raises:
    ValueError: If the dataset name is unknown.
    """
    if name == "aifb":
        data = AIFBDataset()
    elif name == "mutag":
        data = MUTAGDataset()
    elif name == "bgs":
        data = BGSDataset()
    elif name == "am":
        data = AMDataset()
    else:
        raise ValueError("Unknown dataset: {}".format(name))
    return data
