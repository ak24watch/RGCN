from Layer import RgcnLayer
import dgl
import torch.nn as nn


class RgcnModel(nn.Module):
    def __init__(
        self,
        hiden_dim,
        out_dim,
        num_rels,
        regularizer="basis",
        num_bases=0,
        bias=False,
        activation=None,
        self_loop=True,
        dropout=0.0,
        layer_norm=False,
        g=None,
    ):
        """
        Initialize the RGCN model.

        Args:
            hiden_dim (int): Hidden dimension size.
            out_dim (int): Output dimension size.
            num_rels (int): Number of relations.
            regularizer (str): Regularizer type.
            num_bases (int): Number of bases.
            bias (bool): Whether to use bias.
            activation (callable): Activation function.
            self_loop (bool): Whether to add self-loops.
            dropout (float): Dropout rate.
            layer_norm (bool): Whether to use layer normalization.
            g (DGLGraph): Input graph.
        """
        super().__init__()
        self.graph = g

        self.emb = nn.Embedding(g.num_nodes(), hiden_dim)
        self.conv1 = RgcnLayer(hiden_dim, hiden_dim, num_rels, regularizer, num_bases, bias, activation, self_loop, dropout, layer_norm)
        self.conv2 = RgcnLayer(hiden_dim, out_dim, num_rels, regularizer, num_bases, bias, activation, self_loop, dropout, layer_norm)

    def forward(self):
        """
        Forward pass of the RGCN model.

        Returns:
            torch.Tensor: Output node features.
        """
        x = self.emb.weight
        h = self.conv1(self.graph, x, self.graph.edata[dgl.ETYPE])
        h = self.conv1(self.graph, h + x, self.graph.edata[dgl.ETYPE])
        return h
