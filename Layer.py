import os

os.environ["DGLBACKEND"] = "pytorch"
import torch
import torch.nn as nn

class RgcnLayer(nn.Module):
    """
    RGCN Layer for relational graph convolutional networks.

    Parameters:
    in_dim (int): Input feature dimension.
    out_dim (int): Output feature dimension.
    num_rels (int): Number of relation types.
    regularizer (str): Type of regularizer to use ('basis').
    num_bases (int): Number of bases to use for basis regularizer.
    bias (bool): Whether to include a bias term.
    activation (callable): Activation function to apply.
    self_loop (bool): Whether to include self-loop.
    dropout (float): Dropout rate.
    layer_norm (bool): Whether to apply layer normalization.
    """
    def __init__(self, in_dim, out_dim, num_rels, regularizer=None, num_bases=None, bias=False, activation=nn.ReLU(), self_loop=True, dropout=0.0, layer_norm=False):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.regularizer = regularizer
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.dropout = dropout
        self.layer_norm = layer_norm

        if self.regularizer == "basis":
            self.v_b = nn.Parameter(torch.Tensor(self.num_bases, self.in_dim, self.out_dim))
            self.a_rb = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))
            nn.init.xavier_uniform_(self.v_b, gain=nn.init.calculate_gain("relu"))
            nn.init.xavier_uniform_(self.a_rb, gain=nn.init.calculate_gain("relu"))

        if self.bias:
            self.bias = nn.Parameter(torch.Tensor(out_dim))
            nn.init.zeros_(self.bias)
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain("relu"))

        if self.layer_norm:
                self.lr_norm = nn.LayerNorm(out_dim)

        self.dropout = nn.Dropout(dropout)

    def msgFunc(self, edges):
        """
        User-defined message function.

        Parameters:
        edges: The edges of the graph.

        Returns:
        dict: The message to pass along the edges.
        """
        rel_weight = edges.data["w"]
        src_h = edges.src["h"].unsqueeze(1)
        msg = torch.bmm(src_h, rel_weight).squeeze(1)
        msg = msg * edges.data["norm"]
        return {"msg": msg}

    def reduceFunc(self, nodes):
        """
        Reduce function for summing messages.

        Parameters:
        nodes: The nodes of the graph.

        Returns:
        dict: The aggregated message for each node.
        """
        aggregated_msg = torch.sum(nodes.mailbox["msg"], dim=1).squeeze(1)
        if self.self_loop:
            self_emb = torch.matmul(nodes.data["h"], self.loop_weight)
            aggregated_msg_with_self_loop = aggregated_msg + self_emb
        return {"h": aggregated_msg_with_self_loop}

    def forward(self, g, feat, etype):
        """
        Forward pass of the RGCN layer.

        Parameters:
        g: The graph.
        feat: The input features.
        etype: The edge types.

        Returns:
        Tensor: The output features.
        """
        with g.local_scope():
            if self.regularizer == "basis":
                v_b_unsqueezed = self.v_b.unsqueeze(0)
                a_rb_unsqueezed = self.a_rb.unsqueeze(-1).unsqueeze(-1)
                rel_weight = (v_b_unsqueezed * a_rb_unsqueezed).sum(1)
            else:
                raise NotImplementedError("Only basis regularizer is implemented.")
            g.edata["w"] = rel_weight[etype]
            g.ndata["h"] = feat
            g.update_all(self.msgFunc, self.reduceFunc)
            h = g.dstdata["h"]

            if self.bias:
                h = h + self.bias

            if self.activation:
                h = self.activation(h)
            
            if self.dropout:
                h = self.dropout(h)

            if self.layer_norm:
                h = self.lr_norm(h)
            return h
