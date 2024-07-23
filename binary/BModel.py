import torch
import torch.nn
from Tools.hypergraph import Hypergraph
from torch import nn
from torch.autograd import Function

class HGNNPConv(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            bias: bool = True,
            use_bn: bool = False,
            drop_rate: float = 0.5,
            is_last: bool = False,
    ):
        super().__init__()
        self.is_last = is_last
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else None
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.theta = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, X: torch.Tensor, hg: Hypergraph) -> torch.Tensor:
        r"""The forward function.

        Args:
            X (``torch.Tensor``): Input vertex feature matrix. Size :math:`(|\mathcal{V}|, C_{in})`.
            hg (``dhg.Hypergraph``): The hypergraph structure that contains :math:`|\mathcal{V}|` vertices.
        """
        X = self.theta(X)
        if self.bn is not None:
            X = self.bn(X)
        #tips  drop_rate：Randomly dropout the connections in incidence matrix with probability
        # e_weight a = self.state_dict['raw_groups']
        # we_tensor = torch.tensor(we_values)

        a = hg.state_dict['raw_groups']
        e_values = [entry['w_e'] for entry in a['main'].values()]
        we_tensor = torch.tensor(e_values).to("cuda")

        X = hg.v2v(X, aggr="mean",drop_rate=0, e_weight=we_tensor)
        if not self.is_last:
            X = self.drop(self.act(X))
        return X

class HGNNPD(nn.Module):
    r"""The HGNN :sup:`+` model proposed in `HGNN+: General Hypergraph Neural Networks <https://ieeexplore.ieee.org/document/9795251>`_ paper (IEEE T-PAMI 2022).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``hid_channels`` (``int``): :math:`C_{hid}` is the number of hidden channels.
        ``num_classes`` (``int``): The Number of class of the classification task.
        ``use_bn`` (``bool``): If set to ``True``, use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``, optional): Dropout ratio. Defaults to ``0.5``.
    """

    def __init__(
        self,
        in_channels: int,
        hid_channels: int,
        num_classes: int,
        use_bn: bool = False,
        drop_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()

        self.layers.append(
            HGNNPConv(in_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate)
        )

        self.layers.append(
            HGNNPConv(hid_channels, num_classes, use_bn=use_bn, is_last=True)
        )


    def forward(self, X: torch.Tensor, hg: "dhg.Hypergraph") -> torch.Tensor:
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``hg`` (``dhg.Hypergraph``): The hypergraph structure that contains :math:`N` vertices.
        """
        # for layer in self.layers:
        #     X = layer(X, hg)
        X1 = self.layers[0](X, hg)
        X = self.layers[1](X1, hg)

        return X

class HGNNPE(nn.Module):
    r"""The HGNN :sup:`+` model proposed in `HGNN+: General Hypergraph Neural Networks <https://ieeexplore.ieee.org/document/9795251>`_ paper (IEEE T-PAMI 2022).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``hid_channels`` (``int``): :math:`C_{hid}` is the number of hidden channels.
        ``num_classes`` (``int``): The Number of class of the classification task.
        ``use_bn`` (``bool``): If set to ``True``, use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``, optional): Dropout ratio. Defaults to ``0.5``.
    """

    def __init__(
        self,
        in_channels: int,
        hid_channels: int,
        num_classes: int,
        use_bn: bool = False,
        drop_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        # self.attens = nn.ModuleList()
        self.layers.append(
            HGNNPConv(in_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate)
        )

        self.layers.append(
            HGNNPConv(hid_channels, num_classes, use_bn=use_bn, is_last=True)
        )


    def forward(self, X: torch.Tensor, hg: "dhg.Hypergraph") -> torch.Tensor:
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``hg`` (``dhg.Hypergraph``): The hypergraph structure that contains :math:`N` vertices.
        """
        # for layer in self.layers:
        #     X = layer(X, hg)
        X1 = self.layers[0](X, hg)
        X = self.layers[1](X1, hg)

        return X,X1

class HGNNPDis(nn.Module):
    r"""The HGNN :sup:`+` model proposed in `HGNN+: General Hypergraph Neural Networks <https://ieeexplore.ieee.org/document/9795251>`_ paper (IEEE T-PAMI 2022).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``hid_channels`` (``int``): :math:`C_{hid}` is the number of hidden channels.
        ``num_classes`` (``int``): The Number of class of the classification task.
        ``use_bn`` (``bool``): If set to ``True``, use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``, optional): Dropout ratio. Defaults to ``0.5``.
    """

    def __init__(
        self,
        in_channels: int,
        hid_channels: int,
        num_classes: int,
        use_bn: bool = False,
        drop_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        # self.attens = nn.ModuleList()
        self.layers.append(
            HGNNPConv(in_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate)
        )

        self.layers.append(
            HGNNPConv(hid_channels, num_classes, use_bn=use_bn, is_last=True)
        )


    def forward(self, X: torch.Tensor, hg: "dhg.Hypergraph") -> torch.Tensor:
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``hg`` (``dhg.Hypergraph``): The hypergraph structure that contains :math:`N` vertices.
        """
        # for layer in self.layers:
        #     X = layer(X, hg)
        X1 = self.layers[0](X, hg)
        X = self.layers[1](X1, hg)

        return X
class AttentionPooling(nn.Module):
    def __init__(self, size):
        super(AttentionPooling, self).__init__()
        self.n = size -1
        self.sigmoid = nn.Sigmoid()


    def forward(self, outs):
        # Calculate attention scores
        n = self.n
        d = (outs - outs.mean(dim=0)).pow(2)
        v = d.sum(dim=0) / n
        e = d / (4 * (v + 0.001)) + 0.5
        proto = torch.sum((outs * self.sigmoid(e)), dim=0)

        return proto
class MeanPolling(nn.Module):
    def __init__(self):
        super(MeanPolling, self).__init__()

    def forward(self, x):
        proto = torch.mean(x, dim=0)
        return proto

class ModelP(nn.Module):
    def __init__(self, input_size, num_features, num_embedding, width, device):
        super(ModelP, self).__init__()
        # Number of instances  N
        self.input_size = input_size
        # Dimension of embedded feature spaces
        self.num_features = num_features
        self.num_embedding = num_embedding
        self.width = width
        self.trainmodel = True


        # # Encoder
        self.encoder = HGNNPE(self.num_features, self.width * self.num_features, self.num_embedding, use_bn=True).to(device)
        # Decoder
        self.decoder = HGNNPD(self.num_embedding, self.width * self.num_features, self.num_features, use_bn=True).to(device)

        self.fc = nn.Linear(self.width * num_features, num_embedding).to(device)

        self.fc1_update = nn.Linear(num_embedding, num_embedding).to(device)
        self.fc2_update = nn.Linear(num_embedding, num_embedding).to(device)

        self.fc1_reset = nn.Linear(num_embedding, num_embedding).to(device)
        self.fc2_reset = nn.Linear(num_embedding, num_embedding).to(device)

        self.fc1 = nn.Linear(num_embedding, num_embedding).to(device)
        self.fc2 = nn.Linear(num_embedding, num_embedding).to(device)

        self.attention_pooling = AttentionPooling(self.input_size)
        self.mean_pooling = MeanPolling()

        # Discriminator
        self.discrimtor = HGNNPDis(self.num_embedding, self.width * self.num_embedding, 2, use_bn=True).to(device)


    def forward(self, X, hg):

        x_e, x_e1 = self.encoder(X, hg)
        x_e1 = self.fc(x_e1)


        z = torch.sigmoid(self.fc1_update(x_e1) + self.fc2_update(x_e))
        r = torch.sigmoid(self.fc1_reset(x_e1) + self.fc2_reset(x_e))

        out = torch.tanh(self.fc1(x_e1) + self.fc2(r * x_e))

        outs = z * out + (1-z) * x_e


        proto = self.attention_pooling(outs)

        # -----else the discriminator predicts the subgroup assignment for each instance----- #
        reversed_x_e = GradientReversalLayer.apply(outs)
        xdis = self.discrimtor(reversed_x_e,hg)

        if self.trainmodel:
            x_de = self.decoder(outs, hg)
            return outs, x_de, proto, xdis

        x_de = self.decoder(outs, hg)

        return outs, x_de

class GradientReversalLayer(Function):

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg(), None
