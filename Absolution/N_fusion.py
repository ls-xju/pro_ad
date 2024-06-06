import time
import torch.nn
import torch.optim as optim
import Tools.diffusion
from Tools.utils import *

class AutomaticWeightedLoss(torch.nn.Module):
    def __init__(self, num=3):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum


def N_fusion(dataname, device, train_x, train_y, test_x, test_y, args):
    st = time.time()
    if args.t == 0:
        train_x, train_y = train_x, train_y


    else:
        train_x, train_y,_ = Tools.diffusion.diffusion(args.t, "linear", train_x, train_y)

    # weight: t/f
    w_list = args.w_list
    mad = args.mad
    num_embedding = args.num_embedding
    # neighbors
    k_nebor = args.k_nebor
    num_features = train_x.shape[1]

    # convert
    train_x, train_y, test_x, test_y, hg_train, hg_test = convert(args, device, k_nebor, train_x, train_y, test_x, test_y, w_list,mad)
    timetaken = time.time() - st
    # print('hypergraph time:', timetaken)


    auclist, prlist, timetaken = train(dataname, train_x, train_y, test_x, test_y, hg_train, hg_test,num_embedding,num_features,device, args)

    return auclist, prlist, timetaken

def train(dataname,train_x, train_y, test_x, test_y, hg_train, hg_test,num_embedding, num_features,device,args):
    st = time.time()
    # Initial
    net = ModelP(train_x.shape[0], num_features, num_embedding, args.width, device)

    mse = torch.nn.MSELoss()
    aaa = torch.nn.L1Loss(reduction='mean')
    crossloss = torch.nn.CrossEntropyLoss(label_smoothing=0.0)

    optimizer = opt(net)
    loss_module = AutomaticWeightedLoss(num=3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)

    auclist = []
    prlist = []
    maxaucs = 0

    for epoch in range(200):
        net.train()
        optimizer.zero_grad()

        xe, outs, pro,label = net(train_x, hg_train)

        reconloss = mse(train_x, outs)

        dis = crossloss(label,train_y)

        proto1 = pro.reshape(-1,xe.shape[1])
        proto1 = proto1.repeat_interleave(xe.shape[0], dim=0)
        proloss = aaa(proto1,xe)

        L = loss_module.forward(reconloss, dis, proloss)

        L.backward()
        optimizer.step()


        auclist, prlist, maxaucs = Val(net, test_x, hg_test, test_y,  auclist, prlist, pro, maxaucs,dataname)

        scheduler.step()

    timetaken = time.time() - st
    auclist, prlist, timetaken = printResults(dataname, auclist, prlist, timetaken)

    return auclist, prlist, timetaken

def Val(net, test_x, hg_test, test_y, maxauc, maxpr,proto,maxaucs,dataname):
    with torch.no_grad():
        net.eval()
        # -----set model to validate mode, so it only returns the embedded space----- #
        net.trainmodel = False
        xe, outs= net(test_x, hg_test)

        error = getDistanceToPro(xe, proto)

        # Calculating the metrics
        auc, pr = CalMetrics(test_x.cpu().numpy(), test_y.cpu(), error.cpu())

        maxauc.append(auc)
        maxpr.append(pr)
        net.trainmodel = True

    return maxauc, maxpr, maxaucs


def opt(net):
    opt = optim.Adam([
        {'params': net.encoder.parameters(),'lr':0.001},
        {'params': net.decoder.parameters(),'lr':0.001},
        {'params': net.discrimtor.parameters(), 'lr': 0.00001}
    ], lr=0.01)
    return opt


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

        a = hg.state_dict['raw_groups']
        e_values = [entry['w_e'] for entry in a['main'].values()]
        we_tensor = torch.tensor(e_values).to("cuda")

        X = hg.v2v(X, aggr="mean",drop_rate=0, e_weight=we_tensor)
        if not self.is_last:
            X = self.drop(self.act(X))
        return X

class HGNNPD(nn.Module):
    r"""
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
    r"""
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
    r"""
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

        self.attention_pooling = AttentionPooling(self.input_size)
        self.mean_pooling = MeanPolling()

        # Discriminator
        self.discrimtor = HGNNPDis(self.num_embedding, self.width * self.num_embedding, 2, use_bn=True).to(device)

    def forward(self, X, hg):

        x_e, x_e1 = self.encoder(X, hg)

        outs = x_e

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
