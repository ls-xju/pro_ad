import time
import torch.nn
import torch.optim as optim
import Tools.diffusion
from binary.BModel import ModelP
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


def Hypergraph_w_ed(dataname, device, train_x, train_y, test_x, test_y, args):
    st = time.time()
    if args.t == 0:

        train_x, train_y = train_x, train_y


    else:
        train_x, train_y,_ = Tools.diffusion.diffusion(args.t, "linear", train_x, train_y)

    # weight: t/f
    w_list = args.w_list
    mad = 'f'
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
        {'params': net.fc.parameters()},
        {'params': net.fc1.parameters()},
        {'params': net.fc2.parameters()},
        {'params': net.fc1_update.parameters()},
        {'params': net.fc1_reset.parameters()},
        {'params': net.fc2_reset.parameters()},
        {'params': net.fc2_update.parameters()},
        {'params': net.discrimtor.parameters(), 'lr': 0.00001}
    ], lr=0.01)
    return opt

