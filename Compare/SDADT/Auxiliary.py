import torch.nn as nn

import torch.optim as optim
from sklearn.cluster import MiniBatchKMeans

# import metrics.sdhots
from Compare.SDADT.Model import DCFC
from Compare.SDADT.Tool.utils import *
import warnings
from sklearn import metrics

warnings.simplefilter(action='ignore', category=FutureWarning)

# -----indicate which gpu to use for training, devices list will be used in training with DataParellel----- #
gpu = '0'
if gpu == '0':
    cuda = torch.device('cuda:0')
    devices = [0, 1, 2, 3]
else:
    raise NameError('no more GPUs')

def clustering(model, mbk, x, label):
    """
    Initialize cluster centroids with minibatch Kmeans
    Args:
        model:DCFC
        mbk: minibatch Kmeans
        x: embedded x

    Returns: N/A

    """
    model.eval()
    x_e = model(x.float())

    mask_or = [label == 0]
    mask_gs = [label == 1]

    cluster_or = x_e[mask_or]
    cluster_gs = x_e[mask_gs]

    # Clustering centroids for real data
    mbk.partial_fit(cluster_or.data.cpu().numpy())
    # mbk.fit(cluster_or.data.cpu().numpy())
    model.cluster_centers_or = mbk.cluster_centers_  # keep the cluster centers
    model.clusterCenter_or.data = torch.from_numpy(model.cluster_centers_or).to(cuda)

    # Auxiliary data clustering centroids
    mbk.partial_fit(cluster_gs.data.cpu().numpy())
    # mbk.fit(cluster_gs.data.cpu().numpy())
    model.cluster_centers_gs = mbk.cluster_centers_  # keep the cluster centers
    model.clusterCenter_gs.data = torch.from_numpy(model.cluster_centers_gs).to(cuda)

def getTDistance(model, x, label):
    """
    Obtain the distance to centroid for each instance, and calculate the weight module based on that
    Args:
        model: DCFC
        x: embedded x

    Returns: clustering distribution

    """
    mask_or = [label == 0]
    mask_gs = [label == 1]

    cluster_or = x[mask_or]
    cluster_gs = x[mask_gs]

    # dist, dist_to_centers = model.module.getDistanceToClusters(x)
    dist_or, dist_to_centers_or = model.getDistanceToClusters_or(cluster_or)
    dist_to_centers_or = torch.mean(dist_to_centers_or)

    cd = max(0, 0.5-model.ccdistance())

    dist_gs, dist_to_centers_gs = model.getDistanceToClusters_gs(cluster_gs)
    dist_to_centers_gs = torch.mean(dist_to_centers_gs)



    return dist_to_centers_or, dist_to_centers_gs

def Train(dataname, model, train_input, labels,  epochs, batch, lr_cluster, args):

    model.train()
    mbk = MiniBatchKMeans(n_clusters=model.num_classes, n_init=20, batch_size=batch)
    # mbk = KMeans(n_clusters=model.num_classes)
    got_cluster_center = False
    Loss1 = 0.0
    OR = 0.0
    GS = 0.0
    # learning rate
    lr_sae = 0.01
    lr_cluster = 0.01

    a = args.auxiliarya

    optimizer = optim.SGD([
        {'params': model.encoder.parameters()},
        {'params': model.out.parameters()},
        {'params': model.clusterCenter_or, 'lr': lr_cluster},
        {'params': model.clusterCenter_gs, 'lr': lr_cluster}
    ], lr=lr_sae, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)# 调整学习率 https://blog.csdn.net/qyhaill/article/details/103043637
    # loss_module = AutomaticWeightedLoss(num=3)
    max_auc = 0
    max_acc = 0
    max_f1 = 0

    for epoch in range(200):
        for i in range(train_input.shape[0] // batch):
            input_batch = train_input[i * batch: (i + 1) * batch]
            x = input_batch.to(cuda)

            # -----use minibatch Kmeans to initialize the cluster centroids for the clustering layer----- #
            if not got_cluster_center:
                model.setClusteringMode(True)
                total = train_input.to(cuda)
                clustering(model, mbk, total, labels)
                got_cluster_center = True
                model.setClusteringMode(False)

            # Start training the model after initializing the centroids
            else:
                model.train()
                x_e, out = model(x)

                # -----obtain the clustering distance---- #
                dis_or, dis_gs = getTDistance(model, x_e, labels[i * batch: (i + 1) * batch])

                CELoss = nn.CrossEntropyLoss().to(cuda)
                clssify_loss = CELoss(out, labels[i * batch: (i + 1) * batch].long().to(cuda))

                Loss = clssify_loss + a * (dis_or + dis_gs)
                optimizer.zero_grad()
                Loss.backward()
                optimizer.step()

                Loss1 += Loss.data.cpu()
                OR+= dis_or
                GS+= dis_gs

        OR = 0.0
        GS = 0.0
        max_loss = 1000000000

        model.eval()
        with torch.no_grad():
            train_input = train_input.to(cuda)
            xe, out = model(train_input)
            pred = out.argmax(dim=1)
            mask_or = [pred == 0]
            data_or = xe[mask_or]

            train_y = pred[mask_or].cpu()

            fpr, tpr, thresholds = metrics.roc_curve(labels, pred.cpu(), pos_label=1)
            auc = metrics.auc(fpr, tpr)
            acc = metrics.accuracy_score(labels, pred.cpu())
            f1 = metrics.f1_score(labels, pred.cpu())

            if auc > max_auc and acc > max_acc:
                max_loss = Loss1
                max_auc = auc
                max_acc = acc
                max_xe = xe
                max_data = data_or
                max_data_label = train_y

                # print("save model")
                if not os.path.exists(f'./Compare/SDADT/Load_model/{args.dataname}'):
                    os.makedirs(f'./Compare/SDADT/Load_model/{args.dataname}')

                torch.save(model, f'./Compare/SDADT/Load_model/{args.dataname}/{dataname}.pt')
        # -----show loss every 50 epoch---- #
        if epoch % 99 ==0:
            print(f'[{epoch}],Loss:{Loss1:.4f}, AUC:{auc:.4f}')
        Loss1 = 0.0

        scheduler.step()


    return max_data, max_data_label, max_xe

def Auxiliary(args):

    # real data
    train_x = torch.tensor(args.train_x).float()
    train_y = torch.tensor(args.train_y).float()

    noise = torch.randn_like(train_x)
    label_noise = torch.ones(train_x.shape[0]).float()

    # datas with real data and auxiliary
    datas = torch.cat((train_x, noise), dim=0)
    labels = torch.cat((train_y, label_noise), dim=0)

    X_norm, Y = shuffle(datas, labels)

    if datas.shape[0] < 500:
        configuration = 300, 64
    else:
        configuration = 200, 128

    # -----run the model---- #
    model = DCFC(args.feature_dimension, args.num_class, args.embedded_dimension, args.device)

    latent_data, latent_data_label, max_xe = Train(args.dataname, model,  X_norm, Y,  configuration[0], configuration[1], args.lr_cluster, args)


    # Returns the real training data after the auxiliary learning module
    return latent_data, latent_data_label


def Auxiliary_Test(args):

    model = torch.load(f'./Compare/SDADT/Load_model/{args.dataname}/{args.dataname}.pt')
    model = model.eval()
    with torch.no_grad():
        test_x = torch.tensor(args.test_x).float().to(cuda)
        xe, out = model(test_x)
        xe = xe.cpu().detach().numpy()


    return xe, out