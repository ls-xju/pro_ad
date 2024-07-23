import time

import math
import os

import torch
import pandas as pd
import sklearn.cluster
seed = 42
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
import random
import numpy as np
import scipy
import sklearn
import sklearn.decomposition
import sklearn.manifold
import sklearn.metrics
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from Tools.hypergraph import Hypergraph
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture



dataname_list = ['WPBC', 'wdbc', 'breastw', 'pima', 'cardio', 'cardiotocography', 'thyroid', 'Stamps',
                     'SpamBase',
                     'CIFAR10_0', 'mnist', 'celeba', 'glass', 'yeast', 'speech', 'wilt',
                     'landsat', 'imdb', 'campaign', 'census',
                     'http', 'kddcup99', 'wine', 'musk',
                      'fault','waveform']


def seed_torch(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True


def getdataNN(dataname, rato):

    data = np.load(f'./datasets/{dataname}.npz', allow_pickle=True)

    label = data['y'].astype('float32')
    data = data['X'].astype('float32')
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    # if the dataset is too large, subsampling for considering the computational cost
    if len(label) >= 10000:
        idx_sample = np.random.choice(np.arange(len(label)), 10000, replace=False)
        data = data[idx_sample]
        label = label[idx_sample]

    normal_data = data[label == 0]
    normal_label = label[label == 0]
    anom_data = data[label == 1]
    anom_label = label[label == 1]

    test_idy = np.random.choice(np.arange(0, len(anom_data)), int(len(anom_data)*rato), replace=False)
    test_idx = np.random.choice(np.arange(0, len(normal_data)), int(len(normal_data)*rato), replace=False)

    train_idx = np.setdiff1d(np.arange(0, len(normal_data)), test_idx)

    train_x = normal_data[train_idx]
    train_y = normal_label[train_idx]

    test_x = np.concatenate((normal_data[test_idx], anom_data[test_idy]))
    test_y = np.concatenate((normal_label[test_idx], anom_label[test_idy]))

    train_x, train_y = shuffle(train_x, train_y)
    test_x, test_y = shuffle(test_x, test_y)



    return train_x, train_y, test_x, test_y






def save_data(data,label,dataname,data_type):
    # dataname_list = ['WPBC', 'wdbc', 'breastw', 'pima', 'cardio', 'Stamps',
    #                  'SpamBase', 'glass', 'yeast', 'wilt',
    #                  'landsat'
    #                  ]
    result_axis0 = np.concatenate((data, label.reshape(-1, 1)), axis=1)
    save_dir = f'./datasets/generated_data/{data_type}'
    # 将NumPy数组保存为NumPy文件（.npy）
    np.save(os.path.join(save_dir, f'{dataname}.npy'), result_axis0)
    print(dataname)



def shuffle(X, Y):
    """
    Shuffle the datasets
    Args:
        X: input data
        Y: labels

    Returns: shuffled sets
    """
    random_index = np.random.permutation(X.shape[0])
    return X[random_index], Y[random_index]

def convert(args, device, k_nebor, train_x, train_y, test_x, test_y,w_list,mad):

    # tensor
    train_x = torch.tensor(train_x).to(device).float()
    train_y = torch.tensor(train_y).to(device).long()
    test_x = torch.tensor(test_x).to(device).float()
    test_y = torch.tensor(test_y).to(device).long()

    if w_list == 't' and mad == 't':
        s0 = time.time()
        hg_train = from_feature_kNN(args,train_x, k=k_nebor,w_list=True, mad=True)
        hg_train = hg_train.to(device)
        train_g_time = time.time() -s0

        s1 = time.time()
        hg_test = from_feature_kNN(args,test_x, k=k_nebor,w_list=True, mad=True)
        test_g_time = time.time() - s1
        hg_test = hg_test.to(device)


    elif w_list == 't'and mad == 'f':
        s0 = time.time()
        hg_train = from_feature_kNN(args,train_x, k=k_nebor,w_list=True, mad=False)
        hg_train = hg_train.to(device)
        train_g_time = time.time() - s0
        s1= time.time()
        hg_test = from_feature_kNN(args,test_x, k=k_nebor,w_list=True, mad=False)
        test_g_time = time.time() - s1
        hg_test = hg_test.to(device)


    elif w_list == 'f':
        s0 = time.time()
        hg_train = from_feature_kNN(args,train_x, k=k_nebor, w_list=False)
        hg_train = hg_train.to(device)
        train_g_time = time.time() - s0
        s1 = time.time()
        hg_test = from_feature_kNN(args,test_x, k=k_nebor, w_list=False)
        test_g_time = time.time() - s1
        hg_test = hg_test.to(device)


    return train_x, train_y, test_x, test_y, hg_train, hg_test,train_g_time,test_g_time


def Metrics(test_y, error):
    auc = roc_auc_score(test_y, error)
    pr = sklearn.metrics.average_precision_score(test_y, error)
    return auc, pr

def CalMetrics(test_x, test_y, error):
    auc = roc_auc_score(test_y.cpu(), error.cpu())
    pr = sklearn.metrics.average_precision_score(test_y.cpu(), error.cpu())

    return auc, pr

def printResults(dataname, auclist, prlist, tims):

    max_index = np.argmax(auclist)
    auc = auclist[max_index]
    pr = prlist[max_index]

    return auc, pr, tims

def _e_list_from_feature_kNN(features: torch.Tensor, k: int):
    r"""Construct hyperedges from the feature matrix. Each hyperedge in the hypergraph is constructed by the central vertex ans its :math:`k-1` neighbor vertices.

    Args:
        ``features`` (``torch.Tensor``): The feature matrix.
        ``k`` (``int``): The number of nearest neighbors.
    """
    features = features.cpu().numpy()

    assert features.ndim == 2, "The feature matrix should be 2-D."
    assert (
            k <= features.shape[0]
    ), "The number of nearest neighbors should be less than or equal to the number of vertices."

    # cKDTree
    tree = scipy.spatial.cKDTree(features)
    dist, nbr_array = tree.query(features, k=k)
    return dist, nbr_array.tolist()

def _e_list_from_feature_kNN11(features: torch.Tensor, k: int):
    r"""Construct hyperedges from the feature matrix. Each hyperedge in the hypergraph is constructed by the central vertex ans its :math:`k-1` neighbor vertices.

    Args:
        ``features`` (``torch.Tensor``): The feature matrix.
        ``k`` (``int``): The number of nearest neighbors.
    """
    # features = features.cpu().numpy()
    #
    # assert features.ndim == 2, "The feature matrix should be 2-D."
    # assert (
    #         k <= features.shape[0]
    # ), "The number of nearest neighbors should be less than or equal to the number of vertices."
    #
    # # cKDTree
    # tree = scipy.spatial.cKDTree(features)
    # dist, nbr_array = tree.query(features, k=k)


    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(features)

    dist, nbr_array = neigh.kneighbors(features, n_neighbors=k)



    return dist, nbr_array.tolist()


from joblib import Parallel, delayed
def calculate_weight(edge, features, alpha):
    node_indices = np.array(edge)
    data = features[node_indices].cpu().numpy()
    medians = np.median(data, axis=0)
    mad_distances = np.median(np.abs(data - medians), axis=0)
    dis_nor = np.mean(np.exp(-alpha * mad_distances))
    return dis_nor


def from_feature_kNN(args, features: torch.Tensor, k: int, device: torch.device = torch.device("cpu"), w_list: bool = True, mad: bool = True):
    r"""Construct the hypergraph from the feature matrix. Each hyperedge in the hypergraph is constructed by the central vertex ans its :math:`k-1` neighbor vertices.

    .. note::
        The constructed hypergraph is a k-uniform hypergraph. If the feature matrix has the size :math:`N \times C`, the number of vertices and hyperedges of the constructed hypergraph are both :math:`N`.

    Args:
        ``features`` (``torch.Tensor``): The feature matrix.
        ``k`` (``int``): The number of nearest neighbors.
        ``device`` (``torch.device``, optional): The device to store the hypergraph. Defaults to ``torch.device('cpu')``.
    """

    # dis_array, e_list = _e_list_from_feature_kNN(features.cpu(), k)
    dis_array, e_list = _e_list_from_feature_kNN11(features.cpu(), k)
    # tips MAD_distance
    if w_list and mad:
        w_list = []
        alpha = args.alpha

        features_np = features.cpu().numpy()

        e_array = np.array(e_list)

        all_data = features_np[e_array]

        medians = np.median(all_data, axis=1)
        mad_distances = np.median(np.abs(all_data - medians[:, np.newaxis, :]), axis=1)

        dis_nor = np.mean(np.exp(-alpha * mad_distances), axis=1)

        w_list = dis_nor.tolist()

        hg = Hypergraph(features.shape[0], e_list, w_list, device=device)

    if w_list and not mad:
    # # tips ED_distance
        w_list = []
        m_dist1 = sklearn.metrics.pairwise_distances(features.cpu())
        avg_dist = np.median(m_dist1)

        for i, edge in enumerate (e_list):
            node_indices = np.array(edge)
            data = features[node_indices].cpu()

            lower_triangle = sklearn.metrics.pairwise_distances(data)
            exp_term = np.mean(np.exp(-(lower_triangle ** 2 / avg_dist ** 2)))

            w_list.append(exp_term)

        hg = Hypergraph(features.shape[0], e_list, w_list, device=device)

    if not w_list:
        hg = Hypergraph(features.shape[0], e_list, device=device)

    return hg

def getDistanceToPro(x, pro):
    """
    obtain the distance to prototype for each instance
    Args:
        x: sample on the embedded space
    Returns: square of the euclidean distance, and the euclidean distance
    """

    xe = torch.unsqueeze(x, 1) - pro
    dist_to_centers = torch.sum(torch.mul(xe, xe), 2)
    euclidean_dist = torch.sqrt(dist_to_centers)

    return euclidean_dist.squeeze()

def get_err_threhold(fpr, tpr, threshold):
    differ_tpr_fpr_1=tpr+fpr-1.0
    right_index = np.argmin(np.abs(differ_tpr_fpr_1))
    best_th = threshold[right_index]
    dr = tpr[right_index]
    far = fpr[right_index]
    return dr, far, best_th, right_index


def convert_load(args, device, k_nebor, test_x, test_y,w_list,mad):

    # tensor
    test_x = torch.tensor(test_x).to(device).float()
    test_y = torch.tensor(test_y).to(device).long()

    if w_list == 't' and mad == 't':
        hg_test = from_feature_kNN(args,test_x, k=k_nebor,w_list=True, mad=True)
        hg_test = hg_test.to(device)

    elif w_list == 't'and mad == 'f':
        hg_test = from_feature_kNN(args,test_x, k=k_nebor,w_list=True, mad=False)
        hg_test = hg_test.to(device)

    elif w_list == 'f':
        hg_test = from_feature_kNN(args,test_x, k=k_nebor, w_list=False)
        hg_test = hg_test.to(device)

    return  test_x, test_y, hg_test
