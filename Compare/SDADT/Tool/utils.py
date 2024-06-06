
import math
import os
import random

import matplotlib
import numpy as np
import scipy
import sklearn.metrics
import torch
from matplotlib import pyplot as plt
from sklearn import manifold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

def seed_torch(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) #
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	# torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True


# anom_data = np.array([random_permutation(sample) for sample in x])
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

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        return torch.linspace(1e-6, 0.02, num_diffusion_timesteps)
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t+1e-10) / (1+1e-10) * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.02
                        ):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []

    for i in range(num_diffusion_timesteps):
        t1 = i / (num_diffusion_timesteps)
        t2 = (i + 1) / (num_diffusion_timesteps)
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas)

def get_err_threhold(fpr, tpr, threshold):
    differ_tpr_fpr_1=tpr+fpr-1.0
    right_index = np.argmin(np.abs(differ_tpr_fpr_1))
    best_th = threshold[right_index]
    dr = tpr[right_index]
    far = fpr[right_index]
    return dr, far, best_th, right_index

def CalMetrics(test_y, error):
    auc = sklearn.metrics.roc_auc_score(test_y.cpu(), error.cpu())
    pr = sklearn.metrics.average_precision_score(test_y.cpu(), error.cpu())

    return auc, pr

def Metrics(test_y, error):
    auc = sklearn.metrics.roc_auc_score(test_y, error)
    pr = sklearn.metrics.average_precision_score(test_y, error)

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(test_y, error, pos_label=1)
    dr, far, best_th, _ = get_err_threhold(fpr, tpr, thresholds)
    test_labels = np.where(error > best_th, 1, 0)
    f1 = sklearn.metrics.f1_score(test_y, test_labels)

    return auc, pr, f1