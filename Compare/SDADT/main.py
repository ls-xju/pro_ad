import argparse
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from Compare.SDADT.Auxiliary import *
from Compare.SDADT.Diffusion import *

def train(args):

    if args.auxiliary:

        # training data after the auxiliary learning module
        latent_data, latent_data_label= Auxiliary(args)

        # # testing data after the auxiliary learning module
        latent_test_data, outs = Auxiliary_Test(args)

        args.test_x = latent_test_data

        # training the diffusion model
        train_x = latent_data.cpu().float()
        train_y = latent_data_label
        maxauc, maxpr = Diffusion(args, train_x, train_y)

        return maxauc, maxpr

def launch_SDAD(dataname,train_x, train_y, test_x, test_y,T,lamda):
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # Parameters required
    args.auxiliary = True

    args.num_class = 1

    # Clustering learning rate
    args.lr_cluster = 0.01
    args.auxiliarya = 0.1

    # dataname
    args.dataname = dataname

    # train dataset
    args.train_x = train_x
    args.train_y = train_y

    # test dataset
    args.test_x = test_x
    args.test_y = test_y

    # auxiliary learning module
    args.feature_dimension = args.train_x.shape[1]
    args.embedded_dimension = 24 if args.feature_dimension < 100 else 64

    #diffusion model parameters
    args.units = 500
    args.num_steps = T
    args.lamda = lamda
    args.device = "cuda"
    # cosine
    args.betas = get_named_beta_schedule('cosine', args.num_steps) #
    args.betas = args.betas.to(args.device)

    args.alphas = 1 - args.betas
    args.alphas_bar = torch.cumprod(args.alphas, 0)
    args.alphas_bar_sqrt = torch.sqrt(args.alphas_bar)
    args.one_minus_alphas_bar_sqrt = torch.sqrt(1 - args.alphas_bar)

    # training
    maxauc, maxpr = train(args)

    return maxauc, maxpr




