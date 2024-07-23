import time
import argparse
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# seed = 42
import warnings
warnings.filterwarnings("ignore")
from Tools.utils import *
from binary.Bprototype import BPROT
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


#  tips Absolutions
from Absolution.Graph_w_default import Graph_w_default
from Absolution.Graph_w_ed import Graph_w_ed
from Absolution.Graph_w_mad import Graph_w_mad
from Absolution.Hypergraph_w_default import Hypergraph_w_default
from Absolution.Hypergraph_w_ed import Hypergraph_w_ed
from Absolution.N_discriminator import N_discriminator
from Absolution.N_fusion import N_fusion
from Absolution.N_attention import N_attention
from Absolution.N_prototype import N_prototype
from Absolution.Neg_gru import Neg_gru

# tips Load_model
from Well_Trained_Models.PHAD.Bprototype_LOAD import BPROT_load
from Well_Trained_Models.Absolution.Graph_w_default.Graph_w_default_LOAD import Graph_w_default_load
from Well_Trained_Models.Absolution.Graph_w_ed.Graph_w_ed_LOAD import Graph_w_ed_load
from Well_Trained_Models.Absolution.Graph_w_mad.Graph_w_mad_LOAD import Graph_w_mad_load
from Well_Trained_Models.Absolution.Hypergraph_w_default.Hypergraph_w_default_LOAD import Hypergraph_w_default_load
from Well_Trained_Models.Absolution.Hypergraph_w_ed.Hypergraph_w_ed_LOAD import Hypergraph_w_ed_load
from Well_Trained_Models.Absolution.Neg_gru.Neg_gru_LOAD import Neg_gru_load
from Well_Trained_Models.Absolution.N_attention.N_attention_LOAD import N_attention_load
from Well_Trained_Models.Absolution.N_fusion.N_fusion_LOAD import N_fusion_load
from Well_Trained_Models.Absolution.N_discriminator.N_discriminator_LOAD import N_discriminator_load


if __name__ == '__main__':
    st = time.time()
    #tips  0(Load_well_trained_model)  1(method)
    compare_alg = 1
    c_alg = 'PHAD'

    # -------load data---------#
    for dataname in dataname_list:

        seed_torch(seed)
        parser = argparse.ArgumentParser()
        args = parser.parse_args()

        args.device = device
        args.num_embedding = 16
        args.width = 7
        # neighbors
        args.k_nebor = 20
        args.w_list = 't' # t f
        args.mad = 't'  # t f
        args.alpha = 0.7
        args.t = 1000   #1000 0

        train_x, train_y, test_x, test_y = getdataNN(dataname, 0.2)

        if compare_alg == 0:
            #  BPROT_load
            #  Graph_w_default_load  Graph_w_ed_load  Graph_w_mad_load   Hypergraph_w_default_load  Hypergraph_w_ed_load
            #  N_discriminator_load N_fusion_load  N_attention_load  Neg_gru_load
            maxauc, maxpr = Neg_gru_load(dataname, device, test_x, test_y, args)

        else:
            #  BPROT
            #  N_diffusion
            #  Graph_w_default  Graph_w_ed  Graph_w_mad  Hypergraph_w_default Hypergraph_w_ed
            #  N_prototype N_discriminator N_fusion N_attention Neg_gru
            maxauc, maxpr, timetaken = BPROT(dataname, device, train_x, train_y, test_x, test_y, args)


        result_str = '{:.4f}, {:.4f}'.format(maxauc, maxpr)
        print(result_str)


    time_taken = time.time() - st
    print("TimeTaken:",time_taken)

