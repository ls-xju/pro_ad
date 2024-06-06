from Tools.utils import *
import time
import argparse
# tips Load_model
from Well_Trained_Models.PHAD.Bprototype_LOAD import BPROT_load
from Well_Trained_Models.Absolution.N_diffusion.N_diffusion_LOAD import N_diffusion_load
from Well_Trained_Models.Absolution.Only_diffusion.Only_diffusion_LOAD import Only_diffusion_load
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
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.device = device
    args.num_embedding = 16
    args.width = 7
    args.k_nebor = 20
    args.w_list = 't'
    args.mad = 't'
    args.alpha = 0.7
    args.t = 1000

    proto_list = []
    xe_list = []

    print("PHAD  (AUC-ROC,RUC-PR)")
    for dataname in dataname_list:
        seed_torch(seed)
        train_x, train_y, test_x, test_y = getdataNN(dataname, 0.2)
        auc, pr = BPROT_load(dataname, device, test_x, test_y, args)

        result_str = '{}: {:.4f}, {:.4f}'.format(dataname, auc, pr)
        print(result_str)


    print("Absolution")
    print("N_diffusion  (AUC-ROC,RUC-PR)")
    for dataname in dataname_list:
        seed_torch(seed)
        train_x, train_y, test_x, test_y = getdataNN(dataname, 0.2)
        auc, pr = N_diffusion_load(dataname, device, test_x, test_y, args)

        result_str = '{}: {:.4f}, {:.4f}'.format(dataname, auc, pr)
        print(result_str)

    print("Only_diffusion  (AUC-ROC,RUC-PR)")
    for dataname in dataname_list:
        seed_torch(seed)
        train_x, train_y, test_x, test_y = getdataNN(dataname, 0.2)
        auc, pr = Only_diffusion_load(dataname, device, test_x, test_y, args)

        result_str = '{}: {:.4f}, {:.4f}'.format(dataname, auc, pr)
        print(result_str)

    print("Graph_w_default  (AUC-ROC,RUC-PR)")
    for dataname in dataname_list:
        seed_torch(seed)
        train_x, train_y, test_x, test_y = getdataNN(dataname, 0.2)
        auc, pr = Graph_w_default_load(dataname, device, test_x, test_y, args)

        result_str = '{}: {:.4f}, {:.4f}'.format(dataname, auc, pr)
        print(result_str)

    print("Graph_w_ed  (AUC-ROC,RUC-PR)")
    for dataname in dataname_list:
        seed_torch(seed)
        train_x, train_y, test_x, test_y = getdataNN(dataname, 0.2)
        auc, pr = Graph_w_ed_load(dataname, device, test_x, test_y, args)

        result_str = '{}: {:.4f}, {:.4f}'.format(dataname, auc, pr)
        print(result_str)

    print("Graph_w_mad  (AUC-ROC,RUC-PR)")
    for dataname in dataname_list:
        seed_torch(seed)
        train_x, train_y, test_x, test_y = getdataNN(dataname, 0.2)
        auc, pr = Graph_w_mad_load(dataname, device, test_x, test_y, args)

        result_str = '{}: {:.4f}, {:.4f}'.format(dataname, auc, pr)
        print(result_str)

    print("Hypergraph_w_default  (AUC-ROC,RUC-PR)")
    for dataname in dataname_list:
        seed_torch(seed)
        train_x, train_y, test_x, test_y = getdataNN(dataname, 0.2)
        auc, pr = Hypergraph_w_default_load(dataname, device, test_x, test_y, args)

        result_str = '{}: {:.4f}, {:.4f}'.format(dataname, auc, pr)
        print(result_str)

    print("Hypergraph_w_ed  (AUC-ROC,RUC-PR)")
    for dataname in dataname_list:
        seed_torch(seed)
        train_x, train_y, test_x, test_y = getdataNN(dataname, 0.2)
        auc, pr = Hypergraph_w_ed_load(dataname, device, test_x, test_y, args)

        result_str = '{}: {:.4f}, {:.4f}'.format(dataname, auc, pr)
        print(result_str)

    print("N_discriminator  (AUC-ROC,RUC-PR)")
    for dataname in dataname_list:
        seed_torch(seed)
        train_x, train_y, test_x, test_y = getdataNN(dataname, 0.2)
        auc, pr = N_discriminator_load(dataname, device, test_x, test_y, args)

        result_str = '{}: {:.4f}, {:.4f}'.format(dataname, auc, pr)
        print(result_str)

    print("N_fusion  (AUC-ROC,RUC-PR)")
    for dataname in dataname_list:
        seed_torch(seed)
        train_x, train_y, test_x, test_y = getdataNN(dataname, 0.2)
        auc, pr = N_fusion_load(dataname, device, test_x, test_y, args)

        result_str = '{}: {:.4f}, {:.4f}'.format(dataname, auc, pr)
        print(result_str)

    print("N_attention  (AUC-ROC,RUC-PR)")
    for dataname in dataname_list:
        seed_torch(seed)
        train_x, train_y, test_x, test_y = getdataNN(dataname, 0.2)
        auc, pr = N_attention_load(dataname, device, test_x, test_y, args)

        result_str = '{}: {:.4f}, {:.4f}'.format(dataname, auc, pr)
        print(result_str)

    print("Neg_gru  (AUC-ROC,RUC-PR)")
    for dataname in dataname_list:
        seed_torch(seed)
        train_x, train_y, test_x, test_y = getdataNN(dataname, 0.2)
        auc, pr = Neg_gru_load(dataname, device, test_x, test_y, args)

        result_str = '{}: {:.4f}, {:.4f}'.format(dataname, auc, pr)
        print(result_str)

    time_taken = time.time() - st
    print("TimeTaken:",time_taken)

