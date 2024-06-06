from Tools.utils import *
import time
import argparse

from Compare.PyOD import compare
from binary.Bprototype import BPROT
from Absolution.N_diffusion import N_diffusion
from Absolution.Only_diffusion import Only_diffusion
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


if __name__ == '__main__':
    st = time.time()
    #tips  0(compare methods)  1(PHAD)
    compare_alg = 1
    # tips compare methods： ECOD DIF RCA DTPM SDAD NeuTraL ICL SLAD LUNAR
    c_alg = 'ECOD'

    for dataname in dataname_list:
        seed_torch(seed)
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

        train_x, train_y, test_x, test_y = getdataNN(dataname, 0.2)

        if compare_alg == 0:
            name, maxauc, maxpr, timetaken = compare(c_alg, dataname, train_x, train_y, test_x, test_y, seed, args.k_nebor)
        else:
            #  our proposed PHAD: BPROT
            #  tips absolution
            #  N_diffusion Only_diffusion
            #  Graph_w_default  Graph_w_ed  Graph_w_mad  Hypergraph_w_default Hypergraph_w_ed
            #  N_prototype N_discriminator N_fusion N_attention Neg_gru
            maxauc, maxpr, timetaken = BPROT(dataname, device, train_x, train_y, test_x, test_y, args)

        result_str = '{}: {:.4f}, {:.4f}'.format(dataname, maxauc, maxpr)
        print(result_str)

    time_taken = time.time() - st
    print("TimeTaken:",time_taken)

