import torch
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn import manifold

def x_t(x_0, t, args):

    """It is possible to obtain x[t] at any moment t based on x[0]"""
    x_0 = x_0.to(args.device)
    noise = torch.randn_like(x_0).to(args.device)
    alphas_t = args.alphas_bar_sqrt[t].to(args.device)
    alphas_1_m_t = args.one_minus_alphas_bar_sqrt[t].to(args.device)
    # Add noise to x[0]
    return (alphas_t * x_0 + alphas_1_m_t * noise), noise

def show_sample(data,dimensions, rs):
    X = data
    pca = PCA(n_components=dimensions)
    pca_result = pca.fit_transform(X)
    tsne = manifold.TSNE(n_components=dimensions, perplexity=30, n_iter=1000, learning_rate='auto', random_state=rs, init=pca_result)
    X_tsne = tsne.fit_transform(X)

    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)

    return X_norm