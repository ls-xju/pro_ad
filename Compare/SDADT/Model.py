import torch
import torch.nn as nn
from torch.autograd import Function
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


class DCFC(nn.Module):
    """
    DCFC consists of a encoder, and an out layer
    """
    def __init__(self, input_size, num_classes, num_features, cuda):
        super(DCFC, self).__init__()
        #Number of instances  N
        self.input_size = input_size
        self.num_classes = num_classes  # Number of cluster centres
        #Dimension of embedded feature spaces
        self.num_features = num_features

        self.encoder = nn.Sequential(
            nn.Dropout(),
            nn.Linear(input_size, num_features),
            nn.ReLU(),
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_features),
        ).to(cuda)

        self.out = nn.Sequential(
            nn.Linear(num_features, 2)
        ).to(cuda)
        #
        self.clusterCenter_or = nn.Parameter(torch.zeros(num_classes, num_features))
        self.clusterCenter_gs = nn.Parameter(torch.zeros(num_classes, num_features)) 


        self.alpha = 1.0
        self.clusteringMode = False
        self.validateMode = False

        # -----model initialization----- #
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight).to(cuda)

    def setClusteringMode(self, mode):
        self.clusteringMode = mode

    def setValidateMode(self, mode):
        self.validateMode = mode

    def getDistanceToClusters_or(self, x):
        """
        obtain the distance to cluster centroids for each real instance
        Args:
            x: sample on the embedded space

        Returns: square of the euclidean distance, and the euclidean distance
        """

        xe = torch.unsqueeze(x, 1) - self.clusterCenter_or
        dist_to_centers = torch.sum(torch.mul(xe, xe), 2)
        euclidean_dist = torch.sqrt(dist_to_centers)

        return dist_to_centers, euclidean_dist

    def getDistanceToClusters_gs(self, x):
        """
        obtain the distance to cluster centroids for each auxiliary instance
        Args:
            x: sample on the embedded space #

        Returns: square of the euclidean distance, and the euclidean distance

        """

        xe = torch.unsqueeze(x, 1) - self.clusterCenter_gs
        dist_to_centers = torch.sum(torch.mul(xe, xe), 2)
        euclidean_dist = torch.sqrt(dist_to_centers)

        return dist_to_centers, euclidean_dist

    def ccdistance(self):
        """
        obtain the distance to cluster centroids for each auxiliary instance
        Args:
            x: sample on the embedded space #经过encoder降维后的数据，student[64,64],列降维

        Returns: square of the euclidean distance, and the euclidean distance

        """
        cd = torch.norm(self.clusterCenter_or - self.clusterCenter_gs)

        return cd

    def forward(self, x):
        # -----feature embedding----- #
        # x = x.view(-1, self.input_size)
        x_e = self.encoder(x)

        # -----if only wants to initialize cluster centroids
        if self.clusteringMode or self.validateMode:
            return x_e

        y = self.out(x_e)
        return x_e, y

class DiffusionM(nn.Module):
    def __init__(self, args, train_x):
        super(DiffusionM, self).__init__()
        self.features = train_x.shape[1]
        self.num_steps = args.num_steps
        self.num_units = args.units

        self.linears = nn.ModuleList(
            [
                nn.Linear(self.features, self.num_units),# 0
                nn.ReLU(),# 1
                nn.Linear(self.num_units, self.num_units),# 2
                nn.ReLU(),# 3
                nn.Linear(self.num_units, self.num_units),# 4
                nn.ReLU(),# 5
                nn.Linear(self.num_units, self.features),# 6
            ]
        ).to(args.device)

        # embedding the num steps
        self.step_embeddings = nn.ModuleList(
            [
                nn.Embedding(self.num_steps, self.num_units),# 0
                nn.Embedding(self.num_steps, self.num_units),# 1
                nn.Embedding(self.num_steps, self.num_units),# 2
            ]
        ).to(args.device)

    def forward(self, x, t):
        t = torch.flatten(t)
        for idx, embedding_layer in enumerate(self.step_embeddings):
            t_embedding = embedding_layer(t)
            x = self.linears[2 * idx](x)
            x += t_embedding
            x = self.linears[2 * idx + 1](x)

        x = self.linears[-1](x)

        return x
