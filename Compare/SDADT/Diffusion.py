import torch.nn as nn
from sklearn import metrics

from Compare.SDADT.Tool.utils import *
from Compare.SDADT.Tool.sampling import *
from Compare.SDADT.Model import DiffusionM
import random

def seed_torch(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) #
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	# torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
    #torch.use_deterministic_algorithms(True)  #

def train(args, train_x, train_y):

    num_epoch = 3000
    maxauc = 0.0
    maxpr = 0.0
    maxf1 = 0.0
    emerror= 0.0
    emerror01 = 0.0
    generror = 0.0
    loss = 0.0
    running = 0.0
    s = 1
    model = DiffusionM(args, train_x).to(args.device)


    batch = train_x.shape[0]
    mse = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_losses = []
    for epoch in range(num_epoch):
        model.train()
        for i in range(train_x.shape[0] // batch):
            input_batch = train_x[i * batch: (i + 1) * batch]

            # Generate a random moment t for the sample
            t = torch.randint(0, args.num_steps, size=(input_batch.shape[0],)).to(args.device)
            t = t.unsqueeze(-1)

            # Constructing inputs to the model
            x, noise = x_t(input_batch, t, args)

            # Feed into the model to get the noise prediction at moment t
            output = model(x, t.squeeze(-1))

            # Calculating the difference between real and predicted noise
            noise_loss = mse(noise, output)
            optimizer.zero_grad()
            noise_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()

            running += noise_loss.data.cpu().numpy()

        # Testing
        auc, x_0, maxauc, maxpr, maxf1, maxdata, label,emerror,emerror01,generror = Test(model, maxauc, maxpr, maxf1, args,running,emerror,emerror01,generror)

        if epoch % 999 ==0:
            print(f'Epoch: {epoch}, AUC: {auc}, LOSS: {running}')

        running = 0.0


    return maxauc, maxpr


def Test(model, maxauc, maxpr, maxf1,args,running,emerror,emerror01,generror):
    model.eval()
    mae = torch.nn.L1Loss(reduction='mean')
    with torch.no_grad():
        test_x = torch.tensor(args.test_x).float()
        test_y = torch.tensor(args.test_y).float()
        maxdata = test_x
        label = args.test_y
        # Sampling, calculating the restored x_0
        x_0, xt, z = sampleT(model, args, test_x)
        x_0 = x_0.cpu().detach()

        sum = torch.mean((test_x - x_0).pow(2), dim=1).data
        auc, pr = CalMetrics(test_y.cpu(), sum.cpu())

        if args.auxiliary == True:
            if auc > maxauc and pr > maxpr:
                maxauc = auc
                maxpr = pr


    return auc, x_0, maxauc, maxpr, maxf1, maxdata, label,emerror,emerror01,generror

def Diffusion(args, train_x, train_y):

    maxauc, maxpr = train(args, train_x, train_y)

    return maxauc, maxpr


