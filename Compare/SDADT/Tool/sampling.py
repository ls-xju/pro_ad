from Compare.SDADT.Tool.gaussian_diffusion import *

def sample(model, args, train_x):
    model.eval()
    with torch.no_grad():
        """From x[T] to x[T-1]、x[T-2]|...x[0]"""
        #Calculate xt, using the original xt
        T = torch.tensor([args.num_steps-1]).to(args.device)
        xt, z = x_t(train_x, T, args)
        for i in reversed(range(args.num_steps)):
            t = torch.tensor([i]).to(args.device)
            predicted_noise = model(xt, t)

            alphat = args.alphas[t].to(args.device)
            one_minus_alphat_bar_sqrt = args.one_minus_alphas_bar_sqrt[t].to(args.device)
            sigmat = args.betas[t].to(args.device)

            if i > 0:
                xt = (1 / alphat.sqrt()) * (xt - (sigmat / one_minus_alphat_bar_sqrt) * predicted_noise) + sigmat * z
            else:
                xt = (1 / alphat.sqrt()) * (xt - (sigmat / one_minus_alphat_bar_sqrt) * predicted_noise)

        x0 = xt
    return x0


def sampleT(model, args, test_x):
    model.eval()
    with torch.no_grad():
        """From x[T] to x[T-1]、x[T-2]|...x[0]"""
        #Calculate xt, using the original xt
        # tips 10, 100, 200, 300, ...999  args.num_steps
        T = torch.tensor([args.lamda-1]).to(args.device)

        xt, z = x_t(test_x, T, args)
        for i in reversed(range(args.lamda)):

            t = torch.tensor([i]).to(args.device)

            predicted_noise = model(xt, t)

            alphat = args.alphas[t].to(args.device)
            one_minus_alphat_bar_sqrt = args.one_minus_alphas_bar_sqrt[t].to(args.device)
            sigmat = args.betas[t].to(args.device)

            if i > 0:
                xt = (1 / alphat.sqrt()) * (xt - (sigmat / one_minus_alphat_bar_sqrt) * predicted_noise) + sigmat * z
            else:
                xt = (1 / alphat.sqrt()) * (xt - (sigmat / one_minus_alphat_bar_sqrt) * predicted_noise)

        x0 = xt
    return x0, xt, z