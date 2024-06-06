import argparse
import numpy as np
import Tools.utils


def x_t(x_0, t, args):

    """It is possible to obtain x[t] at any moment t based on x[0]"""
    noise = np.random.randn(*x_0.shape)
    alphas_t = args.alphas_bar_sqrt[t]
    alphas_1_m_t = args.one_minus_alphas_bar_sqrt[t]
    # Add noise to x[0]
    return (alphas_t * x_0 + alphas_1_m_t * noise), noise

def diffusion(steps, schedule_name, train_x, train_y):
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.num_steps = steps
    args.betas = get_named_beta_schedule(schedule_name, args.num_steps)

    args.alphas = 1 - args.betas
    args.alphas_bar = np.cumprod(args.alphas, 0)
    args.alphas_bar_sqrt = np.sqrt(args.alphas_bar)
    args.one_minus_alphas_bar_sqrt = np.sqrt(1 - args.alphas_bar)

    # Generate a NumPy array of random integers from 0 to T.
    t = np.random.randint(0, args.num_steps, size=(train_x.shape[0],)).reshape(-1, 1)

    # Constructing inputs to the model
    neg_x, noise = x_t(train_x, t, args)

    neg_y = np.ones(len(neg_x))


    train_x = np.vstack((neg_x, train_x))
    train_y = np.concatenate((neg_y, train_y))

    train_x, train_y = Tools.utils.shuffle(train_x, train_y)

    return train_x, train_y,neg_x

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
        scale = 1e-6
        beta_start = scale * 1e-3
        beta_end = scale * 2e-2
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


