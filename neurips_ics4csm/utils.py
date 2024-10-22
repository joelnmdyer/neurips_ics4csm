import torch
import numpy as np
import os
import time

from neurips_ics4csm.models import sirs_ode
try:
    from neurips_ics4csm.models import sirs_spatial
except ModuleNotFoundError:
    pass
from neurips_ics4csm.networks import MLP, RNN, count_pars

class Omega(torch.nn.Module):

    def __init__(self, ffn):

        super().__init__()
        self.ffn = ffn

    def forward(self, grids, pars):
        x = pars
        x = self.ffn(x)
        return x


# Instantiates the ODE
def instantiate_model(t):
    model = sirs_ode.SIRSODE_naive_int(t)
    return model


# Instantiates the RNN
def create_instantiate_sirsrnn(net):
    def instantiate_sirsrnn(t):
        model = sirs_ode.SIRSRNN(t, net)
        return model
    return instantiate_sirsrnn
    
    
# Helper function for initialising networks
def generate_networks(kind='lodernn', seed=0):
    torch.manual_seed(seed)
    if kind in ['lodernn', 'lrnn']:
        # This is to map from hidden state of RNN to logits
        mlp_net = MLP(input_dim=32, output_dim=3, hidden_dims=[32, 64, 32, 16], final_nonlinearity=torch.nn.Identity()).double()
        # This is to map from ODE output to hidden state of RNN
        rnn_net = RNN(input_size=3, final_ff=mlp_net, flavour='gru').double()
        # This maps from ABM parameters to parameters of ODE
        mlp = MLP(input_dim=3, output_dim=3, hidden_dims=[32, 64, 32], final_nonlinearity=torch.nn.Sigmoid()).double()
    elif kind in ['lodernn_small', 'lrnn_small']:
        # This is to map from hidden state of RNN to logits
        mlp_net = MLP(input_dim=32, output_dim=3, hidden_dims=[32, 64, 32], final_nonlinearity=torch.nn.Identity()).double()
        # This is to map from ODE output to hidden state of RNN
        rnn_net = RNN(input_size=3, final_ff=mlp_net, flavour='gru').double()
        # This maps from ABM parameters to parameters of ODE
        mlp = MLP(input_dim=3, output_dim=3, hidden_dims=[32, 32], final_nonlinearity=torch.nn.Sigmoid()).double()
    elif kind == 'lode':
        rnn_net = torch.nn.Identity()
        # This maps from ABM parameters to parameters of ODE
        mlp = MLP(input_dim=3, output_dim=3, hidden_dims=[32, 64, 64, 64, 32], final_nonlinearity=torch.nn.Sigmoid()).double()
    elif kind == 'lode_small':
        rnn_net = torch.nn.Identity()
        # This maps from ABM parameters to parameters of ODE
        mlp = MLP(input_dim=3, output_dim=3, hidden_dims=[32, 64, 64, 64, 32], final_nonlinearity=torch.nn.Sigmoid()).double()
    omega = Omega(mlp)
    print("Total trainable parameters =", 
          count_pars(omega) + 
          count_pars(rnn_net)
          )
    return rnn_net, omega


# For defining the emission distribution
def create_instantiate_emission(N, kind='lodernn'):
    def instantiate_emission(e_pars):
        if kind in ['lodernn', 'lrnn']:
            return torch.distributions.multinomial.Multinomial(total_count=N, logits=e_pars)
        elif kind == 'lode': 
            return torch.distributions.multinomial.Multinomial(total_count=N, probs=e_pars)
    return instantiate_emission


def create_nll(instantiate_emission, N):
    # This is for computing negative log likelihood of observation x from ODE output y
    def negative_log_likelihood(x, y, rnn_net):
        """
        x is (normalised) S, I, R from ABM, y from ODE, rnn_net is
        """
        x, y = x[0], y[0]
        x = (x * N).int()
        emission_params = rnn_net(y)
        emissions = [instantiate_emission(e_pars) for e_pars in emission_params]
        lps = torch.stack([emissions[j].log_prob(x[j]) for j in range(x.shape[0])])
        if lps.isnan().any():
            print("nan")
        return -torch.sum(lps)
    return negative_log_likelihood


def generate_dists(instantiate_emission, omega, params, model, y0, i, rnn_net):

    new_params = omega(None, params.unsqueeze(0).double())[0]
    y_mac = model(torch.cat((y0, torch.tensor([i])), dim=-1), new_params)[:, :-1]
    e_pars = rnn_net(y_mac.double())
    e_dists = [instantiate_emission(e_par) for e_par in e_pars]
    return e_dists


def build_surrogate_compute_metric(instantiate_emission, omega, params, model, y0, i, rnn_net, x, N, T):

    e_dists = generate_dists(instantiate_emission, omega, params, model, y0, i, rnn_net)
    y_mac_stoch = torch.cat([e_d.sample((1,)) for e_d in e_dists])
    #assert y_mac_stoch.shape[0] == 51, y_mac_stoch.shape[1] == 3
    y_mac_stoch_mean = torch.stack([e_d.mean for e_d in e_dists])
    # Stoch MSE
    this_stoch_mse_loss = mse_loss(x, y_mac_stoch / N)
    # MSE
    #this_mse_loss = mse_loss(x, y_mac_stoch_mean / N)
    # MSE infections
    # this_mseinf_loss = mse_loss(x[:, 1], y_mac_stoch[:, 1] / N)
    #max_inf_idx = torch.argmax(x[:, 1])
    #this_mseinf_loss = torch.abs(x[max_inf_idx, 1] - y_mac_stoch[max_inf_idx, 1] / N)
    # Negative log-likelihood
    ll = 0.
    for t in range(T+1):
        term = e_dists[t].log_prob((x[t] * N).int())
        ll += term
    return (this_stoch_mse_loss, #this_mse_loss, this_mseinf_loss, 
            -ll)


def collect_metrics(xs_test, this_test, instantiate_emission, obs_omega, int_omega, model, obs_rnn_net, int_rnn_net, N, model2=None):

    R = xs_test.shape[0]
    T = xs_test.shape[1] - 1
    if model2 is None:
        model2 = model

    # Test observationally trained surrogate
    test_msesstoch_obs = []
    test_neg_log_probs_obs = []
    # Test interventionally trained surrogate
    test_msesstoch_int = []
    test_neg_log_probs_int = []

    with torch.no_grad():

        for r in range(R):
            this_test_x, i0, (alpha, beta, gamma), i = xs_test[r], xs_test[r, 0, 1], this_test[r, :3], this_test[r, -1].item()
            y0 = torch.tensor([1 - i0, i0, 0.])
            params = torch.tensor([alpha, beta, gamma])
            # LNODE TRAINED OBSERVATIONALLY
            this_obs_stoch_loss, this_obs_nll = build_surrogate_compute_metric(instantiate_emission,
                                                                               obs_omega,
                                                                               params,
                                                                               model,
                                                                               y0,
                                                                               i,
                                                                               obs_rnn_net,
                                                                               this_test_x,
                                                                               N,
                                                                               T)
            test_msesstoch_obs.append(this_obs_stoch_loss)
            test_neg_log_probs_obs.append(this_obs_nll)
            # LNODE TRAINED INTERVENTIONALLY
            this_int_stoch_loss, this_int_nll = build_surrogate_compute_metric(instantiate_emission,
                                                                               int_omega,
                                                                               params,
                                                                               model2,
                                                                               y0,
                                                                               i,
                                                                               int_rnn_net,
                                                                               this_test_x,
                                                                               N,
                                                                               T)
            test_msesstoch_int.append(this_int_stoch_loss)
            test_neg_log_probs_int.append(this_int_nll)

    return test_msesstoch_obs, test_neg_log_probs_obs, test_msesstoch_int, test_neg_log_probs_int


def mse_loss(x, y):
    return torch.pow(x - y, 2).sum()


def run_spatial_intervention(params, i, i0, T, L):

    # TODO: change kwarg N to L here
    model = sirs_spatial.SIRS(n_timesteps=T, i0=i0, N=L)
    x_grid = model.initialize(params)
    true_i0 = (x_grid == 1).sum() / L**2
    x = torch.tensor([1. - true_i0, true_i0, 0.]).unsqueeze(0)
    initial_state = x_grid.clone()
    for t in range(T):
        if (i > 0) and (t >= 5 + i - 1) and (t <= 10 + i - 1):
            int_pars = params * torch.tensor([0., 1., 1.])
            x_grid = model.step(int_pars, x_grid)
        else:
            x_grid = model.step(params, x_grid)
        x_new = torch.tensor([(x_grid == 0).sum(), (x_grid == 1).sum(), (x_grid == 2).sum()]) / L**2
        x = torch.cat((x, x_new.unsqueeze(0)), dim=0)
    return initial_state, x


def collect_data(R=1000, intervene=False):

    init_states = []
    xs = []
    this = []
    
    for i in range(R):
        # Draw initial proportion of infected individuals from Uniform(0,1)
        i0 = torch.rand(1)
        # Same for the three parameters
        alpha, beta, gamma = torch.rand(1), torch.rand(1), torch.rand(1)
        params = torch.tensor([alpha, beta, gamma])
        # Start a lockdown for 5 time steps at time t = intervention if intervention != 0, else no lockdown
        if intervene:
            intervention = torch.randint(0, 5, (1,)).item()
        else:
            intervention = 0
        # Collect data points
        init_state, x = run_spatial_intervention(params, intervention, i0, T)
        init_states.append(init_state)
        xs.append(x)
        this.append(torch.cat((params, torch.tensor([intervention])), dim=-1))
    
    xs = torch.stack(xs)
    init_states = torch.stack(init_states)
    this = torch.stack(this)

    return xs, init_states, this

def check_surrogate_cost(family, dirname, obs_or_int, seed=0):

    """ 
    family in ['lrnn', 'lodernn', 'ode']
    dirname -- directory containing saved torch models
    obs_or_int in ['obs', 'int']
    seed to load saved networks/models from
    """

    R = 50  
    T = 50
    L = 5000
    N = L ** 2
    instantiate_emission = create_instantiate_emission(N, family if family in ['lrnn', 'lodernn'] else 'lode')
    if not family in ['lodernn']:
        omega = torch.load(os.path.join(dirname, "best_{1}_{0}_omega_{2}.pt".format(family, obs_or_int, seed)))
        rnn_net = torch.load(os.path.join(dirname, "best_{1}_{0}_rnn_net_{2}.pt".format(family, obs_or_int, seed)))
    else:
        omega = torch.load(os.path.join(dirname, "best_{1}_omega_{2}.pt".format(family, obs_or_int, seed)))
        rnn_net = torch.load(os.path.join(dirname, "best_{1}_rnn_net_{2}.pt".format(family, obs_or_int, seed)))
    if not family in ['ode', 'lodernn']:
        model = torch.load(os.path.join(dirname, "best_{1}_{0}_model_{2}.pt".format(family, obs_or_int, seed)))
    else:
        model = instantiate_model(torch.linspace(0, T, T+1))

    i0 = 0.1
    y0 = torch.tensor([1 - i0, i0, 0.]).double()

    params = torch.rand((R, 3))

    start_surr = time.process_time()
    for r in range(R):
        e_dists = generate_dists(instantiate_emission, omega, params[r], model, y0, r % 5, rnn_net)
        y_mac_stoch = torch.cat([e_d.sample((1,)) for e_d in e_dists])
    end_surr = time.process_time()
    print("Average surrogate process time = {0}".format((end_surr - start_surr)/R))

    start_model = time.process_time()
    for r in range(R):
        _ = run_spatial_intervention(params[r], r % 5, i0, T, L)
    end_model = time.process_time()
    print("Average abm process time = {0}".format((end_model - start_model)/R))

def check_preferred_action(family, dirname, obs_or_int, seed=0):

    """ 
    family in ['lrnn', 'lodernn', 'ode']
    dirname -- directory containing saved torch models
    obs_or_int in ['obs', 'int']
    seed to load saved networks/models from
    """

    R = 1000
    T = 50
    L = 50
    N = L ** 2
    instantiate_emission = create_instantiate_emission(N, family if family in ['lrnn', 'lodernn'] else 'lode')
    if not family in ['lodernn']:
        omega = torch.load(os.path.join(dirname, "best_{1}_{0}_omega_{2}.pt".format(family, obs_or_int, seed)))
        rnn_net = torch.load(os.path.join(dirname, "best_{1}_{0}_rnn_net_{2}.pt".format(family, obs_or_int, seed)))
    else:
        omega = torch.load(os.path.join(dirname, "best_{1}_omega_{2}.pt".format(family, obs_or_int, seed)))
        rnn_net = torch.load(os.path.join(dirname, "best_{1}_rnn_net_{2}.pt".format(family, obs_or_int, seed)))
    if not family in ['ode', 'lodernn']:
        model = torch.load(os.path.join(dirname, "best_{1}_{0}_model_{2}.pt".format(family, obs_or_int, seed)))
    else:
        model = instantiate_model(torch.linspace(0, T, T+1))

    params = torch.rand((R, 4))

    means_sum = torch.tensor([0., 0., 0., 0., 0.])
    for r in range(R):
        i0 = params[r,0]
        y0 = torch.tensor([1 - i0, i0, 0.]).double()
        e_dists = generate_dists(instantiate_emission, omega, params[r,1:], model, y0, r % 5, rnn_net)
        means_sum[r % 5] += sum([e_dist.mean[1] for e_dist in e_dists]) / len(e_dists)
        #y_mac_stoch = torch.cat([e_d.sample((1,)) for e_d in e_dists])
    print("Average surrogate mean infections for each intervention = {0}".format(means_sum / (R/5.) / N))
