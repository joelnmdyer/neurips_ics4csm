import argparse
import numpy as np
import os
import time
import torch
from torch.optim import Adam

from neurips_ics4csm import training_ode_grid_sirs
from neurips_ics4csm.utils import (build_surrogate_compute_metric, collect_data, collect_metrics, create_instantiate_emission,
                                     create_nll, create_instantiate_sirsrnn, generate_networks, generate_dists, instantiate_model,
                                     mse_loss, run_spatial_intervention)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--family", help="Flag to indicate which method to run. Options: lode, lodernn, lrnn")
    parser.add_argument("--dirname", type=str, help="Firectory for loading and saving data from and to", default="")
    parser.add_argument("--seed", type=int, nargs='*', help="Which seeds to run")
    args = parser.parse_args()
    if args.dirname == "":
        dirname = os.path.join("./", str(time.time()))
    else:
        dirname = args.dirname
    try:
        os.makedirs(dirname)
    except FileExistsError:
        # directory already exists
        pass
    with open(os.path.join(dirname, "config.file"), "w") as fh:
        fh.write(args.family)

    if args.seed is None:
        seeds = range(5)
    elif len(args.seed) > 0:
        seeds = args.seed
    else:
        seeds = range(5)
    print("Running seeds", seeds)

    # Number of grid cells in horiz and vertical directions
    L = 50
    # Total number of agents
    N = L ** 2
    # Total number of time steps
    T = 50

    # Observational train
    xs_train_obs_ = torch.load("../experiments/sirs_ode_spatial/aggregate_ts_OBS.pt")
    this_train_obs_ = torch.load("../experiments/sirs_ode_spatial/par_intervention_OBS.pt")

    # Observational test
    xs_test_obs = torch.load("../experiments/sirs_ode_spatial/aggregate_ts_OBS_TEST.pt")
    this_test_obs = torch.load("../experiments/sirs_ode_spatial/par_intervention_OBS_TEST.pt")

    # Interventional train
    xs_train_int_ = torch.load("../experiments/sirs_ode_spatial/aggregate_ts_INT.pt")
    this_train_int_ = torch.load("../experiments/sirs_ode_spatial/par_intervention_INT.pt")

    # Interventional test
    xs_test_int = torch.load("../experiments/sirs_ode_spatial/aggregate_ts_INT_TEST.pt")
    this_test_int = torch.load("../experiments/sirs_ode_spatial/par_intervention_INT_TEST.pt")

    # TODO: Tidy this up
    zeros = torch.tensor([[0.]]).repeat(xs_train_obs_.shape[0],1).double()
    scheduler = None

    for i in seeds:
        xs_train_obs = torch.roll(xs_train_obs_, (i+1)*200, 0)
        this_train_obs = torch.roll(this_train_obs_, (i+1)*200, 0)
        xs_train_int = torch.roll(xs_train_int_, (i+1)*200, 0)
        this_train_int = torch.roll(this_train_int_, (i+1)*200, 0)

        instantiate_emission = create_instantiate_emission(N, kind=args.family)
        negative_log_likelihood = create_nll(instantiate_emission, N)
        # Observational
        obs_rnn_net, obs_omega = generate_networks(kind=args.family, seed=i)
        if args.family == "lrnn":
            obs_inst_mod = create_instantiate_sirsrnn(obs_rnn_net)
        else:
            obs_inst_mod = instantiate_model
        obs_optimiser = Adam(list(obs_rnn_net.parameters()) +
                         list(obs_omega.parameters()),
                         lr=1e-2)
        # Interventional
        int_rnn_net, int_omega = generate_networks(kind=args.family, seed=i)
        if args.family == "lrnn":
            int_inst_mod = create_instantiate_sirsrnn(int_rnn_net)
        else:
            int_inst_mod = instantiate_model
        int_optimiser = Adam(list(int_rnn_net.parameters()) +
                         list(int_omega.parameters()),
                         lr=1e-2)

        # Observational
        print("Training observational {0}".format(args.family))
        try:
            best_obs_omega = torch.load(os.path.join(dirname, "best_obs_{0}_omega_{1}.pt".format(args.family, i)))
            best_obs_rnn_net = torch.load(os.path.join(dirname, "best_obs_{0}_rnn_net_{1}.pt".format(args.family, i)))
            if args.family == "lrnn":
                best_obs_lrnn_model = torch.load(os.path.join(dirname, "best_obs_lrnn_model_{0}.pt".format(i)))
        except:
            if args.family == "lrnn":
                best_obs_omega, best_obs_rnn_net, best_obs_lrnn_model, loss_hist = training_ode_grid_sirs.train_epi(obs_omega.double(),
                                                                                                                      torch.nn.Identity(),
                                                                                                                      zeros,
                                                                                                                      xs_train_obs.double(),
                                                                                                                      this_train_obs.double(),
                                                                                                                      obs_inst_mod,
                                                                                                                      negative_log_likelihood,
                                                                                                                      obs_optimiser,
                                                                                                                      scheduler=scheduler,
                                                                                                                      batch_size=50,
                                                                                                                      max_epochs_no_improve=20,
                                                                                                                      full_node=True,
                                                                                                                      notebook=False)
                torch.save(best_obs_lrnn_model, os.path.join(dirname, "best_obs_lrnn_model_{0}.pt".format(i)))
            else:
                best_obs_omega, best_obs_rnn_net, loss_hist = training_ode_grid_sirs.train_epi(obs_omega.double(),
                                                                                                   obs_rnn_net.double(),
                                                                                                   zeros,
                                                                                                   xs_train_obs.double(),
                                                                                                   this_train_obs.double(),
                                                                                                   obs_inst_mod,
                                                                                                   negative_log_likelihood,
                                                                                                   obs_optimiser,
                                                                                                   scheduler=scheduler,
                                                                                                   batch_size=50,
                                                                                                   max_epochs_no_improve=20,
                                                                                                   notebook=False)

            torch.save(best_obs_omega, os.path.join(dirname, "best_obs_{0}_omega_{1}.pt".format(args.family, i)))
            torch.save(best_obs_rnn_net, os.path.join(dirname, "best_obs_{0}_rnn_net_{1}.pt".format(args.family, i)))

        print("Training interventional {0}".format(args.family))
        try:
            best_int_omega = torch.load(os.path.join(dirname, "best_int_{0}_omega_{1}.pt".format(args.family, i)))
            best_int_rnn_net = torch.load(os.path.join(dirname, "best_int_{0}_rnn_net_{1}.pt".format(args.family, i)))
            if args.family == "lrnn":
                best_int_lrnn_model = torch.load(os.path.join(dirname, "best_int_lrnn_model_{0}.pt".format(i)))
        except:
            # Interventional
            if args.family == "lrnn":
                best_int_omega, best_int_rnn_net, best_int_lrnn_model, loss_hist = training_ode_grid_sirs.train_epi(int_omega.double(),
                                                                                                                          torch.nn.Identity(),
                                                                                                                          zeros,
                                                                                                                          xs_train_int.double(),
                                                                                                                          this_train_int.double(),
                                                                                                                          int_inst_mod,
                                                                                                                          negative_log_likelihood,
                                                                                                                          int_optimiser,
                                                                                                                          scheduler=scheduler,
                                                                                                                          batch_size=50,
                                                                                                                          max_epochs_no_improve=20,
                                                                                                                          full_node=True,
                                                                                                                          notebook=False)
                torch.save(best_int_lrnn_model, os.path.join(dirname, "best_int_lrnn_model_{0}.pt".format(i)))
            else:
                best_int_omega, best_int_rnn_net, loss_hist = training_ode_grid_sirs.train_epi(int_omega.double(),
                                                                                                   int_rnn_net.double(),
                                                                                                   zeros,
                                                                                                   xs_train_int.double(),
                                                                                                   this_train_int.double(),
                                                                                                   int_inst_mod,
                                                                                                   negative_log_likelihood,
                                                                                                   int_optimiser,
                                                                                                   scheduler=scheduler,
                                                                                                   batch_size=50,
                                                                                                   max_epochs_no_improve=20,
                                                                                                   notebook=False)

            torch.save(best_int_omega, os.path.join(dirname, "best_int_{0}_omega_{1}.pt".format(args.family, i)))
            torch.save(best_int_rnn_net, os.path.join(dirname, "best_int_{0}_rnn_net_{1}.pt".format(args.family, i)))

        ###
        # test
        ###

        # Instantiate model – TODO check if this is the same for lrnn
        if args.family == "lrnn":
            obs_model = best_obs_lrnn_model#.double()
            int_model = best_int_lrnn_model#.double()
            obs_rnn_net = torch.nn.Identity()
            int_rnn_net = torch.nn.Identity()
        else:
            test_ts = torch.linspace(0,T,T+1)
            obs_model = instantiate_model(test_ts)
            int_model = obs_model
            obs_rnn_net = best_obs_rnn_net
            int_rnn_net = best_int_rnn_net

        (test_obs_msesstoch_obs, 
         test_obs_neg_log_probs_obs, 
         test_obs_msesstoch_int, 
         test_obs_neg_log_probs_int) = collect_metrics(xs_test_obs.double(), 
                                                           this_test_obs.double(), 
                                                           instantiate_emission, 
                                                           best_obs_omega.double(), 
                                                           best_int_omega.double(), 
                                                           obs_model.double(), 
                                                           obs_rnn_net.double(), 
                                                           int_rnn_net.double(), 
                                                           N,
                                                           model2=int_model.double())

        R = len(test_obs_msesstoch_obs)
        amse_obs_obs = sum(test_obs_msesstoch_obs) / R
        anll_obs_obs = sum(test_obs_neg_log_probs_obs) / R
        amse_obs_int = sum(test_obs_msesstoch_int) / R
        anll_obs_int = sum(test_obs_neg_log_probs_int) / R
        print("{0} (O): AMSE =".format(args.family), amse_obs_obs, "; ANLL =", anll_obs_obs)
        print("{0} (I): AMSE =".format(args.family), amse_obs_int, "; ANLL =", anll_obs_int)
        print()

        with open(os.path.join(dirname, "{0}_obs_test_int_train.csv".format(args.family)), "a") as fh:
            fh.write("{0}, {1}\n".format(amse_obs_int, anll_obs_int))
        with open(os.path.join(dirname, "{0}_obs_test_obs_train.csv".format(args.family)), "a") as fh:
            fh.write("{0}, {1}\n".format(amse_obs_obs, anll_obs_obs))

        (test_int_msesstoch_obs, 
         test_int_neg_log_probs_obs, 
         test_int_msesstoch_int, 
         test_int_neg_log_probs_int) = collect_metrics(xs_test_int.double(), 
                                                           this_test_int.double(), 
                                                           instantiate_emission, 
                                                           best_obs_omega.double(), 
                                                           best_int_omega.double(), 
                                                           obs_model.double(), 
                                                           obs_rnn_net.double(), 
                                                           int_rnn_net.double(), 
                                                           N,
                                                           model2=int_model.double())

        R = len(test_int_msesstoch_obs)
        amse_int_obs = sum(test_int_msesstoch_obs) / R
        anll_int_obs = sum(test_int_neg_log_probs_obs) / R
        amse_int_int = sum(test_int_msesstoch_int) / R
        anll_int_int = sum(test_int_neg_log_probs_int) / R
        print("{0} (O): AMSE =".format(args.family), amse_int_obs, "; ANLL =", anll_int_obs)
        print("{0} (I): AMSE =".format(args.family), amse_int_int, "; ANLL =", anll_int_int)
        print()

        with open(os.path.join(dirname, "{0}_int_test_int_train.csv".format(args.family)), "a") as fh:
            fh.write("{0}, {1}\n".format(amse_int_int, anll_int_int))
        with open(os.path.join(dirname, "{0}_int_test_obs_train.csv".format(args.family)), "a") as fh:
            fh.write("{0}, {1}\n".format(amse_int_obs, anll_int_obs))
