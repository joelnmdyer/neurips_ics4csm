import argparse
import numpy as np
import os
import time
import torch
from torch.optim import Adam

from neurips_ics4csm import training_ode_grid_bwsg
from neurips_ics4csm.utils_bwsg import (build_surrogate_compute_metric, collect_data, collect_metrics, create_instantiate_emission,
                                     create_nll, create_instantiate_bwsgrnn, generate_networks, generate_dists, instantiate_model,
                                     mse_loss)


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
    L = 20
    # Total number of agents
    N = L ** 2
    # Total number of time steps
    T = 50

    # Observational train
    xs_train_obs_ = torch.load("../experiments/bwsg/aggregate_ts_OBS.pt")
    this_train_obs_ = torch.load("../experiments/bwsg/par_intervention_OBS.pt")

    # Observational test
    xs_test_obs = torch.load("../experiments/bwsg/aggregate_ts_OBS_TEST.pt")
    this_test_obs = torch.load("../experiments/bwsg/par_intervention_OBS_TEST.pt")

    # Interventional train
    xs_train_int_ = torch.load("../experiments/bwsg/aggregate_ts_INT.pt")
    this_train_int_ = torch.load("../experiments/bwsg/par_intervention_INT.pt")

    # Interventional test
    xs_test_int = torch.load("../experiments/bwsg/aggregate_ts_INT_TEST.pt")
    this_test_int = torch.load("../experiments/bwsg/par_intervention_INT_TEST.pt")

    # TODO: Tidy this up
    zeros = torch.tensor([[0.]]).repeat(xs_train_obs_.shape[0],1).double()
    scheduler = None
    LR = 5e-3
    BS = 20
    MENI = 20
    ME = 20

    for i in seeds:
        xs_train_obs = torch.roll(xs_train_obs_, (i+1)*200, 0)
        this_train_obs = torch.roll(this_train_obs_, (i+1)*200, 0)
        xs_train_int = torch.roll(xs_train_int_, (i+1)*200, 0)
        this_train_int = torch.roll(this_train_int_, (i+1)*200, 0)

        instantiate_emission = create_instantiate_emission(N, kind=args.family)
        negative_log_likelihood = create_nll(instantiate_emission, N)
        obs_rnn_net, obs_omega = generate_networks(kind=args.family, seed=i)
        int_rnn_net, int_omega = generate_networks(kind=args.family, seed=i)
        if args.family == "lrnn": 
            obs_model = create_instantiate_bwsgrnn(obs_rnn_net)
            int_model = create_instantiate_bwsgrnn(int_rnn_net)
            obs_optimiser = Adam(list(obs_rnn_net.parameters()) +
                             list(obs_omega.parameters()),
                             lr=LR)
            int_optimiser = Adam(list(int_rnn_net.parameters()) +
                             list(int_omega.parameters()),
                             lr=LR)
        else:
            test_ts = torch.linspace(0,T,T+1)
            # Instantiate model
            obs_model = instantiate_model(test_ts)
            int_model = instantiate_model(test_ts)
            obs_optimiser = Adam(list(obs_rnn_net.parameters()) +
                             list(obs_omega.parameters()) + 
                             list(obs_model.parameters()),
                             lr=LR)
            int_optimiser = Adam(list(int_rnn_net.parameters()) +
                             list(int_omega.parameters()) + 
                             list(int_model.parameters()),
                             lr=LR)


        print("Training " + args.family + "...")
        try:
            best_obs_omega = torch.load(os.path.join(dirname, "best_obs_{0}_omega_{1}.pt".format(args.family, i)))
            best_obs_rnn_net = torch.load(os.path.join(dirname, "best_obs_{0}_rnn_net_{1}.pt".format(args.family, i)))
            best_obs_model = torch.load(os.path.join(dirname, "best_obs_{0}_model_{1}.pt".format(args.family, i)))
            start_time_obs = 0
            end_time_obs = 0
        except:
            start_time_obs = time.process_time()
            best_obs_omega, best_obs_rnn_net, best_obs_model, loss_hist = training_ode_grid_bwsg.train_epi(obs_omega.double(),
                                                                                                   torch.nn.Identity() if args.family == "lrnn" else obs_rnn_net.double(),
                                                                                                   zeros,
                                                                                                   xs_train_obs.double(),
                                                                                                   this_train_obs.double(),
                                                                                                   obs_model if args.family == "lrnn" else instantiate_model,
                                                                                                   negative_log_likelihood,
                                                                                                   obs_optimiser,
                                                                                                   scheduler=scheduler,
                                                                                                   batch_size=BS,
                                                                                                   max_epochs_no_improve=MENI,
                                                                                                   max_epochs=ME,
                                                                                                   notebook=False,
                                                                                                   model=None if args.family == "lrnn" else obs_model,
                                                                                                   full_node=True)
            end_time_obs = time.process_time()

            torch.save(best_obs_omega, os.path.join(dirname, "best_obs_{0}_omega_{1}.pt".format(args.family, i)))
            torch.save(best_obs_rnn_net, os.path.join(dirname, "best_obs_{0}_rnn_net_{1}.pt".format(args.family, i)))   
            torch.save(best_obs_model, os.path.join(dirname, "best_obs_{0}_model_{1}.pt".format(args.family, i)))              

        ## Interventional
        try:
            best_int_omega = torch.load(os.path.join(dirname, "best_int_omega_{0}.pt".format(i)))
            best_int_rnn_net = torch.load(os.path.join(dirname, "best_int_rnn_net_{0}.pt".format(i)))
            best_int_model = torch.load(os.path.join(dirname, "best_int_model_{0}.pt".format(i)))
            start_time_int = 0
            end_time_int = 0
        except:
            start_time_int = time.process_time()
            best_int_omega, best_int_rnn_net, best_int_model, loss_hist = training_ode_grid_bwsg.train_epi(int_omega.double(),
                                                                                                   torch.nn.Identity() if args.family == "lrnn" else int_rnn_net.double(),
                                                                                                   zeros,
                                                                                                   xs_train_int.double(),
                                                                                                   this_train_int.double(),
                                                                                                   int_model if args.family == "lrnn" else instantiate_model,
                                                                                                   negative_log_likelihood,
                                                                                                   int_optimiser,
                                                                                                   scheduler=scheduler,
                                                                                                   batch_size=BS,
                                                                                                   max_epochs_no_improve=MENI,
                                                                                                   max_epochs=ME,
                                                                                                   notebook=False,
                                                                                                   model=None if args.family == "lrnn" else int_model,
                                                                                                   full_node=True)
            end_time_int = time.process_time()

            torch.save(best_int_omega, os.path.join(dirname, "best_int_{0}_omega_{1}.pt".format(args.family, i)))
            torch.save(best_int_rnn_net, os.path.join(dirname, "best_int_{0}_rnn_net_{1}.pt".format(args.family, i)))
            torch.save(best_int_model, os.path.join(dirname, "best_int_{0}_model_{1}.pt".format(args.family, i)))     

        ###
        # test
        ###

        (test_obs_msesstoch_obs, 
         test_obs_neg_log_probs_obs, 
         test_obs_msesstoch_int, 
         test_obs_neg_log_probs_int) = collect_metrics(xs_test_obs, 
                                                               this_test_obs, 
                                                               instantiate_emission, 
                                                               best_obs_omega, 
                                                               best_int_omega, 
                                                               best_obs_model, 
                                                               torch.nn.Identity() if args.family == "lrnn" else best_obs_rnn_net, 
                                                               torch.nn.Identity() if args.family == "lrnn" else best_int_rnn_net, 
                                                               N,
                                                               model2=best_int_model)

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
         test_int_neg_log_probs_int) = collect_metrics(xs_test_int, 
                                                           this_test_int, 
                                                           instantiate_emission, 
                                                           best_obs_omega, 
                                                           best_int_omega, 
                                                           best_obs_model, 
                                                           torch.nn.Identity() if args.family == "lrnn" else best_obs_rnn_net, 
                                                           torch.nn.Identity() if args.family == "lrnn" else best_int_rnn_net, 
                                                           N,
                                                           model2=best_int_model)

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

        with open(os.path.join(dirname, "config.file"), "a") as fh:
            fh.write("Obs train time = {0}\nInt train time = {1}".format(end_time_obs - start_time_obs, end_time_int - start_time_int))
