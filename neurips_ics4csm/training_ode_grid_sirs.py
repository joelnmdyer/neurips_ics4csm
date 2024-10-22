import copy
import logging
import torch
from tqdm.notebook import trange
from tqdm import tqdm as normal_trange

logger = logging.getLogger("training")


def simulate_and_intervene(model, macro_th, i, y0, T):

    y_initial = torch.cat((y0, torch.tensor([i])), dim=0)
    y = model(y_initial, macro_th)

    return y[..., :-1]

def produce_loss(omega, calib, states, aggs, ths, i_s, model, loss_fn):

    macro_ths = omega(states, ths)

    loss = 0.

    for j in range(states.shape[0]):
        macro_th = macro_ths[j]
        i = i_s[j]
        x = aggs[j:j+1]
        y = simulate_and_intervene(model, macro_th, i.item(), x[0,0], aggs.shape[1] - 1).unsqueeze(0)
        # Remove R channel because it is redundant for finite population
        assert (y.shape[0] == 1) and (y.shape[1] == x.shape[1]) and (y.shape[2] == x.shape[2]), print(y.shape, x.shape)

        # loss_fn should take in micromodel draw x, macro ode output y, and network that maps from y to emission distribution parameters
        loss += loss_fn(x, y, calib)

    return loss / states.shape[0]


def train_epi(omega, calib, abm_states, abm_agg_ts, abm_thi,
              ode, loss_fn, optimiser, scheduler=None, batch_size=50,
              max_epochs=1000, max_epochs_no_improve=30, prop_val=0.2, full_node=False,
              notebook=True):

    """
    abm_states: torch.tensor of shape (R, N, N, 3) containing grid states for spatial SIR

    abm_agg_ts: the aggregate state of the ABM forward simulation. Shape (R, T, C_), where T is number of time steps
                and C_ is number of channels to aggregate/macro time series. Assume this isn't learnable

    abm_thi:    (R, D + 1) with first D columns representing parameters and last containing label of intervention

    future extensions: include map from lower to upper intervention as input arg.
    """

    R = abm_states.size(0)
    N_VAL = int(R * prop_val)
    N_TRAIN = R - N_VAL

    train_abm_states = abm_states[:N_TRAIN]
    val_abm_states = abm_states[N_TRAIN:]
    train_abm_agg_ts = abm_agg_ts[:N_TRAIN]
    val_abm_agg_ts = abm_agg_ts[N_TRAIN:]
    train_abm_thi = abm_thi[:N_TRAIN]
    val_abm_thi = abm_thi[N_TRAIN:]

    loss_hist = []

    best_val_loss = float('inf')
    val_loss = float('inf')

    iterator = normal_trange(range(max_epochs), leave=True)
    if notebook:
        iterator = trange(max_epochs, position=0, leave=True)
    m = 0
    T = abm_agg_ts.shape[1] - 1
    t = torch.linspace(0, T, T+1)
    model = ode(t)
    for epoch in iterator:
        
        # Gradient steps

        idx = 0
        total_epoch_loss = 0.
        # Shuffle training data
        #shuffled = torch.randperm(N_TRAIN)
        #train_abm_states = train_abm_states[shuffled]
        #train_abm_agg_ts = train_abm_agg_ts[shuffled]
        #train_abm_thi = train_abm_thi[shuffled]

        try:
            n_batches = 0
            while idx < N_TRAIN:
                optimiser.zero_grad()
                end_idx = min([idx + batch_size, N_TRAIN])

                # Batch
                stte_batch = train_abm_states[idx:end_idx]
                agg_states = train_abm_agg_ts[idx:end_idx]
                thi = train_abm_thi[idx:end_idx]

                # Parameters, intervention
                th, i = thi[:, :-1], thi[:, -1]
                loss = produce_loss(omega, calib, stte_batch, agg_states, th, i, model, loss_fn)
                total_epoch_loss += loss.item()

                loss.backward()
                optimiser.step()

                idx = end_idx
                if epoch == 0:
                    iterator.set_postfix({"best val loss":best_val_loss, "val loss":val_loss, "TEL":"Calculating...", "batch": idx // batch_size, "ESI":m, "train loss":loss.item()})
                else:
                    iterator.set_postfix({"best val loss":best_val_loss, "val loss":val_loss, "TEL":last_tel,         "batch": idx // batch_size, "ESI":m, "train loss":loss.item()})
                n_batches += 1

            with torch.no_grad():
                th, i = val_abm_thi[:, :-1], val_abm_thi[:, -1]
                val_loss = produce_loss(omega, calib, val_abm_states, val_abm_agg_ts, th, i, model, loss_fn)
                if not scheduler is None:
                    scheduler.step(val_loss)
                val_loss = val_loss.item()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_omega = copy.deepcopy(omega)
                    best_calib = copy.deepcopy(calib)
                    if full_node:
                        best_model = copy.deepcopy(model)
                    m = 0
                else:
                    m += 1
                loss_hist.append(val_loss)
                if max_epochs_no_improve < m:
                    logger.info("Converged")
                    if full_node:
                        return best_omega, best_calib, best_model, loss_hist
                    else:
                        return best_omega, best_calib, loss_hist

                last_tel = total_epoch_loss / n_batches
                iterator.set_postfix({"best val loss":best_val_loss, "val loss":val_loss, "TEL":last_tel, "batch":"val", "ESI":m, "train loss":loss.item()})
        except KeyboardInterrupt:
            if full_node:
                return best_omega, best_calib, best_model, loss_hist
            else:
                return best_omega, best_calib, loss_hist

    logger.info("Max epochs reached")
    if full_node:
        return best_omega, best_calib, best_model, loss_hist
    else:
        return best_omega, best_calib, loss_hist
