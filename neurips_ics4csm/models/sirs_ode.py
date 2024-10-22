import torch

###
### NAIVE EULER + Intervention in state
###

class SIRSODE_naive_int(torch.nn.Module):
    def __init__(self, t):
        super().__init__()
        self.t = t

    def forward(self, y0, params):
        i = y0[-1]
        y = y0.unsqueeze(0)
        interventional = (i != 0.).float()
        for j in range(1, len(self.t)):
            S, I, R = y[-1, :3]
            mask = interventional * ( ~( (j >= 5 + i - 1) and (j <= 10 + i - 1) ) ) + (1 - interventional)
            alpha, beta, gamma = params * torch.tensor([mask, 1., 1.])
            assert ((alpha >= 0) #and (alpha <= 1) 
                    and (beta >= 0) #and (beta <= 1) 
                    and (gamma >= 0) #and (gamma <= 1)
                   ), print(alpha, beta, gamma)
            dS = gamma * R - alpha * S * I
            dI = alpha * S * I - beta * I
            dR = beta * I - gamma * R
            dy = torch.stack([dS, dI, dR, torch.tensor(0.)]).unsqueeze(0)
            new_state = y[-1] + dy
            y = torch.cat((y, new_state), dim=0)

        return y


class SIRSRNN(torch.nn.Module):
    def __init__(self, t, net):
        super().__init__()
        self.t = t
        self.net = net

    def forward(self, y0, params):
        """Assumes a 1-layer GRU for self.net"""
        i = int(y0[-1].item())
        h = y0[:-1]
        h = torch.cat((h, torch.zeros(self.net.hdim - h.numel())), dim=-1)
        h = h.unsqueeze(0)
        if not len(params.shape) > 1:
            params = params.unsqueeze(0)
        params = params.repeat(self.t.numel() - 1, 1)
        interventional = (i != 0)
        if interventional:
            params[5 + i - 1:10 + i - 1] *= torch.tensor([[0., 1., 1.]])
        y = self.net(params, h)
        #out_y0 = self.net._fff(h)
        #y = torch.cat((out_y0, y), dim=0)
        y = torch.cat((y0[:-1].unsqueeze(0).log(), y), dim=0)

        y = torch.nn.functional.pad(y, (1,1), "constant", 0.)
        return y[:, 1:]


def create_kl_divergence(net, instantiate_emission, N=2500):

    def kl_divergence(x, y):

        """
        x is from ABM, y from ODE
        """
        #x.clamp_(min=1e-8)
        x, y = x[0], y[0]
        x = (x * N).int()
        emission_params = net(y)
        emissions = [instantiate_emission(e_pars) for e_pars in emission_params]
        lps = torch.stack([emissions[j].log_prob(x[j]) for j in range(x.shape[0])])
        if lps.isnan().any():
            print("nan")
        return -torch.sum(lps)

    return kl_divergence
