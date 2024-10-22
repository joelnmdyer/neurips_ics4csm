import torch

###
### NAIVE EULER + Intervention in state
###

class BWSGODE_naive_int(torch.nn.Module):
    def __init__(self, t):
        super().__init__()
        self.t = t

    def forward(self, y0, params):
        i = y0[-1]
        y = y0.unsqueeze(0)
        interventional = (i != 0.).float()
        for j in range(1, len(self.t)):
            B, W, S, G = y[-1, :4]
            mask = interventional * (j >= 5 + i - 1) + (1 - interventional)
            assert (params >= 0.).all()
            #dG = params[0]*( 1. - G / params[1] )*G - params[2]*S # Set carrying capacity to 1?
            dG = (params[0]*( 1. - G ) - params[1]*S)*G # Set carrying capacity to 1?
            dS = S*(params[2]*G - params[3]*(W + B*mask) - params[4]) # Set W and B coefficients to be the same?
            dW = W*(params[5]*S - params[6]*B*mask - params[7])*W 
            dB = B*mask*(params[8]*(S + W) - params[9]) # Set W and S coefficients to be the same?
            dy = torch.stack([dB, dW, dS, dG, torch.tensor(0.)]).unsqueeze(0)
            new_state = y[-1] + dy
            y = torch.cat((y, new_state), dim=0)

        return y

class BWSGRicker(torch.nn.Module):
    def __init__(self, t):
        super().__init__()
        self.t = t
        self.logpsi = torch.nn.Parameter(torch.randn(7))
        self.logpsi.requires_grad = True

    def forward(self, y0, params):
        """Assumes y0 = torch.tensor([init_prop_B, init_prop_W, init_prop_S, init_prop_G, i]), and that specifically init_prop_B = 0. (i.e. no bears initially)"""
        i = y0[-1]
        y = y0[:-1].unsqueeze(0)
        interventional = (i != 0.).float()
        for j in range(1, len(self.t)):
            B, W, S, G = y[-1, :4]
            mask = interventional * (j == 10 + i - 1)
            assert (params >= 0.).all(), print(params)
            new_G = G*torch.exp(torch.exp(self.logpsi[0])*(1 - G) - torch.exp(self.logpsi[1]*S))
            new_S = S*torch.exp(params[0]*G*(1.-S) - torch.exp(self.logpsi[2])*(W+B) - torch.exp(self.logpsi[3]))
            new_W = W*torch.exp(params[1]*S*(1.-W) - torch.exp(self.logpsi[4])*B - torch.exp(self.logpsi[5]))
            new_B = mask * torch.tensor(10./400.) + B*torch.exp(params[-1]*(S + W)*(1. - B) - torch.exp(self.logpsi[-1]))
            new_state = torch.clamp(torch.cat((new_B.unsqueeze(0), new_W.unsqueeze(0), new_S.unsqueeze(0), new_G.unsqueeze(0)), dim=-1).unsqueeze(0), max=1.)
            y = torch.cat((y, new_state), dim=0)
        y = torch.nn.functional.pad(y, (1,1), "constant", 0.)

        return y[:, 1:]




class BWSGRNN(torch.nn.Module):
    def __init__(self, t, net):
        super().__init__()
        self.net = net

    def forward(self, y0, params):
        """Assumes a 1-layer GRU for self.net"""
        #print(y0, y0.shape)
        i = int(y0[-1].item())
        h = torch.cat((y0[:-1]/400., params), dim=-1)
        h = torch.cat((h, torch.zeros(self.net.hdim - h.numel())), dim=-1)
        h = h.unsqueeze(0)
        interventional = (i != 0)
        control = torch.zeros((51, 1)).double()
        #print(control.shape)
        if interventional:
            control[10 + i, 0] = 10.
        #print(self.net)
        y = self.net(control, h)
        #print(y.shape)
        #out_y0 = self.net._fff(h)
        #y = torch.cat((out_y0, y), dim=0)
        #y = torch.cat((y0[:-1].unsqueeze(0), y), dim=0)

        y = torch.nn.functional.pad(y, (1,1), "constant", 0.)
        #print(y, y.shape)
        return y[:, 1:]


def create_kl_divergence(net, instantiate_emission):

    def kl_divergence(x, y):

        """
        x is from ABM, y from ODE
        """
        #x.clamp_(min=1e-8)
        x, y = x[0], y[0]
        emission_params = net(y)
        emissions = [instantiate_emission(e_pars) for e_pars in emission_params]
        lps = torch.stack([emissions[j].log_prob(x[j]) for j in range(x.shape[0])])
        if lps.isnan().any():
            print("nan")
        return -torch.sum(lps)

    return kl_divergence
