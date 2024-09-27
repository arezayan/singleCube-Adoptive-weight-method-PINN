# -*- coding: utf-8 -*-
"""
Adoptive_LBGFS_singleCube.ipynb

Created on Wed Aug 14 09:57:07 2024
singleCube case
An incompressible turbulent flow over a single cube. Three-dimensional is considered.

@author: Amirreza

"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from PIL import Image

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""# **Define Network | Physics-informed**"""

class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.layers = nn.ModuleList()

        for i in range(len(layers) - 1):
            layer = nn.Linear(layers[i], layers[i + 1])
            nn.init.xavier_uniform_(layer.weight)  # Xavier initialization for weights
            nn.init.zeros_(layer.bias)             # Initialize biases to zero
            self.layers.append(layer)

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = torch.tanh(self.layers[i](x))
        x = self.layers[-1](x)
        return x

# Define the network architecture
neuron_num = 40
#layers = [3,  20, 20, 20, 20, 20, 20, 20, 20,  4]  # Input: (x, y), Output: (u, v, p)
layers = [3, neuron_num, neuron_num, neuron_num, neuron_num, neuron_num, neuron_num, neuron_num, 4]  # Input: (x, y), Output: (u, v, p)
model = PINN(layers).to(device)

def adop_loss_Weight(model, x, y, z, u_exact, v_exact , w_exact , p_exact ,
               x_b, y_b, z_b, u_b, v_b, w_b ,p_b):
    prev_lfu = torch.tensor(1E10)
    prev_lfv = torch.tensor(1E10)
    prev_lfw = torch.tensor(1E10)
    prev_lcont = torch.tensor(1E10)
    prev_lu = torch.tensor(1E10)
    prev_lv = torch.tensor(1E10)
    prev_lw = torch.tensor(1E10)
    prev_lp = torch.tensor(1E10)
    prev_lbc = torch.tensor(1E10)


     # Forward pass
    loss1 = nn.MSELoss()
    mu = 1.85E-5 / 1.225
    lfu , lfv, lfw, lcont = navier_stokes_loss(model, x, y, z, mu)
    uvwp = model(torch.cat((x, y, z), dim=1))
    BC = boundary_condition_loss(model, x_b, y_b,z_b, u_b, v_b,w_b, p_b)

    l_u = loss1(uvwp[:,0:1] , u_exact)
    l_v = loss1(uvwp[:,1:2] , v_exact)
    l_w = loss1(uvwp[:,2:3] , w_exact)
    l_p = loss1(uvwp[:,3:4] , p_exact)

    # Update weights based on the rate of change of the losses
    with torch.no_grad():
      delta_lfu = abs(prev_lfu - lfu.item())
      delta_lfv = abs(prev_lfv - lfv.item())
      delta_lfw = abs(prev_lfw - lfw.item())
      delta_lcont = abs(prev_lcont - lcont.item())

      delat_lu = abs(prev_lu - l_u.item())
      delat_lv = abs(prev_lv - l_v.item())
      delat_lw = abs(prev_lw - l_w.item())
      delta_lp = abs(prev_lp - l_p.item())
      delta_lbc = abs(prev_lbc - BC.item())


      total_delta = delta_lfu + delta_lfv + delta_lfw + delta_lcont + delat_lu + delat_lv + delat_lw + delta_lp + delta_lbc + 1e-8
      # Update weights: give more weight to the losses that are changing less
      w_fu = delta_lfu / total_delta
      w_fv = delta_lfv / total_delta
      w_fw = delta_lfw / total_delta
      w_cont = delta_lcont / total_delta

      wu = delat_lu / total_delta
      wv = delat_lv / total_delta
      ww = delat_lw / total_delta
      wp = delta_lp / total_delta
      wbc = delta_lbc / total_delta

      prev_lfu = lfu.item()
      prev_lfv = lfv.item()
      prev_lfw = lfw.item()
      prev_lcont = lcont.item()
      prev_lu = l_u.item()
      prev_lv = l_v.item()
      prev_lw = l_w.item()
      prev_lp = l_p.item()
      prev_lbc = BC.item()

    #loss = w_fu * lfu + w_fv * lfv + w_fw * lfw +  wu * l_u + wv * l_v + ww * l_w + wp * l_p + wbc * BC
    loss = w_fu * lfu + w_fv * lfv + w_fw * lfw + w_cont * lcont + wu * l_u + wv * l_v + ww * l_w + wp * l_p + wbc * BC
    momentum_loss = ((w_fu * lfu + w_fv * lfv + w_fw * lfw) / 3.0).detach().numpy()
    loss_data = ((wu * l_u + wv * l_v + ww * l_w + wp * l_p) / 4.0).detach().numpy()
    loss_BC = (wbc * BC).detach().numpy()
    continuity_loss = lcont.detach().numpy()

    return loss , momentum_loss , loss_data , loss_BC , continuity_loss

"""# **Define Desired PDE**"""

def navier_stokes_loss(model, x, y, z, mu):
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)
    z = z.requires_grad_(True)

    uvp = model(torch.cat((x, y, z), dim=1))
    u = uvp[:, 0:1]
    v = uvp[:, 1:2]
    w = uvp[:, 2:3]
    p = uvp[:, 3:4]

    # Calculate gradients
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_z = torch.autograd.grad(u, z, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_z = torch.autograd.grad(v, z, grad_outputs=torch.ones_like(v), create_graph=True)[0]

    w_x = torch.autograd.grad(w, x, grad_outputs=torch.ones_like(w), create_graph=True)[0]
    w_y = torch.autograd.grad(w, y, grad_outputs=torch.ones_like(w), create_graph=True)[0]
    w_z = torch.autograd.grad(w, z, grad_outputs=torch.ones_like(w), create_graph=True)[0]

    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    u_zz = torch.autograd.grad(u_z, z, grad_outputs=torch.ones_like(u_z), create_graph=True)[0]

    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]
    v_zz = torch.autograd.grad(v_z, z, grad_outputs=torch.ones_like(v_z), create_graph=True)[0]

    w_xx = torch.autograd.grad(w_x, x, grad_outputs=torch.ones_like(w_x), create_graph=True)[0]
    w_yy = torch.autograd.grad(w_y, y, grad_outputs=torch.ones_like(w_y), create_graph=True)[0]
    w_zz = torch.autograd.grad(w_z, z, grad_outputs=torch.ones_like(w_z), create_graph=True)[0]

    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_z = torch.autograd.grad(p, z, grad_outputs=torch.ones_like(p), create_graph=True)[0]

    # Navier-Stokes equations

    f_u = u*u_x + v*u_y + w*u_z + p_x - mu * (u_xx + u_yy + u_zz)
    f_v = u*v_x + v*v_y + w*v_z + p_y - mu * (v_xx + v_yy + v_zz)
    f_w = u*w_x + v*w_y + w*w_z + p_z - mu * (w_xx + w_yy + w_zz)

    # Continuity equation
    continuity = u_x + v_y + w_z

    # Loss calculation with balancing factors
    #loss_momentum =  (torch.mean(f_u**2) + torch.mean(f_v**2)+ torch.mean(f_w**2))
    lfu = torch.mean(f_u**2)
    lfv = torch.mean(f_v**2)
    lfw = torch.mean(f_w**2)
    loss_continuity = torch.mean(continuity**2)
    return lfu , lfv , lfw , loss_continuity

def boundary_condition_loss(model, x_b, y_b, z_b, u_b, v_b, w_b, p_b):
    uvp_b = model(torch.cat((x_b, y_b, z_b), dim=1))
    u_b_pred = uvp_b[:, 0:1]
    v_b_pred = uvp_b[:, 1:2]
    w_b_pred = uvp_b[:, 2:3]
    p_b_pred = uvp_b[:, 3:4]

    loss_u_b = torch.mean((u_b_pred - u_b) ** 2)
    loss_v_b = torch.mean((v_b_pred - v_b) ** 2)
    loss_w_b = torch.mean((w_b_pred - w_b) ** 2)
    loss_p_b = torch.mean((p_b_pred - p_b) ** 2)

    return loss_u_b + loss_v_b + loss_w_b #+ loss_p_b

def data_loss(model,x,y,z, u_exact, v_exact, w_exact, p_exact):
    uvp_pred = model(torch.cat((x, y, z), dim=1))
    u_pred = uvp_pred[:, 0:1]
    v_pred = uvp_pred[:, 1:2]
    w_pred = uvp_pred[:, 2:3]
    p_pred = uvp_pred[:, 3:4] if p_exact is not None else None

    loss_u = torch.mean((u_pred - u_exact) ** 2)
    loss_v = torch.mean((v_pred - v_exact) ** 2)
    loss_w = torch.mean((w_pred - w_exact) ** 2)
    loss_p = torch.mean((p_pred - p_exact) ** 2) if p_exact is not None else 0

    return loss_u + loss_v + loss_w + (loss_p if p_exact is not None else 0)

"""

# the simple loss calculation
def total_loss(model, x, y, z, u_exact, v_exact,w_exact,p_exact,
               mu, x_b , y_b, z_b, u_b, v_b, w_b,p_b,
               weight_momentum,weight_continuity , loss_data ,lambda_bc):

    # Physics-informed loss
    mom_loss, cont_loss = navier_stokes_loss(model, x, y, z, mu)
    momentum_loss  = mom_loss * weight_momentum
    continuity_loss  = cont_loss * weight_continuity

    # Data loss

    loss_data = data_loss(model, x, y, z, u_exact, v_exact, w_exact, p_exact) * lambda_data

    # Boundary condition loss
    loss_bc = boundary_condition_loss(model, x_b, y_b,z_b, u_b, v_b,w_b, p_b) * lambda_bc if x_b is not None else 0
    loss = momentum_loss + continuity_loss + loss_data + loss_bc

    return loss, momentum_loss , continuity_loss , loss_data , loss_bc
"""
# Load data from CSV
bound = 9500 # number of samples coorporated in traing
data = pd.read_csv('/content/drive/MyDrive/cavity/singleCube/around_building_clip.csv')
data = (data - data.min()) / (data.max() - data.min())
data['y'] = 0

x = torch.tensor(data[['x']][:bound].values, dtype=torch.float32).to(device)
y = torch.tensor(data[['y']][:bound].values, dtype=torch.float32).to(device)
z = torch.tensor(data[['z']][:bound].values, dtype=torch.float32).to(device)
u_exact = torch.tensor(data[['u']][:bound].values, dtype=torch.float32).to(device)
v_exact = torch.tensor(data[['v']][:bound].values, dtype=torch.float32).to(device)
w_exact = torch.tensor(data[['w']][:bound].values, dtype=torch.float32).to(device)
p_exact = torch.tensor(data[['p']][:bound].values, dtype=torch.float32).to(device) if 'p' in data.columns else None

# Example boundary data (you may need to replace this with actual data)

bc_data = pd.read_csv('/content/drive/MyDrive/cavity/singleCube/singleCube_BC.csv')
bc_data = (bc_data - bc_data.min()) / (bc_data.max() - bc_data.min())
#bc_data['v'] = bc_data['w'] = 0
x_b = (torch.tensor(bc_data['x'], dtype=torch.float32).to(device)).reshape(-1,1)
y_b = (torch.tensor(bc_data['y'], dtype=torch.float32).to(device)).reshape(-1,1)
z_b = (torch.tensor(bc_data['z'], dtype=torch.float32).to(device)).reshape(-1,1)
u_b = (torch.tensor(bc_data['u'], dtype=torch.float32).to(device)).reshape(-1,1)  # Boundary u-values
v_b = (torch.tensor(bc_data['v'], dtype=torch.float32).to(device)).reshape(-1,1)  # Boundary v-values
w_b = (torch.tensor(bc_data['w'], dtype=torch.float32).to(device)).reshape(-1,1)  # Boundary w-values
p_b = (torch.tensor(bc_data['p'], dtype=torch.float32).to(device)).reshape(-1,1)  # Boundary w-values


# Training parameters
epo_adam = 25000
epochs =15000
mu = 1E-5/1.225  # Dynamic viscosity
weight_momentum = 1
weight_continuity = 1
lambda_data = 1
lambda_bc = 1

# Define the optimizer
optimizer_adam = torch.optim.Adam(
    model.parameters(),         # Model parameters
    lr=1e-3,                    # Learning rate (adjust as needed)
    betas=(0.9, 0.999),         # Coefficients for momentum and RMSProp-like behavior
    eps=1e-8,                   # Small term to avoid division by zero
    weight_decay=0.01,          # L2 regularization (use 0 to disable)
    amsgrad=False               # AMSGrad variant (set to True if needed)
)
optimizer=torch.optim.LBFGS(model.parameters(),
    lr=0.001,  # or adjust based on your problem
    max_iter=50,  # More iterations for better convergence
    max_eval=None,  # Default
    tolerance_grad=1e-7,  # Increase sensitivity to gradients
    tolerance_change=1e-9,  # Keep default unless facing early stops
    history_size=100,  # Use larger history for better approximations
    line_search_fn="strong_wolfe"  # Use strong Wolfe line search for better convergence

)

def closure():
    optimizer.zero_grad()

    loss , momentum_loss , loss_data , loss_BC , continuity_loss = adop_loss_Weight(model, x, y, z, u_exact, v_exact , w_exact , p_exact ,
               x_b, y_b, z_b, u_b, v_b, w_b ,p_b)
    loss.backward()
    return loss

for epo in range(epo_adam):
    model.train()
    optimizer_adam.zero_grad()
    loss , momentum_loss , loss_data , loss_BC , continuity_loss = adop_loss_Weight(model, x, y, z, u_exact, v_exact , w_exact , p_exact ,
               x_b, y_b, z_b, u_b, v_b, w_b ,p_b)
    loss.backward()
    if epo %500 == 0:
      print(f'Epoch Adam {epo}, Total Loss: {loss.item():.5f}')
    if loss.item() <=0.5 :
      print("Optimzation Method is swtiching to LBGF-S soon  . . . ")
      break



for epochs in range(epochs):
    model.train()
    loss = optimizer.step(closure)

    if epochs % 10 == 0:
        print(f'Epoch LBGF-s {epochs}, Total Loss: {loss.item():.5f}')
        #print(f'The highest Loss is:  {max(momentum_loss.item() , continuity_loss.item() , loss_data.item() , loss_bc.item()):.6f}')
        #print(time.time())

model.eval()
with torch.no_grad():
    uvp_pred = model(torch.cat((x, y, z), dim=1))
    u_pred = uvp_pred[:, 0:1]
    v_pred = uvp_pred[:, 1:2]
    w_pred = uvp_pred[:, 2:3]
    p_pred = uvp_pred[:, 3:4]


#plotting section
def plot_result(model, x,y,z, u,v,w, al):
    uvwp_pred = model(torch.cat((x, y, z), dim=1))
    u_pred = uvwp_pred[:,0]
    v_pred = torch.tensor(0)
    w_pred = torch.tensor(0)
    Umag_pred = torch.sqrt(u_pred**2 + v_pred **2 + w_pred**2)
    plt.figure(dpi = 100)
    plt.subplot(2,1,1)
    plt.title("comparision: " + "u" )
    plt.plot(Umag_pred.cpu().detach().numpy(),ls = "", marker = "+",  color = "r",label= "predicition with PINN" )

    Umag_exact = torch.sqrt(u**2 + v **2 + w**2)
    plt.plot( u.cpu(), ls = "", marker = "^",color = "b", alpha = al , label = "Exact")
    plt.legend(loc = "best")



    plt.subplot(2,1,2)
    plt.title("Error: " + "%" )
    plt.plot((Umag_pred.cpu().detach().numpy() - Umag_exact.cpu().detach().numpy() / Umag_pred.cpu().detach().numpy()),ls = "", marker = "+", color =  "r",label= "PINN Error" )
    plt.plot(u_exact.cpu(),ls = "", marker = "^", color = "b" ,alpha = al, label = "Exact")
    plt.legend(loc = "best")

    plt.figure(dpi = 100)

plot_result(model, x, y, z, u_exact, v_exact, w_exact, 0.1)

#predict with Unseen data
# Load data from CSV
bound = 14700 # number of samples coorporated in traing
#data_sa = pd.read_csv('/content/drive/MyDrive/cavity/singleCube/sampled_data.csv')
data_sa = pd.read_csv('/content/drive/MyDrive/cavity/singleCube/around_building_clip.csv')

data_sa= (data_sa - data_sa.min()) / (data_sa.max() - data_sa.min())
data_sa['y'] = 0

x_sa = torch.tensor(data_sa[['x']][bound:].values, dtype=torch.float32).to(device)
y_sa = torch.tensor(data_sa[['y']][bound:].values, dtype=torch.float32).to(device)
z_sa = torch.tensor(data_sa[['z']][bound:].values, dtype=torch.float32).to(device)
u_sa = torch.tensor(data_sa[['u']][bound:].values, dtype=torch.float32).to(device)
v_sa = torch.tensor(data_sa[['v']][bound:].values, dtype=torch.float32).to(device)
w_sa = torch.tensor(data_sa[['w']][bound:].values, dtype=torch.float32).to(device)
p_sa = torch.tensor(data_sa[['p']][bound:].values, dtype=torch.float32).to(device)



#plot_result(model, x[bound:], y[bound:], z[bound:], u_exact[bound:], v_exact[bound:], w_exact[bound:], 0.25)
uvwp_pred = model(torch.cat((x_sa, y_sa, z_sa), dim=1))
u_pred = uvwp_pred[:,0]
plt.plot(u_sa.cpu() , marker = "o" , color = "k" , label = "exact")
plt.plot(u_pred.cpu().detach().numpy() ,ls = "-", marker="+" ,label = "PINN")
plt.legend(loc = "best")
plt.ylim(0,1.2)
plt.show()

