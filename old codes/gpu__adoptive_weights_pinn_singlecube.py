
"""
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
            x = torch.relu(self.layers[i](x))
        x = self.layers[-1](x)
        return x

# Define the network architecture
neu_num = 40
layers = [3, neu_num , neu_num, neu_num,neu_num , neu_num, neu_num, neu_num, neu_num,  4]  # Input: (x, y), Output: (u, v, p)
#layers = [3, 200, 200, 200, 200, 200, 200, 200, 4]  # Input: (x, y), Output: (u, v, p)
model = PINN(layers).to(device)

"""# **Define Desired PDE**"""

def navier_stokes_loss(model, x, y, z, mu):#, lambda_momentum=0.3, lambda_continuity=0.2):
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
    #loss_f = torch.mean(f_u**2) + torch.mean(f_v**2)+ torch.mean(f_w**2) + torch.mean(continuity**2)
    #return loss_f
    return torch.mean(f_u**2) , torch.mean(f_v**2), torch.mean(f_w**2) , torch.mean(continuity**2)

def boundary_condition_loss(model, x_b, y_b, z_b, u_b, v_b, w_b, p_b=None):
    uvp_b = model(torch.cat((x_b, y_b, z_b), dim=1))
    u_b_pred = uvp_b[:, 0:1]
    v_b_pred = uvp_b[:, 1:2]
    w_b_pred = uvp_b[:, 2:3]
    p_b_pred = uvp_b[:, 3:4]

    loss_u_b = torch.mean((u_b_pred - u_b) ** 2)
    loss_v_b = torch.mean((v_b_pred - v_b) ** 2)
    loss_w_b = torch.mean((w_b_pred - w_b) ** 2)
    loss_p_b = torch.mean((p_b_pred - p_b) ** 2)

    return loss_u_b + loss_v_b + loss_w_b + loss_p_b

def data_loss(model,x,y,z, u_exact, v_exact, w_exact, p_exact=None):
    uvp_pred = model(torch.cat((x, y, z), dim=1))
    u_pred = uvp_pred[:, 0:1]
    v_pred = uvp_pred[:, 1:2]
    w_pred = uvp_pred[:, 2:3]
    p_pred = uvp_pred[:, 3:4]

    loss_u = torch.mean((u_pred - u_exact) ** 2)
    loss_v = torch.mean((v_pred - v_exact) ** 2)
    loss_w = torch.mean((w_pred - w_exact) ** 2)
    loss_p = torch.mean((p_pred - p_exact) ** 2)

    #return loss_u + loss_v + loss_w + (loss_p if p_exact is not None else 0)
    return loss_u , loss_v , loss_w , loss_p

# Load data from CSV
data = pd.read_csv('around_building_clip.csv')
data = (data - data.min()) / (data.max() - data.min())
data['y']  = 0
bound = 350
x = torch.tensor(data[['x']][:bound].values, dtype=torch.float32).to(device)
y = torch.tensor(data[['y']][:bound].values, dtype=torch.float32).to(device)
z = torch.tensor(data[['z']][:bound].values, dtype=torch.float32).to(device)
u_exact = torch.tensor(data[['u']][:bound].values, dtype=torch.float32).to(device)
v_exact = torch.tensor(data[['v']][:bound].values, dtype=torch.float32).to(device)
w_exact = torch.tensor(data[['w']][:bound].values, dtype=torch.float32).to(device)
p_exact = torch.tensor(data[['p']][:bound].values, dtype=torch.float32).to(device) if 'p' in data.columns else None



# Example boundary data (you may need to replace this with actual data)

bc_data = pd.read_csv('singleCube_BC.csv')
bc_data = (bc_data - bc_data.min()) / (bc_data.max() - bc_data.min())
bc_data['v'] = bc_data['w']  = 0
x_b = (torch.tensor(bc_data['x'], dtype=torch.float32).to(device)).reshape(-1,1)
y_b = (torch.tensor(bc_data['y'], dtype=torch.float32).to(device)).reshape(-1,1)
z_b = (torch.tensor(bc_data['z'], dtype=torch.float32).to(device)).reshape(-1,1)
u_b = (torch.tensor(bc_data['u'], dtype=torch.float32).to(device)).reshape(-1,1)  # Boundary u-values
v_b = (torch.tensor(bc_data['v'], dtype=torch.float32).to(device)).reshape(-1,1)  # Boundary v-values
w_b = (torch.tensor(bc_data['w'], dtype=torch.float32).to(device)).reshape(-1,1)  # Boundary w-values
p_b = (torch.tensor(bc_data['p'], dtype=torch.float32).to(device)).reshape(-1,1)  # Boundary w-values

data_loss(model, x, y, z, u_exact, v_exact, w_exact, p_exact)

def plot_losses(loss_history):
    losses, momentum_losses, continuity_losses, boundary_losses, data_loss = zip(*loss_history)
    epochs = range(len(losses))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, label="Total Loss", color='b')
    #plt.plot(epochs, momentum_losses, label="Momentum Loss", color='r')
    #plt.plot(epochs, continuity_losses, label="Continuity Loss", color='g')
    #plt.plot(epochs, boundary_losses, label="Boundary Loss", color='orange')
    plt.plot(epochs, data_loss, label="Data Loss", color='k')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Losses during Training")
    plt.legend()
    plt.show()

def adop_loss_Weight(model, x, y, z, u_exact, v_exact , w_exact ,
               x_b, y_b, z_b, u_b, v_b, w_b):
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
    momentum_loss = ((w_fu * lfu + w_fv * lfv + w_fw * lfw) / 3.0)
    loss_data = ((wu * l_u + wv * l_v + ww * l_w + wp * l_p) / 4.0)
    loss_BC = (wbc * BC)
    continuity_loss = lcont

    return loss # , momentum_loss , loss_data , loss_BC , continuity_loss

"""# **Train with LBGFS optimizers**"""

### Only LBGFS optimizers
# Define training loop with hybrid optimization
def train_pinn(model, x, y, z, u_exact, v_exact , w_exact ,
               x_b, y_b, z_b, u_b, v_b, w_b ,
               num_epochs_adam, num_epochs_lbfgs,
               prev_lfu , prev_lfv , prev_lfw , prev_lcont , prev_lu , prev_lv , prev_lw , prev_lp , prev_lbc):
  # Optimizers
  loss1 = nn.MSELoss()
  num_epochs = 10000
  mu = 1.85E-5 / 1.225



    #mu = 0.01
  #optimizer_adam = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)  # L2 regularization to prevent overfitting
  optimizer =torch.optim.LBFGS(
    model.parameters(),
    lr=0.01,  # or adjust based on your problem
    max_iter=500,  # More iterations for better convergence
    max_eval=None,  # Default
    tolerance_grad=1e-7,  # Increase sensitivity to gradients
    tolerance_change=1e-9,  # Keep default unless facing early stops
    history_size=100  # Use larger history for better approximations
)



  # Phase 1: Adam Optimization
  loss_history = []
  for epoch in range(num_epochs):

    model.train()
    
    def closure():
      optimizer.zero_grad()
      loss = adop_loss_Weight(model, x, y, z, u_exact, v_exact , w_exact ,
                     x_b, y_b, z_b, u_b, v_b, w_b)
      loss.backward()
      # Apply gradient clipping
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Adjust max_norm as needed
      return loss
    optimizer.zero_grad()
    loss = optimizer.step(closure)

    if epoch % 1 == 0:
      model.eval()
      loss_history.append(loss.item())
      print(f"Epoch {epoch}, Total Loss: {loss.item():.4f}, "
                f"Momentum Loss: {loss.item():.4f}, "
                f"Continuity Loss: {loss.item():.4f}, "
                f"Data Loss: {loss.item():.4f}, "
                f"Boundary Loss: {loss.item():.4f}  ")
    plot_losses(loss_history)

num_epochs_adam = 15000
num_epochs_lbfgs = 4000

prev_lfu = torch.tensor(1E10)
prev_lfv = torch.tensor(1E10)
prev_lfw = torch.tensor(1E10)
prev_lcont = torch.tensor(1E10)
prev_lu = torch.tensor(1E10)
prev_lv = torch.tensor(1E10)
prev_lw = torch.tensor(1E10)
prev_lp = torch.tensor(1E10)
prev_lbc = torch.tensor(1E10)

train_pinn(model, x, y, z, u_exact, v_exact , w_exact,
               x_b, y_b, z_b, u_b, v_b, w_b ,
               num_epochs_adam, num_epochs_lbfgs,
               prev_lfu , prev_lfv , prev_lfw , prev_lcont , prev_lu , prev_lv , prev_lw , prev_lp , prev_lbc)

##countur plot visualization

# **Plot section**


# Convert tensors to numpy arrays
xx = x[:50]
yy = y[:50]
zz = z[:50]
uu = u_exact[:50]
vv = v_exact[:50]
ww = w_exact[:50]
pp = p_exact[:50]

x_train_np = xx.cpu().detach().numpy()
y_train_np = yy.cpu().detach().numpy()
z_train_np = zz.cpu().detach().numpy()
u_exact_np = uu.cpu().detach().numpy()
v_exact_np = vv.cpu().detach().numpy()
w_exact_np = ww.cpu().detach().numpy()
p_exact_np = pp.cpu().detach().numpy()

uvp_pred = model(torch.cat((xx, yy, zz), dim=1))
u_pred = uvp_pred[:, 0:1]
v_pred = uvp_pred[:, 1:2]
w_pred = uvp_pred[:, 2:3]
p_pred = uvp_pred[:, 3:4]

u_pred_np = u_pred.cpu().detach().numpy()
v_pred_np = v_pred.cpu().detach().numpy()
w_pred_np = w_pred.cpu().detach().numpy()
p_pred_np = p_pred.cpu().detach().numpy()

plt.figure(dpi = 150)
plt.plot(u_exact_np, label='Exact u', marker='o')
plt.plot(u_pred_np, label='Predicted u', marker='x')
plt.legend()
plt.title('Comparison of u component')


# Plotting the results
plt.figure(figsize=(20, 5))

plt.subplot(1, 3, 1)
plt.plot(v_exact_np, label='Exact v', marker='o')
plt.plot(v_pred_np, label='Predicted v', marker='x')
plt.legend()

plt.title('Comparison of u component')

plt.subplot(1, 3, 2)
plt.plot(w_exact_np, label='Exact w', marker='o')
plt.plot(w_pred_np, label='Predicted w', marker='x')
plt.legend()
plt.title('Comparison of v component')

plt.subplot(1, 3, 3)
plt.plot(p_exact_np, label='Exact p', marker='o')
plt.plot(p_pred_np, label='Predicted p', marker='x')
plt.legend()

plt.title('Predicted Pressure')



plt.show()

# **Test with new Data**

bound = 250

x_test = torch.tensor(data[['x']][bound:].values, dtype=torch.float32).to(device)
y_test = torch.tensor(data[['y']][bound:].values, dtype=torch.float32).to(device)
z_test = torch.tensor(data[['z']][bound:].values, dtype=torch.float32).to(device)
u_test = torch.tensor(data[['u']][bound:].values, dtype=torch.float32).to(device)
v_test = torch.tensor(data[['v']][bound:].values, dtype=torch.float32).to(device)
w_test = torch.tensor(data[['w']][bound:].values, dtype=torch.float32).to(device)
p_test = torch.tensor(data[['p']][bound:].values, dtype=torch.float32).to(device)

uvp_test = model(torch.cat((x_test, y_test, z_test), dim=1))
ut_pred = uvp_test[:,0:1]

ut_pred = ut_pred.cpu().detach().numpy()
plt.plot(u_test.cpu(), label = "Exact")
plt.plot(ut_pred, label = "PINN")
plt.savefig('./uvp.png', dpi=300)
plt.legend()

x.shape

import matplotlib.cm as cm
xx = ((torch.tensor(data['x'][:], dtype=torch.float32)).reshape(-1,1)).to(device)
yy = ((torch.tensor(data['y'][:], dtype=torch.float32)).reshape(-1,1)).to(device)
zz = ((torch.tensor(data['z'][:], dtype=torch.float32)).reshape(-1,1)).to(device)
uu = ((torch.tensor(data['u'][:], dtype=torch.float32)).reshape(-1,1)).to(device)

uvwp_test = model(torch.cat((xx, yy, zz), dim=1))
up = uvwp_test[:,0:1]

cmap = cm.get_cmap(name='RdBu_r', lut=None)
plt.subplots(1,1)
plt.title("Exact")
plt.tricontourf(xx.cpu().flatten(), yy.cpu().flatten(), uu.cpu().flatten(), cmap = 'RdBu_r')
plt.colorbar()

plt.subplots(1,1)
plt.title("PINN")
plt.tricontourf(xx.cpu().flatten(),yy.cpu().flatten(),up.cpu().detach().numpy().flatten(), cmap="RdBu_r")
plt.colorbar()
plt.show()

up.shape


