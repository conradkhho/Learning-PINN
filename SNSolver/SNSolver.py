# ============================================================
# SNSolver.py
# Schrödinger–Newton PINN Solver
# Developed by Conrad Ho with assistance from OpenAI ChatGPT (GPT-5)
# ============================================================

# --- Imports ---
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import time
from PIL import Image
import datetime

# ============================================================
# 1. Helper Components
# ============================================================

class FCN(torch.nn.Module):
    """Fully Connected Neural Network for ψ(x,t) or φ(x,t)."""
    def __init__(self, in_dim, hidden_dim, hidden_layers, activation="tanh"):
        super().__init__()
        layers = [torch.nn.Linear(in_dim, hidden_dim)]
        for _ in range(hidden_layers-1):
            layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
        layers.append(torch.nn.Linear(hidden_dim,2))
        self.layers = torch.nn.ModuleList(layers)
        self.act = getattr(torch, activation) if hasattr(torch, activation) else torch.tanh

    def forward(self,x):
        for layer in self.layers[:-1]:
            x = self.act(layer(x))
        return torch.chunk(self.layers[-1](x),2,dim=1)

def derivatives(u, xt):
    grads = torch.autograd.grad(u, xt, grad_outputs=torch.ones_like(u),
                                create_graph=True, retain_graph=True)[0]
    u_t = grads[:,1:2]
    u_x = grads[:,0:1]
    u_xx = torch.autograd.grad(u_x, xt, grad_outputs=torch.ones_like(u_x),
                               create_graph=True, retain_graph=True)[0][:,0:1]
    return u_t, u_xx

def save_gif_PIL(gif_path, frames, fps=5):
    imgs = [Image.open(f) for f in frames]
    imgs[0].save(gif_path, save_all=True, append_images=imgs[1:], duration=int(1000/fps), loop=0)

# ============================================================
# 2. Schrödinger–Newton Solver
# ============================================================

class SNSolver:
    """PINN Solver for Schrödinger–Newton equation with grid plots and GIF."""

    def __init__(self,
                 psi0_func,
                 V_func=None,
                 enable_SN=True,
                 enforce_norm=True,
                 x_range=(-3,3),
                 t_range=(0,1),
                 nx=50,
                 nt=50,
                 hidden_dim=64,
                 hidden_layers=4,
                 activation="tanh",
                 optimizer_type="adam",
                 lr=1e-4,
                 epochs=2000,
                 hbar=1.0,
                 G=1.0,
                 m=1.0,
                 lambda_SN=1.0,
                 lambda_PEq=1,
                 lambda_IC=1,
                 lambda_norm=1,
                 loss_tol=None,
                 show_plot=True,
                 save_plot=True,
                 save_gif=True,
                 save_data=True,
                 save_model=True,
                 grid_plot=True,
                 grid_cols=4,
                 grid_cell_size=(4,3),
                 xlim = None,
                 ylim = None,
                 output_dir="Results",
                 seed=123,
                 device=None):

        self.psi0_func = psi0_func
        self.V_func = V_func
        self.enable_SN = enable_SN
        self.enforce_norm = enforce_norm
        self.x_range, self.t_range = x_range, t_range
        self.nx, self.nt = nx, nt
        self.hidden_dim, self.hidden_layers = hidden_dim, hidden_layers
        self.activation = activation
        self.optimizer_type = optimizer_type
        self.lr, self.epochs = lr, epochs
        self.hbar, self.G, self.m = hbar, G, m
        self.lambda_SN, self.lambda_PEq = lambda_SN, lambda_PEq
        self.lambda_IC, self.lambda_norm = lambda_IC, lambda_norm
        self.loss_tol = loss_tol
        self.show_plot, self.save_plot = show_plot, save_plot
        self.save_gif, self.save_data, self.save_model = save_gif, save_data, save_model
        self.grid_plot, self.grid_cols = grid_plot, grid_cols
        self.grid_cell_size = grid_cell_size
        self.xlim, self.ylim = xlim, ylim
        self.seed = seed
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # Output folder
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(output_dir,timestamp)
        os.makedirs(self.output_dir,exist_ok=True)

        # Seed
        torch.manual_seed(seed)
        np.random.seed(seed)

    def run(self):
        """Train PINN and generate plots/GIF."""
        # Grids
        x_min, x_max = self.x_range
        t_min, t_max = self.t_range
        x_train = torch.linspace(x_min, x_max, self.nx).view(-1,1)
        t_train = torch.linspace(t_min, t_max, self.nt).view(-1,1)
        X,T = torch.meshgrid(x_train.squeeze(), t_train.squeeze(), indexing='ij')
        xt = torch.cat((X.reshape(-1,1), T.reshape(-1,1)),dim=1).to(self.device)
        xt.requires_grad_(True)

        # Networks
        psi_model = FCN(2,self.hidden_dim,self.hidden_layers,self.activation).to(self.device)
        phi_model = FCN(2,self.hidden_dim,self.hidden_layers,self.activation).to(self.device)
        params = list(psi_model.parameters()) + list(phi_model.parameters())

        # Optimizer
        if self.optimizer_type.lower() == "adam":
            optimizer = torch.optim.Adam(params, lr=self.lr)
        elif self.optimizer_type.lower() == "sgd":
            optimizer = torch.optim.SGD(params, lr=self.lr, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_type}")

        # Utilities
        def psi_initial(xt):
            x_np = xt[:,0:1].detach().cpu().numpy()
            psi0_val = self.psi0_func(x_np)
            return torch.tensor(np.real(psi0_val),dtype=torch.float32,device=self.device)

        def V_eval(x,t):
            if self.V_func is None:
                return torch.zeros_like(x)
            x_np,t_np = x.detach().cpu().numpy(), t.detach().cpu().numpy()
            V_np = self.V_func(x_np,t_np)
            return torch.tensor(V_np,dtype=torch.float32,device=self.device)

        def total_loss(xt):
            psi_re, psi_im = psi_model(xt)
            phi_re,_ = phi_model(xt)
            psi_re_t, psi_re_xx = derivatives(psi_re, xt)
            psi_im_t, psi_im_xx = derivatives(psi_im, xt)
            _, phi_xx = derivatives(phi_re, xt)

            V = V_eval(xt[:,0:1], xt[:,1:2])
            SN_re = self.hbar*psi_im_t + (self.hbar**2/(2*self.m))*psi_re_xx - self.m*phi_re*psi_re - V*psi_re
            SN_im = -self.hbar*psi_re_t + (self.hbar**2/(2*self.m))*psi_im_xx - self.m*phi_re*psi_im - V*psi_im

            PD = psi_re**2 + psi_im**2
            PEq = phi_xx - 4*np.pi*self.G*self.m*PD if self.enable_SN else phi_xx*0

            # Initial Condition at t = 0
            xt0 = xt[xt[:,1] < 1e-6]  # slice at t ≈ 0
            if len(xt0) > 0:
                psi0_re, psi0_im = psi_model(xt0)
                psi_target = psi_initial(xt0)
                
                # If psi0_func is real, psi_target_im = 0
                psi_target_im = torch.zeros_like(psi_target)
                
                loss_IC = torch.mean((psi0_re - psi_target)**2 + (psi0_im - psi_target_im)**2)
            else:
                loss_IC = torch.tensor(0.0, device=self.device)
            

            # Normalisation Condition for all t
            if self.enforce_norm:
                psi_re_all, psi_im_all = psi_model(xt)
                PD_all = psi_re_all**2 + psi_im_all**2  # shape: (Nx * Nt, 1)
                
                # dx for integration along x
                dx = (self.x_range[1] - self.x_range[0]) / (self.nx - 1)
                
                # Reshape PD to (Nx, Nt) for easier integration along x
                PD_grid = PD_all.view(self.nx, self.nt)
                
                # Integrate over x for each time slice, then compare to 1
                norm_error = PD_grid.sum(dim=0) * dx - 1.0
                
                # Average squared error over all time slices
                loss_norm = torch.mean(norm_error**2)
            else:
                loss_norm = torch.tensor(0.0, device=self.device)

            loss_SN = torch.mean(SN_re**2 + SN_im**2)
            loss_PEq = torch.mean(PEq**2)
            
            total = self.lambda_SN*loss_SN + self.lambda_PEq*loss_PEq + self.lambda_IC*loss_IC + self.lambda_norm*loss_norm
            return total, PD

        # Training
        PD_snapshots, loss_history, runtime_history = [], [], []
        interval = max(1,self.epochs//10)
        for epoch in tqdm(range(self.epochs),desc="Training PINN",unit=" epoch"):
            start_epoch = time.time()
            optimizer.zero_grad()
            L, PD = total_loss(xt)
            L.backward()
            optimizer.step()
            loss_history.append(L.item())
            runtime_history.append(time.time()-start_epoch)
            if epoch%interval==0 or epoch==self.epochs-1:
                tqdm.write(f"Epoch {epoch}: Loss={L.item():.6e}")
                PD_snapshots.append(PD.detach().cpu().numpy().reshape(self.nx,self.nt))

        # Final wavefunction & export
        psi_re, psi_im = psi_model(xt)
        psi_re_np = psi_re.detach().cpu().numpy().flatten()
        psi_im_np = psi_im.detach().cpu().numpy().flatten()
        psi_np = psi_re_np + 1j*psi_im_np
        PD_np = np.abs(psi_np)**2
        x_np = xt[:,0:1].detach().cpu().numpy().flatten()
        t_np = xt[:,1:2].detach().cpu().numpy().flatten()

        if self.save_data:
            data = np.column_stack([x_np, t_np, psi_re_np, psi_im_np, psi_np, PD_np])
            data_file = os.path.join(self.output_dir,"psi_xt.dat")
            np.savetxt(data_file,data,header="x t Re(psi) Im(psi) psi PD",fmt="%s")
            print(f"Final data saved to: {data_file}")

        # -----------------------------
        # 1) Multi-column grid of all epochs
        # -----------------------------
        if self.grid_plot:
            n_snap = len(PD_snapshots)
            n_rows = int(np.ceil(n_snap/self.grid_cols))
            fig, axes = plt.subplots(n_rows, self.grid_cols,
                                     figsize=(self.grid_cols*self.grid_cell_size[0],
                                              n_rows*self.grid_cell_size[1]),
                                     squeeze=False)
            for idx, PD_grid in enumerate(PD_snapshots):
                r,c = divmod(idx,self.grid_cols)
                ax = axes[r,c]
                for j in np.linspace(0,self.nt-1,min(5,self.nt),dtype=int):
                    ax.plot(np.linspace(*self.x_range,self.nx), PD_grid[:,j], label=f"t={t_train[j].item():.2f}")
                ax.set_title(f"Epoch {idx*interval}")
                if self.xlim is not None:
                    ax.set_xlim(*self.xlim)
                if self.ylim is not None:
                    ax.set_ylim(*self.ylim)
                ax.set_xlabel("x")
                ax.set_ylabel("|ψ|²")
                ax.legend(fontsize=6)
            # Hide empty axes
            for idx2 in range(len(PD_snapshots),n_rows*self.grid_cols):
                r,c = divmod(idx2,self.grid_cols)
                axes[r,c].axis('off')
            plt.tight_layout()
            multi_grid_file = os.path.join(self.output_dir,"grid_plot.png")
            plt.savefig(multi_grid_file)
            plt.close(fig)
            print(f"Multi-column epoch grid saved to: {multi_grid_file}")

        # -----------------------------
        # 2) GIF of evolving single subplot
        # -----------------------------
        if self.save_gif and PD_snapshots:
            gif_files = []
            for idx, PD_grid in enumerate(PD_snapshots):
                fig, ax = plt.subplots(figsize=(self.grid_cell_size[0], self.grid_cell_size[1]))
                for j in np.linspace(0,self.nt-1,min(5,self.nt),dtype=int):
                    ax.plot(np.linspace(*self.x_range,self.nx), PD_grid[:,j], label=f"t={t_train[j].item():.2f}")
                ax.set_title(f"Epoch {idx*interval}")
                if self.xlim is not None:
                    ax.set_xlim(*self.xlim)
                if self.ylim is not None:
                    ax.set_ylim(*self.ylim)
                ax.set_xlabel("x")
                ax.set_ylabel("|ψ|²")
                ax.legend()
                plt.tight_layout()
                png_file = os.path.join(self.output_dir,f"gif_epoch_{idx*interval}.png")
                plt.savefig(png_file)
                plt.close(fig)
                gif_files.append(png_file)
            gif_file = os.path.join(self.output_dir,"evolution.gif")
            save_gif_PIL(gif_file,gif_files,fps=2)
            print(f"GIF saved to: {gif_file}")

        return psi_model, loss_history, runtime_history

    @staticmethod
    def help():
        print("""
SNSolver.help()

Solves the time-dependent Schrödinger–Newton equation using a PINN.

Arguments:
  psi0_func : callable - Initial wavefunction ψ(x,0)
  V_func : callable, optional - External potential V(x,t)
  enable_SN : bool - Enable Schrödinger–Newton coupling
  enforce_norm : bool - Enforce normalization ∫|ψ|²dx=1
  bc_type : str - Boundary condition type
  x_range, t_range : tuple - Domain ranges
  nx, nt : int - Grid points
  hidden_dim, hidden_layers : int - Neural network size
  activation : str - Activation
  optimizer_type : str - 'adam' or 'sgd'
  lr : float - Learning rate
  epochs : int - Number of epochs
  hbar, G, m : float - Physical constants
  lambda_SN, lambda_PEq, lambda_IC, lambda_norm : float - Loss weights
  loss_tol : float - Early stopping
  show_plot, save_plot, save_gif, save_data, save_model : bool - Output control
  grid_plot : bool - Show grid subplot figure
  grid_cols : int - Columns in multi-grid
  grid_cell_size : tuple - Subplot size
  xlim, ylim: tuple - Plot limit on x and y axis
  output_dir : str - Directory
  seed : int
  device : str - 'cpu','cuda','mps'

Returns:
  psi_model : trained model
  loss_history : list
  runtime_history : list
""")
