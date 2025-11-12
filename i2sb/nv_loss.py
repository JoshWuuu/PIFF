import os
import random
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from natsort import natsorted

def _cdx(f, dx, bc="periodic"):
    if bc == "periodic":
        return (np.roll(f, -1, axis=2) - np.roll(f, 1, axis=2)) / (2.0*dx)
    elif bc == "neumann0":
        g = f.copy()
        g[:, 0]    = g[:, 1]
        g[:, -1]   = g[:, -2]
        return (np.roll(g, -1, axis=2) - np.roll(g, 1, axis=2)) / (2.0*dx)
    else:
        raise ValueError("bc must be 'periodic' or 'neumann0'")

def _cdy(f, dy, bc="periodic"):
    if bc == "periodic":
        return (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2.0*dy)
    elif bc == "neumann0":
        g = f.copy()
        g[0, :]    = g[1, :]
        g[-1, :]   = g[-2, :]
        return (np.roll(g, -1, axis=1) - np.roll(g, 1, axis=1)) / (2.0*dy)
    else:
        raise ValueError("bc must be 'periodic' or 'neumann0'")

def _lap(f, dx, dy, bc="periodic"):
    if bc == "periodic":
        d2x = (np.roll(f, -1, 2) - 2.0*f + np.roll(f, 1, 2)) / (dx*dx)
        d2y = (np.roll(f, -1, 1) - 2.0*f + np.roll(f, 1, 1)) / (dy*dy)
    elif bc == "neumann0":
        g = f.copy()
        # duplicate interior values to form zero-normal-gradient ghosts
        g[:, 0]  = g[:, 1]
        g[:, -1] = g[:, -2]
        g[0, :]  = g[1, :]
        g[-1, :] = g[-2, :]
        d2x = (np.roll(g, -1, 1) - 2.0*g + np.roll(g, 1, 1)) / (dx*dx)
        d2y = (np.roll(g, -1, 0) - 2.0*g + np.roll(g, 1, 0)) / (dy*dy)
    else:
        raise ValueError("bc must be 'periodic' or 'neumann0'")
    return d2x + d2y

def navier_stokes_operators(prev_u, cur_u, prev_v, cur_v,
                            rho=1000, dx=20.0, dy=20.0, dt=3600.0, nu=1e-6,
                            p=None, g=9.81, h=None,
                            bc="neumann0"):
    """
    Returns (Lu, Lv, Lg) on a collocated grid using central differences.

    Required:
      prev_u, cur_u, prev_v, cur_v : 2D arrays (Ny, Nx)
      rho : density
      nu  : kinematic viscosity (mu/rho)
      dx, dy, dt : spacings

    Pressure:
      Provide either p (pressure field) OR g and h (so p = rho*g*h).
    """
    if p is None:
        if g is None or h is None:
            raise ValueError("Provide either p, or g and h (for p = rho*g*h).")
        p = g * h
        p = p.astype(np.float64)

    # Time derivatives (backward Euler / first-order backward difference)
    du_dt = (cur_u - prev_u) / dt
    dv_dt = (cur_v - prev_v) / dt

    # Spatial derivatives
    ux = _cdx(cur_u, dx, bc); uy = _cdy(cur_u, dy, bc)
    vx = _cdx(cur_v, dx, bc); vy = _cdy(cur_v, dy, bc)

    px = _cdx(p, dx, bc);     py = _cdy(p, dy, bc)

    lap_u = _lap(cur_u, dx, dy, bc)
    lap_v = _lap(cur_v, dx, dy, bc)

    # Convective terms (u·∇u, u·∇v)
    conv_u = cur_u * ux + cur_v * uy

    conv_v = cur_u * vx + cur_v * vy

    # Operators
    Lu = du_dt + conv_u + (1.0/rho)*px - nu*lap_u
    Lv = dv_dt + conv_v + (1.0/rho)*py - nu*lap_v
    # Lg = ux + vy
    Lg = 0
    return Lu, Lv, Lg

def navier_stokes_operators_torch(prev_u, prev_v, cur_u, cur_v, h, rho=1000, 
                                  dx=20.0, dy=20.0, dt=3600.0, nu=1e-6, g=9.81):
    """
    prev_u, prev_v, cur_u, cur_v: (B,1,H,W) torch tensors
    h: total height field (B,H,W) torch tensor
    Returns lu, lv, lg each (B,H,W) as simple residuals:
      lu ~ time_derivative(u) + u*u_x + v*u_y - nu*Laplacian(u)
      lv ~ time_derivative(v) + u*v_x + v*v_y - nu*Laplacian(v)
      lg ~ divergence(u,v) (incompressibility)
    """
    device = cur_u.device
    dtype = cur_u.dtype

    p = g * h
    # 3x3 finite-diff kernels
    # central differences for first derivatives, 5-point Laplacian
    kx = torch.tensor([[0., 0., 0.],
                       [-0.5, 0., 0.5],
                       [0., 0., 0.]], device=device, dtype=dtype).view(1,1,3,3)
    ky = torch.tensor([[0., -0.5, 0.],
                       [0.,   0., 0.],
                       [0.,  0.5, 0.]], device=device, dtype=dtype).view(1,1,3,3)
    lap = torch.tensor([[0.,  1., 0.],
                        [1., -4., 1.],
                        [0.,  1., 0.]], device=device, dtype=dtype).view(1,1,3,3)

    padding = 1

    def deriv_x(t, dx):  # (B,1,H,W) -> (B,1,H,W)
        return F.conv2d(t, kx, padding=padding) / dx
    def deriv_y(t, dy):
        return F.conv2d(t, ky, padding=padding) / dy
    def laplacian(t, dx=1.0, dy=1.0):
        return F.conv2d(t, lap, padding=padding) / (dx*dx)  # assume dx=dy

    u = cur_u
    v = cur_v

    u = u.unsqueeze(1) if u.dim() == 3 else u  # (B,1,H,W)
    v = v.unsqueeze(1) if v.dim() == 3 else v
    prev_u = prev_u.unsqueeze(1) if prev_u.dim() == 3 else prev_u
    prev_v = prev_v.unsqueeze(1) if prev_v.dim() == 3 else prev_v
    p = p.unsqueeze(1) if p.dim() == 3 else p

    # derivatives
    u_x = deriv_x(u, dx); u_y = deriv_y(u, dy)
    v_x = deriv_x(v, dx); v_y = deriv_y(v, dy)
    Δu  = laplacian(u, dx, dy); Δv  = laplacian(v, dx, dy)

    p_x = deriv_x(p, dx)  
    p_y = deriv_y(p, dy)

    # simple time derivative (backward Euler)
    du_dt = (u - prev_u) / dt
    dv_dt = (v - prev_v) / dt

    # advection terms
    adv_u = u * u_x + v * u_y
    adv_v = u * v_x + v * v_y

    # pressure term
    total_p_u = (1.0/rho) * p_x
    total_p_v = (1.0/rho) * p_y

    # viscosity
    visc_u = -nu * Δu
    visc_v = -nu * Δv

    # residuals (strip the channel dim to (B,H,W))
    lu = (du_dt + adv_u + total_p_u + visc_u).squeeze(1)
    lv = (dv_dt + adv_v + total_p_v + visc_v).squeeze(1)

    # divergence constraint as "lg"
    div = u_x + v_y  # (B,1,H,W)
    lg = div.squeeze(1)

    # optional: weight by height h if desired (e.g., shallow-water coupling)
    # lu = lu * (1 + 0.0*h)
    # lv = lv * (1 + 0.0*h)

    return lu, lv, lg

import torch
import torch.nn.functional as F

def manning_velocity_torch(cur_h, bed_elevation, manning_n, 
                           dx=20.0, dy=20.0, k_manning=1.0, epsilon=1e-6):
    """
    Calculates 2D velocity components based on the Manning equation.

    Assumes the friction slope is equal to the water surface slope.
    Assumes the hydraulic radius is equal to the water depth (h).

    Args:
        cur_h (torch.Tensor): Current water depth field (B,H,W) or (B,1,H,W).
        bed_elevation (torch.Tensor): Bed elevation field (B,H,W) or (B,1,H,W).
        manning_n (float or torch.Tensor): Manning's roughness coefficient. 
                                           Can be a scalar or a field (B,H,W).
        dx (float): Grid spacing in the x-direction.
        dy (float): Grid spacing in the y-direction.
        k_manning (float): Unit conversion factor (1.0 for SI, 1.49 for English).
        epsilon (float): Small value to prevent division by zero in flat areas.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: cur_ux, cur_uy as (B,H,W) tensors.
    """
    device = cur_h.device
    dtype = cur_h.dtype

    # 3x3 finite-diff kernels for central differences
    kx = torch.tensor([[0., 0., 0.],
                       [-0.5, 0., 0.5],
                       [0., 0., 0.]], device=device, dtype=dtype).view(1,1,3,3)
    ky = torch.tensor([[0., -0.5, 0.],
                       [0.,   0., 0.],
                       [0.,  0.5, 0.]], device=device, dtype=dtype).view(1,1,3,3)
    
    padding = 1

    def deriv_x(t, dx):
        return F.conv2d(t, kx, padding=padding) / dx
    def deriv_y(t, dy):
        return F.conv2d(t, ky, padding=padding) / dy

    # --- Ensure inputs are (B,1,H,W) for conv2d ---
    h = cur_h.unsqueeze(1) if cur_h.dim() == 3 else cur_h
    zb = bed_elevation.unsqueeze(1) if bed_elevation.dim() == 3 else bed_elevation

    # 1. Calculate Water Surface Elevation (eta)
    eta = zb + h

    # 2. Calculate Water Surface Slope (approximates friction slope)
    # The negative sign indicates that flow is down the gradient.
    s_x = -deriv_x(eta, dx)
    s_y = -deriv_y(eta, dy)

    eps = 1e-8
    n_safe   = torch.as_tensor(manning_n, device=h.device, dtype=h.dtype).clamp_min(eps)
    rh_safe  = h.clamp_min(0.0)                 # avoid negative depth
    s_safe    = torch.sqrt(s_x**2 + s_y**2 + 1e-8)      # true magnitude
    # s_safe   = s_mag.clamp_min(eps)             # avoid /0 and 0**(1/2)

    speed = (k_manning / n_safe) * (rh_safe + eps)**(2.0/3.0) * s_safe**0.5
    u_dir_x, u_dir_y = s_x / s_safe, s_y / s_safe
    ux, uy = speed * u_dir_x, speed * u_dir_y

    # optionally zero out in nearly-dry or flat cells
    mask = ((h > 0.0) & (s_safe > 0.0)).to(h.dtype)
    ux, uy = ux * mask, uy * mask

    return ux, uy

def continuity_residual_torch(prev_h, cur_h, cur_ux, cur_uy, elevation, rainfall,
                              dx=20.0, dy=20.0, dt=3600.0):
    """
    Calculates the residual of the 2D Shallow Water continuity (mass conservation) equation.
    
    prev_h, cur_h: (B,H,W) or (B,1,H,W) torch tensors for water depth at prev/current time
    cur_ux, cur_uy: (B,H,W) or (B,1,H,W) torch tensors for x and y velocities at current time
    
    Returns lh (B,H,W) as the residual:
      lh ~ time_derivative(h) + divergence(h*u)
    """
    device = cur_h.device
    dtype = cur_h.dtype


    # cur_ux, cur_uy = manning_velocity_torch(cur_h, bed_elevation=elevation, manning_n=0.03)
    # 3x3 finite-diff kernels for central differences
    kx = torch.tensor([[0., 0., 0.],
                       [-0.5, 0., 0.5],
                       [0., 0., 0.]], device=device, dtype=dtype).view(1,1,3,3)
    ky = torch.tensor([[0., -0.5, 0.],
                       [0.,   0., 0.],
                       [0.,  0.5, 0.]], device=device, dtype=dtype).view(1,1,3,3)

    padding = 1

    def deriv_x(t, dx):  # (B,1,H,W) -> (B,1,H,W)
        return F.conv2d(t, kx, padding=padding) / dx
    def deriv_y(t, dy):
        return F.conv2d(t, ky, padding=padding) / dy

    # --- Ensure all inputs are (B,1,H,W) for conv2d ---
    h = cur_h.unsqueeze(1) if cur_h.dim() == 3 else cur_h
    prev_h = prev_h.unsqueeze(1) if prev_h.dim() == 3 else prev_h
    ux = cur_ux.unsqueeze(1) if cur_ux.dim() == 3 else cur_ux
    uy = cur_uy.unsqueeze(1) if cur_uy.dim() == 3 else cur_uy
    rainfall = rainfall.unsqueeze(1).unsqueeze(1).unsqueeze(1)

    # --- Calculate residual terms ---

    # 1. Time derivative: (h_cur - h_prev) / dt
    dh_dt = (h - prev_h) / dt

    # 2. Flux divergence: d(h*ux)/dx + d(h*uy)/dy
    # First, compute the fluxes (discharge per unit width)
    q_x = h * ux  # (B,1,H,W)
    q_y = h * uy  # (B,1,H,W)

    # Now, compute the divergence of the flux
    dqx_dx = deriv_x(q_x, dx)
    dqy_dy = deriv_y(q_y, dy)
    
    div_q = dqx_dx + dqy_dy

    rainfall = rainfall * 2.77778e-7  # convert mm/hr to m/s

    # --- Combine terms for the final residual ---
    # lh = dh/dt + div(q)
    lh = (dh_dt + div_q - rainfall).squeeze(1)  # Squeeze channel dim to get (B,H,W)

    return lh

def _mk_grid(Nx, Ny):
    x = np.linspace(0.0, 1.0, Nx, endpoint=False)
    y = np.linspace(0.0, 1.0, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y)
    return X, Y, (x[1]-x[0]), (y[1]-y[0])

def _manufactured_fields(X, Y, rho=1.0, g=9.81):
    """
    Smooth periodic manufactured solution:
      ψ = sin(2πx) sin(2πy)
      u =  ∂ψ/∂y = 2π cos(2πy) sin(2πx)
      v = -∂ψ/∂x = -2π cos(2πx) sin(2πy)
      h = sin(2πx) cos(2πy)  (so p = rho g h)
    This choice is divergence-free analytically: ∂u/∂x + ∂v/∂y = 0
    """
    twopi = 2.0*np.pi
    u =  twopi * np.cos(twopi*Y) * np.sin(twopi*X)
    v = -twopi * np.cos(twopi*X) * np.sin(twopi*Y)
    h =  np.sin(twopi*X) * np.cos(twopi*Y)
    p = rho * g * h
    return u, v, p, h

def _analytic_residuals(X, Y, u, v, p, *, rho, nu):
    """
    Compute the *continuous* Lu, Lv, Lg from closed-form derivatives.
      Lu = u ux + v uy + (1/rho) px - nu (uxx + uyy)
      Lv = u vx + v vy + (1/rho) py - nu (vxx + vyy)
      Lg = ux + vy
    """
    twopi = 2.0*np.pi
    # handy trig aliases
    sx, cx = np.sin(twopi*X), np.cos(twopi*X)
    sy, cy = np.sin(twopi*Y), np.cos(twopi*Y)

    # For our chosen fields:
    # u =  2π cy sx
    # v = -2π cx sy
    # p =  rho g (sx cy)

    # First derivatives
    ux  =  twopi**2 * ( -sy * sx )        # d/dx of (2π cy sx) = 2π cy * 2π cx? careful—derive generally:
    # Let's re-derive cleanly using the actual u, v expressions:
    # u = 2π * cy * sx
    ux  =  2.0*np.pi * (cy * (twopi*cx))               # du/dx
    uy  =  2.0*np.pi * ((-twopi*sy) * sx)              # du/dy
    vx  = -2.0*np.pi * ((-twopi*sx) * sy)              # dv/dx
    vy  = -2.0*np.pi * (cx * (twopi*cy))               # dv/dy

    # Second derivatives
    uxx =  2.0*np.pi * (cy * (-(twopi**2)*sx))         # d/dx(ux)
    uyy =  2.0*np.pi * ((-(twopi**2)*cy) * sx)         # d/dy(uy)
    vxx = -2.0*np.pi * ((-(twopi**2)*cx) * sy)
    vyy = -2.0*np.pi * (cx * (-(twopi**2)*sy))

    # Pressure derivatives with p = rho g (sx cy)
    px  = rho * ( (twopi*cx) * cy ) * 1.0 * 9.81 / 9.81  # keep g symbolic below; simpler: compute with g explicitly
    # Let's recompute px, py including g explicitly to avoid cancellation assumptions:
    g = 9.81
    px = rho * g * (twopi*cx) * cy
    py = rho * g * (sx) * (-(twopi*sy))

    # Residuals
    Lu = u*ux + v*uy + (1.0/rho)*px - nu*(uxx + uyy)
    Lv = u*vx + v*vy + (1.0/rho)*py - nu*(vxx + vyy)
    Lg = ux + vy
    return Lu, Lv, Lg

def _norms(err):
    l2  = np.sqrt(np.mean(err**2))
    linf = np.max(np.abs(err))
    return l2, linf

def test_navier_stokes_operators_convergence(rho=1.0, nu=1e-3, dt=1.0, sizes=(32, 64, 128), bc="periodic"):
    """
    Convergence test under grid refinement for periodic BCs.
    Uses a steady manufactured solution with prev=cur, so ∂/∂t term = 0.
    Verifies that discrete (Lu, Lv, Lg) match analytic residuals with ~O(Δx^2) error.
    Prints L2/L∞ errors and estimated convergence rates.
    """
    errs_u, errs_v, errs_g = [], [], []
    dxs = []

    for N in sizes:
        Nx = Ny = N
        X, Y, dx, dy = _mk_grid(Nx, Ny)
        dxs.append(dx)

        # Manufactured fields
        u, v, p, h = _manufactured_fields(X, Y, rho=rho)
        # Steady test: prev = cur eliminates time derivative
        prev_u = u.copy(); cur_u = u.copy()
        prev_v = v.copy(); cur_v = v.copy()

        # Discrete operators
        Lu_h, Lv_h, Lg_h = navier_stokes_operators(
            prev_u, cur_u, prev_v, cur_v,
            rho=rho, nu=nu, dx=dx, dy=dy, dt=dt,
            p=p, bc=bc
        )

        # Analytic (continuous) residuals
        Lu_c, Lv_c, Lg_c = _analytic_residuals(X, Y, u, v, p, rho=rho, nu=nu)

        # Errors
        eu = Lu_h - Lu_c
        ev = Lv_h - Lv_c
        eg = Lg_h - Lg_c

        errs_u.append(_norms(eu))
        errs_v.append(_norms(ev))
        errs_g.append(_norms(eg))

    def _rates(vals, dxs):
        # vals: list of (l2, linf) across refinements
        rates_l2, rates_linf = [], []
        for k in range(1, len(vals)):
            r_l2   = np.log(vals[k-1][0]/vals[k][0])   / np.log(dxs[k-1]/dxs[k])
            r_linf = np.log(vals[k-1][1]/vals[k][1])   / np.log(dxs[k-1]/dxs[k])
            rates_l2.append(r_l2); rates_linf.append(r_linf)
        return rates_l2, rates_linf

    ru_l2, ru_linf = _rates(errs_u, dxs)
    rv_l2, rv_linf = _rates(errs_v, dxs)
    rg_l2, rg_linf = _rates(errs_g, dxs)

    print("Grid sizes:", sizes)
    print("\nLu errors (L2, Linf):")
    for N, e in zip(sizes, errs_u):
        print(f"  N={N:4d}: L2={e[0]:.3e}, Linf={e[1]:.3e}")
    if ru_l2:
        print("  Estimated order (between grids):")
        for i in range(len(ru_l2)):
            print(f"    {sizes[i]}→{sizes[i+1]}: order_L2≈{ru_l2[i]:.2f}, order_Linf≈{ru_linf[i]:.2f}")

    print("\nLv errors (L2, Linf):")
    for N, e in zip(sizes, errs_v):
        print(f"  N={N:4d}: L2={e[0]:.3e}, Linf={e[1]:.3e}")
    if rv_l2:
        print("  Estimated order (between grids):")
        for i in range(len(rv_l2)):
            print(f"    {sizes[i]}→{sizes[i+1]}: order_L2≈{rv_l2[i]:.2f}, order_Linf≈{rv_linf[i]:.2f}")

    print("\nLg (divergence) errors (L2, Linf):")
    for N, e in zip(sizes, errs_g):
        print(f"  N={N:4d}: L2={e[0]:.3e}, Linf={e[1]:.3e}")
    if rg_l2:
        print("  Estimated order (between grids):")
        for i in range(len(rg_l2)):
            print(f"    {sizes[i]}→{sizes[i+1]}: order_L2≈{rg_l2[i]:.2f}, order_Linf≈{rg_linf[i]:.2f}")

if __name__ == "__main__":
    # test_navier_stokes_operators_convergence()
    # B, Ny, Nx, C = 16, 256, 256, 1
    # prev_u = np.zeros((B, Ny, Nx))
    # cur_u  = np.zeros((B, Ny, Nx))
    # prev_v = np.zeros((B, Ny, Nx))
    # cur_v  = np.zeros((B, Ny, Nx))
    # h      = np.zeros((B, Ny, Nx))  # if you want to define p via rho*g*h

    # Lu, Lv, Lg = navier_stokes_operators(prev_u, cur_u, prev_v, cur_v,
    #                                     rho=1000.0, nu=0.002,
    #                                     dx=1/256, dy=1/256, dt=1e-3,
    #                                     g=9.81, h=h, bc="periodic")
    # print(Lu.shape, Lv.shape, Lg.shape)  # (16, 256, 256, 1)

    u_folder = "C:\\Users\\User\\Desktop\\dev\\new_train\\Vx"
    v_folder = "C:\\Users\\User\\Desktop\\dev\\new_train\\Vy"
    h_folder = "C:\\Users\\User\\Desktop\\dev\\new_train\\tainan_png"

    u = []
    prev_u = []
    v = []
    prev_v = []
    h = []
    prev_h = []

    u_filename = []
    v_filename = []
    h_filename = []
    prev_u_filename = []
    prev_v_filename = []
    prev_h_filename = []
    
    dem_height_path = "C:\\Users\\User\\Desktop\\dev\\new_train\\tainan_new.png"
    dem = cv2.imread(dem_height_path, cv2.IMREAD_UNCHANGED)[:,:,0]
    dem_height_norm = (dem - dem.min()) / (dem.max() - dem.min()) * 17.0
    dem_height = np.ceil(np.clip(dem_height_norm, 0, 17))

    rainfall_path = 'C:\\Users\\User\\Desktop\\dev\\new_train\\train.csv'
    rainfall = pd.read_csv(rainfall_path)
        # remove first row, no 0 row 
    rainfall = rainfall.iloc[:, :]
    rainfall_list = []
    rainfall_record = []
    for col in rainfall.columns:
        if col == "time":
            continue
        for row in range(len(rainfall)):
            cell_value = rainfall.iloc[row][col]
            rainfall_list.append(cell_value)
            rainfall_record.append(f"{col}_{row}")

    for RF_folder in natsorted(os.listdir(u_folder)):
        for filename in sorted(os.listdir(os.path.join(u_folder, RF_folder))):
            cur_u = cv2.imread(os.path.join(u_folder, RF_folder, filename), cv2.IMREAD_UNCHANGED).astype(np.float32)
            u_filename.append(os.path.join(u_folder, RF_folder, filename))
            cur_u = (cur_u - 127) / 32
            u.append(cur_u)
            if filename.endswith("000_00.png"):
                prev_u_filename.append(os.path.join(u_folder, RF_folder, filename))
                prev_u.append(cur_u)
            else:
                # RF_d_000_00.png get the 000 as integer
                prev_filename = filename.split("_")
                prev_filename[2] = str(int(prev_filename[2]) - 1).zfill(3)
                prev_filename = "_".join(prev_filename)
                cur_prev = cv2.imread(os.path.join(u_folder, RF_folder, prev_filename), cv2.IMREAD_UNCHANGED).astype(np.float32)
                cur_prev = (cur_prev - 127) / 32
                prev_u.append(cur_prev)
                prev_u_filename.append(os.path.join(u_folder, RF_folder, prev_filename))

    for RF_folder in natsorted(os.listdir(v_folder)):
        for filename in sorted(os.listdir(os.path.join(v_folder, RF_folder))):
            cur_v = cv2.imread(os.path.join(v_folder, RF_folder, filename), cv2.IMREAD_UNCHANGED).astype(np.float32)
            v_filename.append(os.path.join(v_folder, RF_folder, filename))
            cur_v = (cur_v - 127) / 32
            v.append(cur_v)
            if filename.endswith("000_00.png"):
                prev_v.append(cur_v)
                prev_v_filename.append(os.path.join(v_folder, RF_folder, filename))
            else:
                # RF_d_000_00.png get the 000 as integer
                prev_filename = filename.split("_")
                prev_filename[2] = str(int(prev_filename[2]) - 1).zfill(3)
                prev_filename = "_".join(prev_filename)
                cur_prev = cv2.imread(os.path.join(v_folder, RF_folder, prev_filename), cv2.IMREAD_UNCHANGED).astype(np.float32)
                cur_prev = (cur_prev - 127) / 32
                prev_v.append(cur_prev)
                prev_v_filename.append(os.path.join(v_folder, RF_folder, prev_filename))

    for RF_folder in natsorted(os.listdir(h_folder)):
        for filename in sorted(os.listdir(os.path.join(h_folder, RF_folder))):
            if filename.endswith("Max.png"):
                continue
            cur_h = cv2.imread(os.path.join(h_folder, RF_folder, filename), cv2.IMREAD_UNCHANGED).astype(np.float32)
            cur_h = (1-cur_h / 255) * 4
            # cur_h = cur_h + dem_height
            h.append(cur_h)
            h_filename.append(os.path.join(h_folder, RF_folder, filename))

            if filename.endswith("000_00.png"):
                prev_h.append(cur_h)
                prev_h_filename.append(os.path.join(h_folder, RF_folder, filename))
            else:
                # RF_d_000_00.png get the 000 as integer
                prev_filename = filename.split("_")
                prev_filename[2] = str(int(prev_filename[2]) - 1).zfill(3)
                prev_filename = "_".join(prev_filename)
                cur_prev = cv2.imread(os.path.join(h_folder, RF_folder, prev_filename), cv2.IMREAD_UNCHANGED).astype(np.float32)
                cur_prev = (1-cur_prev / 255) * 4
                # cur_prev = cur_prev + dem_height
                prev_h.append(cur_prev)
                prev_h_filename.append(os.path.join(h_folder, RF_folder, prev_filename))

    u = np.stack(u, axis=0)  # (B, H, W)
    v = np.stack(v, axis=0)
    h = np.stack(h, axis=0)
    prev_u = np.stack(prev_u, axis=0)
    prev_v = np.stack(prev_v, axis=0)
    prev_h = np.stack(prev_h, axis=0)
    rainfall = np.stack(rainfall_list, axis=0) # (B, H, W)

    print(u.shape, v.shape, h.shape, prev_u.shape, prev_v.shape, prev_h.shape, rainfall.shape)  # (B, H, W)

    # Lu, Lv, Lg = navier_stokes_operators(prev_u=prev_u, prev_v=prev_v, cur_u=u, cur_v=v, h=h)
    # # print(Lu.shape, Lv.shape, Lg.shape)  # (16, 256, 256, 1)
    # print("ns Lu norm:", np.linalg.norm(Lu, axis=(1,2)))
    # print("ns Lv norm:", np.linalg.norm(Lv, axis=(1,2)))

    # print("lu mse:", np.mean(Lu**2, axis=(1,2)))
    # print("lv mse:", np.mean(Lv**2, axis=(1,2)))
    # print("Lg norm:", np.linalg.norm(Lg, axis=(1,2)))

    lu = continuity_residual_torch(prev_h=torch.tensor(prev_h), 
                                  cur_h=torch.tensor(h),
                                    cur_ux=torch.tensor(u),
                                    cur_uy=torch.tensor(v),
                                    elevation=torch.tensor(dem_height.astype(np.float32)),
                                    rainfall=torch.tensor(rainfall.astype(np.float32)),
                                    dx=20.0, dy=20.0, dt=3600.0)
    print('mass equation-----------------')
    print("lh norm:", torch.norm(lu, dim=(1,2)))
    print("lu sum:", torch.sum(lu**2, dim=(1,2)))

    print("lh norm mean:", torch.mean(torch.norm(lu, dim=(1,2))))
    print("lu mean sum:", torch.mean(torch.sum(lu**2, dim=(1,2))))

    print("lu mean squared error:", torch.mean(torch.mean(lu**2, dim=(1,2))))


