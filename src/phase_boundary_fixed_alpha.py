# Verify the analytical derivation of transition boundary
# For figure 4 (b)
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import root_scalar
from scipy.stats import norm as normal_dist
from mpl_toolkits.mplot3d import Axes3D
import random

# Mutual information of context
def I_context(alpha, fraction, p):
    M = len(fraction)
    H = 0
    coef1 = np.zeros(M)
    coef2 = np.zeros(M)
    for s in range(M):
        coef1[s] = (1+alpha) * p[s] / (1+alpha*fraction[s])
        coef2[s] = p[s] / (1+alpha*fraction[s])
    for s in range(M):
        tmp = 0
        for r in range(M):
            tmp += coef2[r]
        tmp -= coef2[s]
        H += fraction[s] * (coef1[s] + tmp) * np.log(coef1[s] + tmp)
    H += (1-np.sum(fraction)) * (np.sum(coef2)) * np.log(np.sum(coef2))
    I = 0
    for s in range(M):
        I += fraction[s] * coef1[s] * np.log(coef1[s] / p[s]) + (1-fraction[s]) * coef2[s] * np.log(coef2[s] / p[s])
    return I - H

# Mutual information of space
def I_space(alpha, fraction, p):
    M = len(fraction)
    I = 0
    for s in range(M):
        I -= fraction[s] * (1+alpha) * p[s] / (1 + alpha * fraction[s]) * np.log(1+alpha)
    return I

# Total information
def tilde_I(alpha, W, fraction, p):
    return W * I_space(alpha, fraction, p) +  I_context(alpha, fraction, p)

# Transition boundary between two regimes on the phase plane of alpha and W
def critical_W(alpha):
    return 1 - alpha / ((1+alpha)*np.log(1+alpha))

# Solve the optimization problem with given parameters
def optimize_fraction(alpha, W, p):
    N = len(p)

    if N == 1:
        return np.array([0.0])

    fraction0 = np.ones(N-1) / (N-1)

    bounds = [(0.0, 1.0) for _ in range(N-1)]

    # constraint: sum of f must be less than 1
    constraints = {
        'type': 'ineq',
        'fun': lambda b: 1.0 - np.sum(b)
    }

    result = minimize(
        lambda b: -tilde_I(alpha, W, np.concatenate(([0.0], b)), p),
        fraction0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000}
    )

    if not result.success:
        print("Warning: optimization did not converge")

    fraction_opt = np.concatenate(([0.0], result.x))
    print(result)
    print("p:", p)

    return fraction_opt

# Transition boundary between two regimes on the phase plane of alpha and W
def critical_W(alpha):
    return 1 - alpha / ((1+alpha)*np.log(1+alpha))

# Number of contexts
M_list = [10]
# Parameters
alpha_list = np.linspace(0, 1.0, 101)
W_list = np.linspace(0, 0.3, 100)
alpha_finelist = np.linspace(0, 1.0, 1001)
# Number of trials
num_trials = 100
rng = np.random.default_rng(0)
alpha_val = 0


for M in M_list:
    # fraction_max
    fraction_max_grid = np.zeros((len(alpha_list), len(W_list)))
    alpha_fix = np.zeros(len(W_list))
    for i, alpha in enumerate(alpha_list):
        if(i == 20 or i==50 or i == 80):
            alpha_val = alpha
            for j, W in enumerate(W_list):
                fraction_maxs = []
                for _ in range(num_trials):
                    p = rng.dirichlet(np.ones(M_list[0]))
                    p = np.sort(p)

                    fraction = optimize_fraction(alpha, W, p)

                    fraction_maxs.append(np.max(fraction))
                    print(f"alpha={alpha:.3f}, W={W:.3f}")
                fraction_max_grid[i, j] = np.mean(fraction_maxs)
                print(f"alpha={alpha:.3f}, W={W:.3f}, fraction_max={fraction_max_grid[i, j]:.6f}")
                alpha_fix[j] = np.mean(fraction_maxs)

    # # Transition boundary obtained analytically
    # critical_W_values = [critical_W(alpha) for alpha in alpha_finelist]

    # fig, ax = plt.subplots(figsize=(12, 8))
    # masked = np.ma.masked_where(np.log(fraction_max_grid) <= -5, np.log(fraction_max_grid))
    # im = ax.imshow(masked, extent=[W_list[0], W_list[-1], alpha_list[0], alpha_list[-1]], 
    #                aspect='auto', cmap='viridis', origin='lower')
    # cbar = plt.colorbar(im, ax=ax, label='f_max')
    # cbar.set_label('log(f_max)', fontsize=20)
    # cbar.ax.tick_params(labelsize=16)

    # ax.plot(critical_W_values, alpha_finelist, 'r-', linewidth=2.5, label='critical_W(alpha)', zorder=5)

    # ax.set_xlabel('W', fontsize=20)
    # ax.set_ylabel('alpha', fontsize=20)
    # ax.set_title(f'f_max on alpha-W plane (M={M}, num_trials={num_trials})', fontsize=20)

    # ax.legend(fontsize=13)
    # ax.tick_params(labelsize=16)
    # ax.grid(True, alpha=0.3)

    # plt.tight_layout()
    # plt.savefig(f"./figure/alpha_W_heatmap_M{M}.png", dpi=150, bbox_inches='tight')
    # plt.show()
    # plt.close()


            plt.figure(figsize=(16, 3))

            plt.xlabel("W", fontsize=25)
            plt.ylabel(r"$f_{max}$", fontsize=25)
            plt.xlim(0,0.3)
            plt.ylim(0,0.35)
            plt.plot(W_list, alpha_fix, linewidth=2.5)
            plt.vlines(critical_W(alpha_val), 0, 0.35, linewidth=2.5, colors="red", linestyles='dashed')
            # plt.grid(True)
            # plt.legend(fontsize=16)
            plt.subplots_adjust(left=0.1, right=0.9, bottom=0.15, top=0.9)
            plt.tick_params(labelsize=28)
            plt.savefig(f"./figure/W_fmax_alpha{alpha_val:.2f}.pdf")
            plt.close()
