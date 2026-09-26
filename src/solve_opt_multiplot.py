# Solve the optimization problem with several sets of (alpha, W, M) with multi plot
# For figure 3
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import root_scalar
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
alpha_list = [0.9, 0.5, 0.1]
W_list = [0.01, 0.1, 0.2, 0.3]
# For plot
colors = ["red", "blue"]
num_trials = 10
rng = np.random.default_rng(0)

fig, axes = plt.subplots(len(alpha_list),len(W_list), figsize=(15, 10), constrained_layout=True)
plt.subplots_adjust(
    left=0.1, right=0.95, bottom=0.1, top=0.95, wspace=0.4, hspace=0.4
)
tmp = 0

for alpha in alpha_list:
    for M in M_list:
        for W in W_list:
            fraction_list = []
            p_list = []
            for _ in range(num_trials):
                p = rng.dirichlet(np.ones(M))
                p = np.sort(p)

                fraction = optimize_fraction(alpha, W, p)

                p_list.append(p)
                fraction_list.append(fraction)

            i = int(tmp / len(W_list))
            j = tmp % len(W_list)
            ax = axes[i][j]
            if(j == 0 and i == 1):
                ax.yaxis.set_label_coords(-0.4, 0.5)
                # ax.set_ylabel("Engram cell fraction", fontsize=25)
            if(i == len(alpha_list)-1 and j == 1):
                ax.xaxis.set_label_coords(1.15, -0.3)
                # ax.set_xlabel("Occurrence probability", fontsize=25)
            for k, (p, fraction) in enumerate(zip(p_list, fraction_list)):
                if W <= critical_W(alpha):
                    ax.plot(p, fraction, marker='o', color=colors[0])
                else:
                    ax.plot(p, fraction, marker='o', color=colors[1])
            # ax.set_title(f"alpha={alpha}, W={W}", fontsize=15)
            ax.set_ylim(0, 0.40)
            ax.set_xlim(-0.05, 0.55)
            ax.tick_params(labelsize=20)

            # plt.figure(figsize=(10, 8))
            # for j, (p, fraction) in enumerate(zip(p_list, fraction_list)):
            #     if W <= critical_W(alpha):
            #         plt.plot(p, fraction, marker='o', color=colors[0])
            #     else:
            #         plt.plot(p, fraction, marker='o', color=colors[1])

            # plt.xlabel("Occurrence probability", fontsize=25)
            # plt.ylabel("Engram cell fraction", fontsize=25)
            # plt.title(f"alpha={alpha}, W={W}", fontsize=25)
            # plt.grid(True)
            # plt.legend(fontsize=16)
            # plt.tick_params(labelsize=20)
            # plt.ylim(0, 0.35)
            # # plt.show()
            # plt.savefig(f"./multicontext_homecage/fraction_p_alpha{alpha}_W{W}_M{M}.pdf")
            # plt.close()
            tmp += 1
plt.savefig(f"./figure/fig3.pdf")
