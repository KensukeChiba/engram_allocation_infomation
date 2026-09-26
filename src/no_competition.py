# For the case of single context
# For figure S1
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

# tilde I
def f(alpha, fraction, p, w):
    return (1-w)*(1+alpha)*p*np.log(1+alpha) - (alpha**2*(1-p)*fraction**2+2*alpha*(1-p)*fraction+alpha*p+1)*np.log(1+alpha*p/(1+alpha*fraction-alpha*fraction*p))

# Solving optimization problem
def find_fraction_root(alpha, p, w, fraction_min=0.0, fraction_max=1.0):
    try:
        sol = root_scalar(
            lambda b: f(alpha, b, p, w),
            bracket=[fraction_min, fraction_max],
            method="bisect"
        )
        if sol.converged:
            return sol.root
        else:
            return 0
    except:
        return 0


# Setting
alphas = [0.1, 0.5, 0.9]
p_values = np.linspace(1/300, 1, 300)
params = [0.01, 0.05, 0.1]
# colors = []
# for i in range(len(alphas)):
#     colors.append(plt.cm.viridis(i / (len(alphas)-1)))
colors=['blue', 'red', 'green']


fig, axes = plt.subplots(1,3, figsize=(15, 7), constrained_layout=True)



for i, w in enumerate(params):
    j = i//5
    k = i%5
    ax = axes[k]
    fraction_roots = {}
    # ---- 計算 ----
    for alpha in alphas:
        fraction_roots[alpha] = np.array([
            find_fraction_root(alpha, p, w) for p in p_values
        ])
    tmp = 0
    for alpha in alphas:
        ax.plot(p_values, fraction_roots[alpha], label=f"alpha={alpha}", color = colors[tmp], lw = 5.0)
        tmp += 1

    ax.set_xlabel(r"Occurrence probability", fontsize=20)
    if(k==0):
        ax.set_ylabel(r"Maximal fraction", fontsize=20)
    ax.set_title(f"W={w:.2f}", fontsize=20)
    ax.tick_params(labelsize=16)
    ax.set_ylim(-0.025, 0.425)
    ax.set_xlim(-0.05, 1.05)

# plt.legend(fontsize=13)


plt.savefig(f"./figure/figS1.pdf", dpi=150, bbox_inches='tight')
# plt.show()