# Solve the optimization problem with several sets of (alpha, W, M)
# For Figure 2, alpha=0.5, W = 0.1, M = 10
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

# Estimation of the mean value of given data with trancated normal distribution in [0,1]
def truncated_normal_mle(data, bounds=(0.0, 1.0)):
    a, b = bounds
    
    def neg_log_likelihood(params):
        mu, sigma = params
        if sigma <= 0:
            return 1e10
        
        alpha = (a - mu) / sigma
        fraction_std = (b - mu) / sigma
        
        Z = normal_dist.cdf(fraction_std) - normal_dist.cdf(alpha)
        
        if Z <= 0 or np.isnan(Z):
            return 1e10
        
        log_pdf = normal_dist.logpdf(data, mu, sigma)
        nll = -np.sum(log_pdf) + len(data) * np.log(Z)
        
        return nll
    
    if len(data) == 0:
        return 0.5, 0.2
    
    if len(data) == 1:
        return float(data[0]), 1e-6
    
    mu_init = np.mean(data)
    sigma_init = np.std(data)
    
    if sigma_init < 1e-6:
        sigma_init = 0.1
    
    mu_init = np.clip(mu_init, a + 1e-6, b - 1e-6)
    
    best_result = None
    best_nll = float('inf')
    
    init_patterns = [
        [mu_init, sigma_init],
        [0.5, 0.2],
        [(a + b) / 2, (b - a) / 4],
    ]
    
    for init_params in init_patterns:
        try:
            result = minimize(
                neg_log_likelihood,
                init_params,
                method="L-BFGS-B",
                bounds=[(a + 1e-6, b - 1e-6), (1e-6, b - a)]
            )
            
            if result.fun < best_nll:
                best_nll = result.fun
                best_result = result
        except:
            continue
    
    if best_result is not None and best_result.success:
        mu, sigma = best_result.x
        return mu, sigma
    else:
        mu_fallback = np.mean(data)
        sigma_fallback = np.std(data)
        if sigma_fallback < 1e-6:
            sigma_fallback = 0.1
        return mu_fallback, sigma_fallback

# Estimation of the confidense interval of given data with trancated normal distribution in [0,1]
def truncated_normal_confidence_interval(mu, sigma, bounds=(0.0, 1.0), confidence=0.95):
    a, b = bounds
    
    if sigma < 1e-6:
        epsilon = max(1e-6, abs(mu) * 1e-6)
        return np.clip(mu - epsilon, a, b), np.clip(mu + epsilon, a, b)
    
    alpha = (a - mu) / sigma
    fraction_std = (b - mu) / sigma
    
    Z = normal_dist.cdf(fraction_std) - normal_dist.cdf(alpha)
    
    if Z <= 0:
        return a, b
    
    p_lower = (1 - confidence) / 2
    p_upper = 1 - p_lower
    

    cdf_lower = normal_dist.cdf(alpha) + p_lower * Z
    cdf_upper = normal_dist.cdf(alpha) + p_upper * Z
    
    z_lower = normal_dist.ppf(cdf_lower)
    z_upper = normal_dist.ppf(cdf_upper)
    
    lower = mu + sigma * z_lower
    upper = mu + sigma * z_upper
    
    lower = np.clip(lower, a, b)
    upper = np.clip(upper, a, b)
    
    if lower > upper:
        lower, upper = upper, lower
    
    mu_clipped = np.clip(mu, a, b)
    lower = min(lower, mu_clipped)
    upper = max(upper, mu_clipped)
    
    return lower, upper

# Getting interpolated curve from given data points
def linear_interpolate_curve(x_points, y_points, num_points=100):
    if len(y_points.shape) == 1:
        y_points = y_points.reshape(1, -1)
        squeeze_output = True
    else:
        squeeze_output = False
    
    num_curves = y_points.shape[0]
    
    t = np.linspace(0, 1, len(x_points))
    t_interp = np.linspace(0, 1, num_points)
    
    x_interp = np.interp(t_interp, t, x_points)
    
    y_interp = np.zeros((num_curves, num_points))
    for i in range(num_curves):
        y_interp[i, :] = np.interp(t_interp, t, y_points[i, :])
    
    if squeeze_output:
        y_interp = y_interp[0, :]
    
    return x_interp, y_interp

# Parameters (for figure 2, alpha = 0.5, W = 0.1, M = 10)
# Number of contexts
M_list = [10]
# Parameters
alpha_list = [0.1, 0.5, 0.9]
W_list = [0.1, 0.01]
# Number of trials
num_trials = 100
rng = np.random.default_rng(0)


for alpha in alpha_list:
    for W in W_list:
        for M in M_list:
            p_list = []
            fraction_list = []
            for _ in range(num_trials):
                p = rng.dirichlet(np.ones(M))
                p = np.sort(p)

                fraction = optimize_fraction(alpha, W, p)

                p_list.append(p)
                fraction_list.append(fraction)

            num_interp = 100
            
            p_ranges = [(g.min(), g.max()) for g in p_list]

            p_global_min = min([g_min for g_min, g_max in p_ranges])
            p_global_max = max([g_max for g_min, g_max in p_ranges])
            
            p_common = np.linspace(p_global_min, p_global_max, num_interp)
            
            fraction_mean = np.zeros(num_interp)
            fraction_std = np.zeros(num_interp)
            fraction_lower = np.zeros(num_interp)
            fraction_upper = np.zeros(num_interp)
            
            for i, g_eval in enumerate(p_common):
                fraction_values = []
                for p, fraction in zip(p_list, fraction_list):
                    if p.min() <= g_eval <= p.max():
                        fraction_val = np.interp(g_eval, p, fraction)
                        fraction_values.append(fraction_val)
                
                if len(fraction_values) > 0:
                    mu_mle, sigma_mle = truncated_normal_mle(np.array(fraction_values), bounds=(0.0, 1.0))
                    fraction_mean[i] = mu_mle
                    fraction_std[i] = sigma_mle
                    
                    lower, upper = truncated_normal_confidence_interval(
                        mu_mle, sigma_mle, bounds=(0.0, 1.0), confidence=0.95
                    )
                    fraction_lower[i] = np.clip(lower, 0.0, 1.0)
                    fraction_upper[i] = np.clip(upper, 0.0, 1.0)
                else:
                    fraction_mean[i] = np.nan
                    fraction_std[i] = np.nan
                    fraction_lower[i] = np.nan
                    fraction_upper[i] = np.nan
            
            valid_idx = ~np.isnan(fraction_mean)
            
            all_fraction_values = []
            for fraction in fraction_list:
                all_fraction_values.extend(fraction.tolist() if hasattr(fraction, 'tolist') else fraction)
            
            if len(all_fraction_values) > 0:
                fraction_min = float(np.min(all_fraction_values))
                fraction_max = float(np.max(all_fraction_values))
                fraction_margin = (fraction_max - fraction_min) * 0.1
                y_min = max(-fraction_margin, fraction_min - fraction_margin)
                y_max = min(1.0, fraction_max + fraction_margin)
            else:
                y_min = -fraction_margin
                y_max = 1.0

            # Plotting
            plt.figure(figsize=(8, 6))
            
            for p, fraction in zip(p_list, fraction_list):
                plt.plot(p, fraction, marker='o', alpha=0.2, color='gray', linewidth=1.5, markersize=5)
            
            if np.any(valid_idx):
                plt.fill_between(p_common[valid_idx], 
                                np.clip(fraction_lower[valid_idx], 0.0, 1.0),
                                np.clip(fraction_upper[valid_idx], 0.0, 1.0),
                                alpha=0.15, color='red', label='95% Confidence Interval (Truncated Normal MLE)')
            
            if np.any(valid_idx):
                plt.plot(p_common[valid_idx], fraction_mean[valid_idx], 'red', linewidth=3.5, label='Mean (MLE)')
            
            plt.axhline(y=0, color='k', linestyle='--', linewidth=2.0, alpha=1.0)
            plt.axhline(y=1, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
            
            plt.xlabel("Occurrence probability", fontsize=20)
            plt.ylabel("Engram cell fraction", fontsize=20)
            plt.ylim(y_min, y_max)
            plt.tick_params(labelsize=16)
            # plt.grid(True, alpha=0.3)
            # plt.legend(fontsize=13)
            plt.savefig(f"./figure/fraction_p_alpha{alpha}_M{M}_W{W}.pdf", dpi=150, bbox_inches='tight')
            plt.close()
