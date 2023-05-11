import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interactive, fixed
from types import SimpleNamespace
import scipy.optimize as optimize

class StandardSolow:
    def __init__(self):
        self.par = {
            'alpha': 0.33,
            'g': 0.02,
            'lambda_': 0.5,
            's': 0.2,
            'delta': 0.05,
            'n': 0.02,
        }
    
    @staticmethod
    def steady_state_equations(x, par):
        K, A = x
        alpha = par['alpha']
        s = par['s']
        delta = par['delta']
        g = par['g']
        eq1 = K**alpha * (A * par['L'])**(1 - alpha) - s * K - delta * K
        eq2 = A - (1 + g) * A
        return [eq1, eq2]

    def solve_steady_state(self):
        x0 = [4.0, 4.0]  # initial guess for K and A
        sol = root(self.steady_state_equations, x0, args=(self.par,))
        if sol.success:
            K_ss, A_ss = sol.x
            Y_ss = K_ss**self.par['alpha'] * (A_ss * self.par['L'])**(1 - self.par['alpha'])
            Y_per_L_ss = Y_ss / self.par['L']
            return Y_per_L_ss, Y_ss, K_ss, A_ss
        else:
            raise ValueError("Steady state calculation did not converge.")

class SemiEndogenous:
    def __init__(self):
        self.par = {
            'alpha': 0.3,
            'beta': 0.2,
            'phi': 0.5,
            'g': 0.02,
            's': 0.2,
            'delta': 0.05,
            'n': 0.02,
            'rho': 0.02,
            's_R': 0.2,
            'kappa': 0.2
        }

    def steady_state_equations(self, x):
        par = self.par
        A, K, L, LA = x
        eq1 = A - ((1 + par.rho) / (1 + par.n + par.rho)) ** (1 / (1 - par.phi))
        eq2 = K - (par.s / (par.n + par.delta)) ** (1 / (1 - par.alpha)) * A ** (par.alpha / (1 - par.alpha)) * L ** (par.beta / (1 - par.alpha)) * (par.s_R + L) ** (par.kappa / (1 - par.alpha))
        eq3 = LA - L * A
        eq4 = LA - K * A
        return [eq1, eq2, eq3, eq4]

    def solve_steady_state(self):
        x0 = [1.0, 1.0, 1.0, 1.0]  # initial guess for A, K, L, and LA
        sol = root(self.steady_state_equations, x0, method='hybr')
        if sol.success:
            A, K, L, LA = sol.x
            Y = K ** self.par['alpha'] * (A * L) ** self.par['beta']
            Y_per_L = Y / L
            return Y_per_L, Y, L, K, A, LA
        else:
            raise ValueError("Steady state calculation did not converge.")