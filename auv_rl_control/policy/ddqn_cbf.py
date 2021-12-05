
from scipy.optimize import zeros
from policy.ddqn_policy import DDQNPolicy

import numpy as np
from math import radians

# Optimizer
from scipy import optimize


class DDQNPolicyCBF(DDQNPolicy):
    
    def __init__(self, action_map, n_states=3, n_actions=41, n_hidden=64, lr=0.0001, gamma=0.99):
        super().__init__(action_map, n_states=n_states, n_actions=n_actions, n_hidden=n_hidden, lr=lr, gamma=gamma)

        # Safety violation penalty
        self.K_eps = 10e12

        # CBF strength
        self.eta = 0.5

        # Optimization matrices
        self.H = np.array([[1, 0],[0, 0]])
        self.c = np.array([0, self.K_eps])

    def update(self):
        return super().update()

    def get_action(self, state):
        # Get the RL agent action
        u_rl = super().get_action(state)

        # Convert the RL action to physical terms
        rl_ctrl = self.action_map[u_rl]

        # Get the CBF action
        u_cbf = self.cbf_action(state, rl_ctrl)

        # Combine the actions
        u_k = u_rl + u_cbf

        # Constrain the action
        u_k = self.clamp(u_k, radians(-20), radians(20))

        # Find the action sum on the action map
        rounded_action = round(u_k * 2) / 2
        action_idx = self.action_map.index(rounded_action)

        return action_idx

    def cbf_action(self, state, rl_ctrl):
        # initial state
        x0 = np.array([0,0.01])

        # Create constraints
        p = np.array([1, 1, 1])
        q = 1
        Axb = lambda x:  np.array([[((1 - self.eta)*(p.dot(self.sys_f(state)) + q) - x[1]) - (p.dot(self.sys_f(state)) + p.dot(self.sys_g(state))*(x[0]+rl_ctrl) + q)], [1]])
        Axb_jac = lambda x:  np.array([[-p.dot(self.sys_g(state))], [1]])
        fun_constraint = Axb
        jac_constraint = Axb_jac

        # Constraints
        cons = {
            'type':'ineq',
            'fun': fun_constraint,
            'jac': jac_constraint
        }

        # Hide optimizer output
        opt = {'disp': False}

        action = optimize.minimize(self.objective_fn, x0, jac = self.jac_fn, constraints=cons, method='SLSQP', options=opt)

        return action[0]

    def jac_fn(self, x):
        return 2*x[0] + self.K_eps

    def objective_fn(self, x):
        J = (0.5 * np.dot(x.T, np.dot(self.H, x))+ np.dot(self.c, x))
        return J

    def sys_f(self, state):
        """ Estimate of the dynamics of the AUV """
        # Estimated parameters
        buoy = 0.001
        mass = 100
        heave = 0.1*abs(state[0])
        speed = 2
        dt = 0.1

        # states: pitch, pitch rate, depth (all errors)
        pitch_rate = state[1]
        pitch = state[0] + 0.1 * pitch_rate
        depth = state[2] - buoy*mass*dt + (heave*np.cos(pitch) - speed*np.sin(pitch))*dt

        return np.array([pitch, pitch_rate, depth])

    def sys_g(self, state):
        return np.array([0.001, 0.03, 0])

    def clamp(self, x, x_min, x_max):
        return max(min(x, x_max), x_min)