import os
import numpy as np
import casadi as ca
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from quadrotor import Quadrotor

class Optimization():
    def __init__(self, quad: Quadrotor, wp_num: int, Ns: list, tol=0.3, _tol_term=0.01):
        self._quad = quad
        self._ddynamics = self._quad.ddynamics_dt()
        self._tol = tol
        self._tol_term = _tol_term
        self._wp_num = wp_num
        self._seg_num = len(Ns)
        assert(self._seg_num == wp_num + 1)
        self._Ns = Ns
        self._Horizon = sum(Ns)
        self._N_wp_base = [sum(Ns[:i]) for i in range(self._seg_num + 1)]
        self._X_dim = self._ddynamics.size1_in(0)
        self._U_dim = self._ddynamics.size1_in(1)
        self._X_lb = self._quad._X_lb
        self._X_ub = self._quad._X_ub
        self._U_lb = self._quad._U_lb
        self._U_ub = self._quad._U_ub
        self._DTs = ca.SX.sym('DTs', self._seg_num)
        self._Xs = ca.SX.sym('Xs', self._X_dim, self._Horizon)
        self._Us = ca.SX.sym('Us', self._U_dim, self._Horizon)
        self._WPs_p = ca.SX.sym('WPs_p', 3, self._wp_num)
        self._X_init = ca.SX.sym('X_init', self._X_dim)
        self._X_end = ca.SX.sym('X_end', self._X_dim)
        self._cost_WP_p = ca.diag([1,1,1])
        self._cost_state = ca.diag([1,1,1,0.5,0.5,0.5,0.1,0.1,0.1,0.1,0.05,0.05,0.05])
        self._opt_t_option = {
            'verbose': False,
            'ipopt.max_iter': 1000,
            'ipopt.print_level': 0
        }
        self._initialize_nlp_variables()
        self._ml_model = LinearRegression()
        self._train_ml_model()
        
        # Initialize the attributes here
        self._lam_x0 = np.zeros(self._X_dim * self._Horizon + self._U_dim * self._Horizon + self._seg_num)
        self._lam_g0 = np.zeros(len(self._nlp_g_dyn) * self._X_dim + len(self._nlp_g_wp_p))

    def _initialize_nlp_variables(self):
        self._nlp_x_x = []
        self._nlp_lbx_x = []
        self._nlp_ubx_x = []
        self._nlp_x_u = []
        self._nlp_lbx_u = []
        self._nlp_ubx_u = []
        self._nlp_x_t = []
        self._nlp_lbx_t = []
        self._nlp_ubx_t = []
        self._nlp_g_orientation = []
        self._nlp_lbg_orientation = []
        self._nlp_ubg_orientation = []
        self._nlp_g_dyn = []
        self._nlp_lbg_dyn = []
        self._nlp_ubg_dyn = []
        self._nlp_g_wp_p = []
        self._nlp_lbg_wp_p = []
        self._nlp_ubg_wp_p = []
        self._nlp_g_quat = []
        self._nlp_lbg_quat = []
        self._nlp_ubg_quat = []
        self._nlp_p_xinit = [self._X_init]
        self._xinit = np.array([0,0,0, 0,0,0, 1,0,0,0, 0,0,0])
        self._nlp_p_xend = [self._X_end]
        self._xend = np.array([0,0,0, 0,0,0, 1,0,0,0, 0,0,0])
        self._nlp_p_wp_p = []
        self._nlp_obj_time = 0
        self._setup_nlp()

    def _setup_nlp(self):
        for i in range(self._seg_num):
            self._nlp_x_x += [self._Xs[:, self._N_wp_base[i]]]
            self._nlp_lbx_x += self._X_lb
            self._nlp_ubx_x += self._X_ub
            self._nlp_x_u += [self._Us[:, self._N_wp_base[i]]]
            self._nlp_lbx_u += self._U_lb
            self._nlp_ubx_u += self._U_ub
            self._nlp_x_t += [self._DTs[i]]
            self._nlp_lbx_t += [0]
            self._nlp_ubx_t += [0.5]
            if i == 0:
                dd_dyn = self._Xs[:, 0] - self._ddynamics(self._X_init, self._Us[:, 0], self._DTs[0])
                self._nlp_g_dyn += [dd_dyn]
            else:
                dd_dyn = self._Xs[:, self._N_wp_base[i]] - self._ddynamics(self._Xs[:, self._N_wp_base[i] - 1], self._Us[:, self._N_wp_base[i]], self._DTs[i])
                self._nlp_g_dyn += [dd_dyn]
            self._nlp_lbg_dyn += [-0.0 for _ in range(self._X_dim)]
            self._nlp_ubg_dyn += [0.0 for _ in range(self._X_dim)]
            self._nlp_obj_time += self._DTs[i] * self._Ns[i]
            for j in range(1, self._Ns[i]):
                self._nlp_x_x += [self._Xs[:, self._N_wp_base[i] + j]]
                self._nlp_lbx_x += self._X_lb
                self._nlp_ubx_x += self._X_ub
                self._nlp_x_u += [self._Us[:, self._N_wp_base[i] + j]]
                self._nlp_lbx_u += self._U_lb
                self._nlp_ubx_u += self._U_ub
                dd_dyn = self._Xs[:, self._N_wp_base[i] + j] - self._ddynamics(self._Xs[:, self._N_wp_base[i] + j - 1], self._Us[:, self._N_wp_base[i] + j], self._DTs[i])
                self._nlp_g_dyn += [dd_dyn]
                self._nlp_lbg_dyn += [-0.0 for _ in range(self._X_dim)]
                self._nlp_ubg_dyn += [0.0 for _ in range(self._X_dim)]
            if i == self._seg_num - 1:
                self._nlp_g_wp_p += [(self._Xs[:, self._N_wp_base[i + 1] - 1] - self._X_end).T @ (self._Xs[:, self._N_wp_base[i + 1] - 1] - self._X_end)]
                self._nlp_lbg_wp_p += [0]
                self._nlp_ubg_wp_p += [self._tol_term * self._tol_term]
            else:
                self._nlp_g_wp_p += [(self._Xs[:3, self._N_wp_base[i + 1] - 1] - self._WPs_p[:, i]).T @ (self._Xs[:3, self._N_wp_base[i + 1] - 1] - self._WPs_p[:, i])]
                self._nlp_lbg_wp_p += [0]
                self._nlp_ubg_wp_p += [self._tol * self._tol]
                self._nlp_p_wp_p += [self._WPs_p[:, i]]

    def _train_ml_model(self):
        data_dir = "../../resources/trajectory"
        all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
        
        df_list = [pd.read_csv(file) for file in all_files]
        data = pd.concat(df_list, ignore_index=True)
        
        # Assuming the CSV structure provided, extract relevant features and labels
        features = data[[
            'p_x', 'p_y', 'p_z', 'v_x', 'v_y', 'v_z',
            'q_w', 'q_x', 'q_y', 'q_z', 'w_x', 'w_y', 'w_z',
            'a_lin_x', 'a_lin_y', 'a_lin_z', 'a_rot_x', 'a_rot_y', 'a_rot_z'
        ]]
        labels = data[[
            'u_1', 'u_2', 'u_3', 'u_4'
        ]]
        
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
        
        self._ml_model.fit(X_train, y_train)
        print(f"Model R^2 score: {self._ml_model.score(X_test, y_test)}")

    def set_initial_guess(self, xut0):
        self._xut0 = xut0

    def define_opt_t(self):
        nlp_dect = {
            'f': self._nlp_obj_time,
            'x': ca.vertcat(*(self._nlp_x_x + self._nlp_x_u + self._nlp_x_t)),
            'p': ca.vertcat(*(self._nlp_p_xinit + self._nlp_p_xend + self._nlp_p_wp_p)),
            'g': ca.vertcat(*(self._nlp_g_dyn + self._nlp_g_wp_p)),
        }
        self._opt_t_solver = ca.nlpsol('opt_t', 'ipopt', nlp_dect, self._opt_t_option)

    def solve_opt_t(self, xinit, xend, wp_p, warm=False):
        p = np.zeros(2 * self._X_dim + 3 * self._wp_num)
        p[:self._X_dim] = xinit
        p[self._X_dim:2 * self._X_dim] = xend
        p[2 * self._X_dim:2 * self._X_dim + 3 * self._wp_num] = wp_p

        if warm:
            features = np.concatenate((xinit, wp_p)).reshape(1, -1)
            self._xut0 = self._ml_model.predict(features)[0]
        else:
            self._xut0 = np.zeros(self._X_dim * self._Horizon + self._U_dim * self._Horizon + self._seg_num)

        res = self._opt_t_solver(
            x0=self._xut0,
            lam_x0=self._lam_x0,
            lam_g0=self._lam_g0,
            lbx=(self._nlp_lbx_x + self._nlp_lbx_u + self._nlp_lbx_t),
            ubx=(self._nlp_ubx_x + self._nlp_ubx_u + self._nlp_ubx_t),
            lbg=(self._nlp_lbg_dyn + self._nlp_lbg_wp_p),
            ubg=(self._nlp_ubg_dyn + self._nlp_ubg_wp_p),
            p=p
        )

        self._xut0 = res['x'].full().flatten()
        self._lam_x0 = res["lam_x"]
        self._lam_g0 = res["lam_g"]
        return res

# Function to compare optimization results
def compare_optimizations(quad, wp_num, Ns, xinit, xend, wp_p):
    opt = Optimization(quad, wp_num, Ns)

    # Solve without warm start
    result_without_warm = opt.solve_opt_t(xinit, xend, wp_p, warm=False)
    print("Result without warm start:", result_without_warm)

    # Solve with warm start
    result_with_warm = opt.solve_opt_t(xinit, xend, wp_p, warm=True)
    print("Result with warm start:", result_with_warm)

    # Compare key metrics
    # For demonstration, we print the optimized trajectories
    print("Optimized trajectory without warm start:")
    print(opt._xut0.reshape(-1, opt._X_dim))
    
    print("Optimized trajectory with warm start:")
    print(opt._xut0.reshape(-1, opt._X_dim))

    # Additional metrics such as convergence time, number of iterations, etc., can be added

# Example usage
# quad = Quadrotor()  # Initialize your quadrotor object here
# wp_num = 5  # Number of waypoints
# Ns = [10, 10, 10, 10, 10, 10]  # Number of segments
# xinit = np.zeros(13)  # Initial state
# xend = np.zeros(13)  # Final state
# wp_p = np.random.rand(3 * wp_num)  # Random waypoints

# compare_optimizations(quad, wp_num, Ns, xinit, xend, wp_p)

