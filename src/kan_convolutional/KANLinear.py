import torch
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from numpy.polynomial import Polynomial
from scipy.optimize import minimize
import numpy as np
import sympy as sp
from sympy import symbols, lambdify





fit_lib = {
    # poly
    # 'poly_0': lambda c0 : c0,
    'poly_1': lambda x, c0, c1: c0 + c1*x,
    'poly_2': lambda x, c0, c1, c2: c0 + c1*x + c2*x**2,
    'poly_3': lambda x, c0, c1, c2, c3: c0 + c1*x + c2*x**2 + c3*x**3,
    'poly_4': lambda x, c0, c1, c2, c3, c4: c0 + c1*x + c2*x**2 + c3*x**3 + c4*x**4,
    '1/x': lambda x, c0, c1: c0/(x+c1),
    '1/x^2': lambda x, c0, c1: c0/(x+c1)**2,
    # 'sqrt': lambda x, c0, c1: c0*sp.sqrt(x+c1),
    # '1/sqrt(x)': lambda x, c0, c1: c0/sp.sqrt(x+c1),
    'exp': lambda x, c0, c1: c0 * sp.exp(x+c1),
    # 'log': lambda x, c0, c1: c0 * sp.log(x+c1),
    'sin': lambda x, c0, c1: c0 * sp.sin(x+c1),
    'tan': lambda x, c0, c1: c0 * sp.tan(x+c1),
    'tanh': lambda x, c0, c1: c0 * sp.tanh(x+c1),
    'arcsin':lambda x, c0, c1: c0 * sp.asin(x+c1),
    'arctan':lambda x, c0, c1: c0 * sp.atan(x+c1),
    'arctanh':lambda x, c0, c1: c0 * sp.atanh(x+c1),
    # 'gaussian': lambda x, c0, c1, c2: c0 * sp.exp(-(x+c1)**2/c2)
}






class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.saved_x = None  
        self.saved_yi = None  
        self.saved_b_splines = None


        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        self.saved_x = x.detach() # save x
        self.saved_b_splines = self.b_splines(x).detach()  # 保存 B 样条的输出
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        ) 
        

        return base_output + spline_output
    
    @torch.no_grad()
    def save_yi(self):

        k_spines = self.scaled_spline_weight  


        k_base = self.base_weight  

        saved_b_splines = self.saved_b_splines  

        yi_list = []
        for i in range(self.in_features):

            b_spline_output = saved_b_splines[:, i, :]  
            base_output = self.base_activation(self.saved_x[:, i])  

            # calculate yi = k_spines_i * self.b_splines(xi) + b_spines_i + k_base_i * self.base_activation(xi) + b_base_i
            yi = (
                torch.sum(k_spines[:, i, :] * b_spline_output, dim=-1)  
                + k_base[:, i] * base_output  
            )
            
            yi_list.append(yi)


        self.saved_yi = torch.stack(yi_list, dim=1)  
        # return self.saved_yi





    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.cat(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )
    

    @torch.no_grad()
    def fit_function(self, x, y, func,x_sym):


        x_np = np.array(x).flatten()
        y_np = np.array(y).flatten()


        num_params = func.__code__.co_argcount - 1
        params = symbols(f'c0:{num_params}')
        expr = func(x_sym, *params)

        def residuals(param_values):
            substituted_expr = expr.subs({p: v for p, v in zip(params, param_values)})
            y_pred = [substituted_expr.subs(x_sym, val) for val in x_np]
            return np.sum((np.array(y_pred, dtype=np.float64) - y_np) ** 2)


        result = minimize(residuals, np.ones(num_params))
        best_params = result.x
        best_fit_expr = expr.subs({p: v for p, v in zip(params, best_params)})

        y_pred = [best_fit_expr.subs(x_sym, val) for val in x_np]
        r2 = r2_score(y_np, np.array(y_pred, dtype=np.float64))

        return best_fit_expr, r2



    @torch.no_grad()
    def fit_symbolic_for_each_feature(self,fit_lib=fit_lib):


        if not hasattr(self, 'saved_x') or not hasattr(self, 'saved_yi'):
            raise ValueError("You need to call the forward method first to save the inputs and outputs.")


        total_expr = 0
        total_r2 = 0
        in_features = self.saved_x.shape[1]


        for i in range(in_features):
            
            x_sym = symbols(f'x{i+1}')
            x_i = self.saved_x[:, i].cpu().numpy()  
            y_i = self.saved_yi[:, i].cpu().numpy()  

            best_r2 = -float('inf')
            best_expr = None

            
            for func_name, func in fit_lib.items():
                try:
                    
                    expr, r2 = self.fit_function(x_i, y_i, func, x_sym)
                    if r2 > best_r2:
                        best_r2 = r2
                        best_expr = expr
                except Exception as e:
                    print(f"Error fitting {func_name} for x_{i}: {e}")
                    continue

            
            print(f" x_{i+1} best_expr: {best_expr}, R² = {best_r2}")

            
            if best_expr is not None:
                total_expr += best_expr  
            total_r2 += best_r2  


        avg_r2 = total_r2 / in_features if in_features > 0 else 0
        return total_expr, avg_r2



    
