"""
Python module to obtain empirical pitching moment factor coefficient regression model.
"""
#  This file is part of FAST-OAD_CS23 : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2025  ONERA & ISAE-SUPAERO
#  FAST is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import plotly.graph_objects as go
from scipy import optimize


# Second order polynomial function
def polynomial_func(x, a, b, c):
    """Define regression model"""
    return a * x**2 + b * x + c


if __name__ == "__main__":
    # Data points from Raymer's Aircraft Design: A Conceptual Approach (Figure 16.14)
    x_0_25 = np.array(
        [
            0.1105691057,
            0.1373983740,
            0.1642276423,
            0.1910569106,
            0.2178861789,
            0.2414634146,
            0.2674796748,
            0.2918699187,
            0.3097560976,
            0.3235772358,
            0.3357723577,
            0.3495934959,
            0.3658536585,
            0.3796747967,
            0.3910569106,
            0.4048780488,
            0.4178861789,
            0.4365853659,
            0.4520325203,
            0.4682926829,
            0.4894308943,
            0.5065040650,
            0.5284552846,
            0.5536585366,
            0.5715447154,
            0.5894308943,
            0.6065040650,
            0.6211382114,
            0.5975609756,
            0.5634146341,
            0.5414634146,
            0.5170731707,
            0.50,
            0.4788617886,
            0.09756097561,
        ]
    )
    k_fus = np.array(
        [
            0.005,
            0.005409836066,
            0.005819672131,
            0.006311475410,
            0.007049180328,
            0.007868852459,
            0.008852459016,
            0.009918032787,
            0.010901639344,
            0.011721311475,
            0.012540983607,
            0.013606557377,
            0.014918032787,
            0.016229508197,
            0.017295081967,
            0.018688524590,
            0.020081967213,
            0.022377049180,
            0.024262295082,
            0.026557377049,
            0.029590163934,
            0.032131147541,
            0.035655737705,
            0.039918032787,
            0.043032786885,
            0.046147540984,
            0.049672131148,
            0.052295081967,
            0.047704918033,
            0.041311475410,
            0.037540983607,
            0.033770491803,
            0.031147540984,
            0.027868852459,
            0.004918032787,
        ]
    )

    # Fit polynomial regression
    polynomial_params, _ = optimize.curve_fit(polynomial_func, x_0_25, k_fus)

    # Calculate fitted y values and R-squared
    k_fus_polynomial = polynomial_func(x_0_25, *polynomial_params)
    r2_polynomial = 1 - np.sum((k_fus - k_fus_polynomial) ** 2) / np.sum(
        (k_fus - np.mean(k_fus)) ** 2
    )

    # Generate points for smooth curve
    x_smooth = np.linspace(min(x_0_25), max(x_0_25), 100)
    y_smooth = polynomial_func(x_smooth, *polynomial_params)

    # Create interactive plot
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=x_0_25, y=k_fus, mode="markers", name="Data points", marker=dict(color="blue"))
    )
    fig.add_trace(
        go.Scatter(
            x=x_smooth,
            y=y_smooth,
            mode="lines",
            name=f"Polynomial (R² = {r2_polynomial:.3f})",
            line=dict(color="red"),
        )
    )

    # Add function text to the plot
    function_text = (
        f"k_fus = {polynomial_params[0]:.6f}x_0_25² +"
        f" {polynomial_params[1]:.6f}x_0_25 + {polynomial_params[2]:.6f}"
    )
    fig.add_annotation(
        x=max(x_0_25),
        y=min(k_fus),
        text=function_text,
        showarrow=False,
        font=dict(size=12, color="blue"),
        xanchor="right",
        yanchor="bottom",
    )

    fig.update_layout(
        title="Second Order Polynomial Regression for empirical pitching moment factor coefficient",
        xaxis_title="x_0_25",
        yaxis_title="k_fus (1/deg)",
    )
    fig.show()

    print(
        f"Polynomial: k_fus = {polynomial_params[0]:.6f}x_0_25² +"
        f" {polynomial_params[1]:.6f}x_0_25 + {polynomial_params[2]:.6f}"
    )
    print(f"R² = {r2_polynomial:.3f}")
