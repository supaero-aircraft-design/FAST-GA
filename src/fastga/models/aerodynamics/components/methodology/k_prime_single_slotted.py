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

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

if __name__ == "__main__":
    # 1. Load CSV
    df = pd.read_csv("./data/k_single_slot.csv")

    # 2. Extract x, y, z
    x = df["flap_angle"].values
    y = df["chord_ratio"].values
    z = df["k_prime"].values

    # 3. Stack x and y for 2D input
    X = np.column_stack((x, y))

    # 4. Create polynomial features (e.g., degree 2)
    poly = PolynomialFeatures(degree=3, include_bias=True)
    X_poly = poly.fit_transform(X)

    # 5. Fit linear regression to the polynomial features
    model = LinearRegression()
    model.fit(X_poly, z)

    # 6. Get coefficients
    coeffs = model.coef_
    intercept = model.intercept_
    feature_names = poly.get_feature_names_out(["x", "y"])

    # 7. Show the fitted polynomial
    print("Fitted polynomial:")
    for name, coef in zip(feature_names, coeffs):
        print(f"{coef:+.4f} * {name}")
    print(f"{intercept:+.4f} (intercept)")

    # Make predictions using the fitted model
    z_pred = model.predict(X_poly)

    # Calculate R² (coefficient of determination)
    r2 = r2_score(z, z_pred)
    print(f"R² Score: {r2:.4f}")

    # Calculate various error metrics
    mse = mean_squared_error(z, z_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(z, z_pred)

    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")

    # Calculate relative error metrics
    mean_z = np.mean(z)
    relative_rmse = rmse / mean_z * 100
    relative_mae = mae / mean_z * 100

    print(f"Relative RMSE: {relative_rmse:.2f}%")
    print(f"Relative MAE: {relative_mae:.2f}%")

    # Additional useful metrics
    residuals = z - z_pred
    print(f"Mean of residuals: {np.mean(residuals):.6f} (should be close to 0)")
    print(f"Standard deviation of residuals: {np.std(residuals):.4f}")


# Fitted polynomial:
# +0.0000 * 1
# +0.0060 * x
# +2.6633 * y
# -0.0002 * x^2
# -0.0121 * x y
# -2.9929 * y^2
# +0.0000 * x^3
# -0.0002 * x^2 y
# +0.0354 * x y^2
# -0.5931 * y^3
# +0.0239 (intercept)
# R² Score: 0.9931
# Mean Squared Error (MSE): 0.0001
# Root Mean Squared Error (RMSE): 0.0115
# Mean Absolute Error (MAE): 0.0096
# Relative RMSE: 3.18%
# Relative MAE: 2.68%
# Mean of residuals: -0.000000 (should be close to 0)
# Standard deviation of residuals: 0.0115
