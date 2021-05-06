"""
Test module for basicIC_engine.py
"""

#  This file is part of FAST : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2020  ONERA & ISAE-SUPAERO
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
import pandas as pd
import pytest
from fastoad.model_base import FlightPoint, Atmosphere
from fastoad.constants import EngineSetting

from ..basicIC_engine import BasicICEngine


def test_compute_flight_points():
    # BasicICEngine(max_power(W), design_altitude(m), design_speed(m/s), fuel_type, strokes_nb, prop_layout)
    engine = BasicICEngine(130000.0, 2400.0, 81.0, 1.0, 4.0, 1.0)  # load a 4-strokes 130kW gasoline engine

    # Test with scalars
    flight_point = FlightPoint(
        mach=0.3, altitude=0.0, engine_setting=EngineSetting.CLIMB.value, thrust=528.46263916
    )  # with engine_setting as int
    engine.compute_flight_points(flight_point)
    np.testing.assert_allclose(flight_point.thrust_rate, 0.5, rtol=1e-2)
    np.testing.assert_allclose(flight_point.sfc, 7.336237820294389e-06, rtol=1e-2)

    flight_point = FlightPoint(
        mach=0, altitude=0.0, engine_setting=EngineSetting.TAKEOFF, thrust_rate=0.8
    )  # with engine_setting as EngineSetting
    engine.compute_flight_points(flight_point)
    np.testing.assert_allclose(flight_point.thrust, 3532.6 * 0.8, rtol=1e-2)
    np.testing.assert_allclose(flight_point.sfc, 3.130755e-17, rtol=1e-2)

    # Test full arrays
    # 2D arrays are used, where first line is for thrust rates, and second line
    # is for thrust values.
    # As thrust rates and thrust values match, thrust rate results are 2 equal
    # lines and so are thrust value results.
    machs = [0, 0.3, 0.3, 0.8, 0.8]
    altitudes = [0, 0, 0, 10000, 13000]
    thrust_rates = [0.8, 0.5, 0.5, 0.4, 0.7]
    thrusts = [2813.473364,  528.462639,  528.462639,   44.872307, 36.148186]
    engine_settings = [
        EngineSetting.TAKEOFF,
        EngineSetting.TAKEOFF,
        EngineSetting.CLIMB,
        EngineSetting.IDLE,
        EngineSetting.CRUISE,
    ]  # mix EngineSetting with integers
    expected_sfc = [3.130755e-17, 7.336238e-06, 7.336238e-06, 1.482160e-05, 1.949231e-05]

    flight_points = pd.DataFrame()
    flight_points["mach"] = machs + machs
    flight_points["altitude"] = altitudes + altitudes
    flight_points["engine_setting"] = engine_settings + engine_settings
    flight_points["thrust_is_regulated"] = [False] * 5 + [True] * 5
    flight_points["thrust_rate"] = thrust_rates + [0.0] * 5
    flight_points["thrust"] = [0.0] * 5 + thrusts
    engine.compute_flight_points(flight_points)
    np.testing.assert_allclose(flight_points.sfc, expected_sfc + expected_sfc, rtol=1e-4)
    np.testing.assert_allclose(flight_points.thrust_rate, thrust_rates + thrust_rates, rtol=1e-4)
    np.testing.assert_allclose(flight_points.thrust, thrusts + thrusts, rtol=1e-4)


def test_engine_weight():
    # BasicICEngine(max_power(W), design_altitude(m), design_speed(m/s), fuel_type, strokes_nb, prop_layout)
    _50kw_engine = BasicICEngine(50000.0, 2400.0, 81.0, 1.0, 4.0, 1.0)
    np.testing.assert_allclose(_50kw_engine.compute_weight(), 82, atol=1)
    # BasicICEngine(max_power(W), design_altitude(m), design_speed(m/s), fuel_type, strokes_nb, prop_layout)
    _250kw_engine = BasicICEngine(250000.0, 2400.0, 81.0, 1.0, 4.0, 1.0)
    np.testing.assert_allclose(_250kw_engine.compute_weight(), 569, atol=1)
    # BasicICEngine(max_power(W), design_altitude(m), design_speed(m/s), fuel_type, strokes_nb, prop_layout)
    _130kw_engine = BasicICEngine(130000.0, 2400.0, 81.0, 1.0, 4.0, 1.0)
    np.testing.assert_allclose(_130kw_engine.compute_weight(), 277, atol=1)


def test_engine_dim():
    # BasicICEngine(max_power(W), design_altitude(m), design_speed(m/s), fuel_type, strokes_nb, prop_layout)
    _50kw_engine = BasicICEngine(50000.0, 2400.0, 81.0, 1.0, 4.0, 1.0)
    np.testing.assert_allclose(_50kw_engine.compute_dimensions(), [0.45, 0.67, 0.89, 2.03, 1.50, 0.30], atol=1e-2)
    # BasicICEngine(max_power(W), design_altitude(m), design_speed(m/s), fuel_type, strokes_nb, prop_layout)
    _250kw_engine = BasicICEngine(250000.0, 2400.0, 81.0, 1.0, 4.0, 1.0)
    np.testing.assert_allclose(_250kw_engine.compute_dimensions(), [0.77, 1.15, 1.53, 5.93, 2.06, 0.411], atol=1e-2)


def test_sfc_at_max_thrust():
    """
    Checks model against values from :...

    .. bibliography:: ../refs.bib
    """

    # Check with arrays
    # BasicICEngine(max_power(W), design_altitude(m), design_speed(m/s), fuel_type, strokes_nb)
    _50kw_engine = BasicICEngine(50000.0, 2400.0, 81.0, 1.0, 4.0, 1.0)
    atm = Atmosphere([0, 10668, 13000], altitude_in_feet=False)
    sfc = _50kw_engine.sfc_at_max_power(atm)
    # Note: value for alt==10668 is different from PhD report
    #       alt=13000 is here just for testing in stratosphere
    np.testing.assert_allclose(sfc, [7.09319444e-08, 6.52497276e-08, 6.44112478e-08], rtol=1e-4)

    # Check with scalars
    # BasicICEngine(max_power(W), design_altitude(m), design_speed(m/s), fuel_type, strokes_nb)
    _250kw_engine = BasicICEngine(250000.0, 2400.0, 81.0, 1.0, 4.0, 1.0)
    atm = Atmosphere(0, altitude_in_feet=False)
    sfc = _250kw_engine.sfc_at_max_power(atm)
    np.testing.assert_allclose(sfc, 8.540416666666667e-08, rtol=1e-4)

    # Check with scalars
    # BasicICEngine(max_power(W), design_altitude(m), design_speed(m/s), fuel_type, strokes_nb)
    _130kw_engine = BasicICEngine(130000.0, 2400.0, 81.0, 1.0, 4.0, 1.0)
    atm = Atmosphere(0, altitude_in_feet=False)
    sfc = _130kw_engine.sfc_at_max_power(atm)
    np.testing.assert_allclose(sfc, 7.965417e-08, rtol=1e-4)


def test_sfc_ratio():
    """    Checks SFC ratio model    """
    engine = BasicICEngine(75000.0, 2400.0, 81.0, 1.0, 4.0, 1.0)

    # Test different altitude (even negative: robustness) with constant thrust rate/power ratio
    altitudes = np.array([-2370, -1564, -1562.5, -1560, -846, 678, 2202, 3726])
    ratio, _ = engine.sfc_ratio(altitudes, 0.8)
    assert ratio == pytest.approx(
        [0.958656, 0.958656, 0.958656, 0.958656, 0.958656, 0.9603615, 0.96421448, 0.96808154], rel=1e-3
    )

    # Because there some code differs when we have scalars:
    ratio, _ = engine.sfc_ratio(1562.5, 0.6)
    assert ratio == pytest.approx(0.845, rel=1e-3)
