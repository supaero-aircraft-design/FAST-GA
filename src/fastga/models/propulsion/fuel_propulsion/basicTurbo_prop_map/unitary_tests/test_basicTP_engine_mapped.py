"""
Test module for basicIC_engine.py
"""

#  This file is part of FAST-OAD_CS23 : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2022  ONERA & ISAE-SUPAERO
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

import copy

import fastoad.api as oad
from fastoad.constants import EngineSetting

from stdatm import Atmosphere

from fastga.models.propulsion.fuel_propulsion.basicTurbo_prop_map.basicTP_engine_mapped import (
    BasicTPEngineMapped,
)
from fastga.models.propulsion.fuel_propulsion.basicTurbo_prop.basicTP_engine import BasicTPEngine

from .data.dummy_maps import *

INVALID_SFC = 0.0


def test_compute_flight_points():
    engine = BasicTPEngineMapped(
        power_design=485.429,
        t_41t_design=1200,
        opr_design=8.0,
        cruise_altitude_propeller=6096.0,
        design_altitude=0.0,
        design_mach=0.2,
        prop_layout=1.0,
        bleed_control=0.0,
        itt_limit=1000.0,
        power_limit=404.524,
        opr_limit=11.0,
        speed_SL=SPEED,
        thrust_SL=THRUST_SL,
        thrust_limit_SL=THRUST_SL_LIMIT,
        efficiency_SL=EFFICIENCY_SL,
        speed_CL=SPEED,
        thrust_CL=THRUST_CL,
        thrust_limit_CL=THRUST_CL_LIMIT,
        efficiency_CL=EFFICIENCY_CL,
        effective_J=1.0,
        effective_efficiency_ls=1.0,
        effective_efficiency_cruise=1.0,
        turbo_mach_SL=MACH_ARRAY,
        turbo_thrust_SL=THRUST_ARRAY_SL,
        turbo_thrust_max_SL=THRUST_MAX_ARRAY_SL,
        turbo_sfc_SL=SFC_SL,
        turbo_mach_CL=MACH_ARRAY,
        turbo_thrust_CL=THRUST_ARRAY_CL,
        turbo_thrust_max_CL=THRUST_MAX_ARRAY_CL,
        turbo_sfc_CL=SFC_CL,
        turbo_mach_IL=MACH_ARRAY,
        turbo_thrust_IL=THRUST_ARRAY_IL,
        turbo_thrust_max_IL=THRUST_MAX_ARRAY_IL,
        turbo_sfc_IL=SFC_IL,
        level_IL=3048,
    )  # load a 1000 kW turboprop engine

    # Test full arrays
    # 2D arrays are used, where first line is for thrust rates, and second line
    # is for thrust values.
    # As thrust rates and thrust values match, thrust rate results are 2 equal
    # lines and so are thrust value results.
    machs = [0.06, 0.12, 0.18, 0.22, 0.375]
    altitudes = [0, 0, 0, 1000, 2400]
    thrust_rates = [0.8, 0.5, 0.5, 0.4, 0.7]
    thrusts = [8723.73643444, 4352.43513423, 3670.56367899, 2378.52672608, 2754.44695866]
    engine_settings = [
        EngineSetting.TAKEOFF,
        EngineSetting.TAKEOFF,
        EngineSetting.CLIMB,
        EngineSetting.IDLE,
        EngineSetting.CRUISE,
    ]  # mix EngineSetting with integers
    expected_sfc = [7.11080994e-06, 1.14707988e-05, 1.40801854e-05, 1.75188296e-05, 1.82845376e-05]

    flight_points = oad.FlightPoint(
        mach=machs + machs,
        altitude=altitudes + altitudes,
        engine_setting=engine_settings + engine_settings,
        thrust_is_regulated=[False] * 5 + [True] * 5,
        thrust_rate=thrust_rates + [0.0] * 5,
        thrust=[0.0] * 5 + thrusts,
    )

    engine.compute_flight_points(flight_points)
    np.testing.assert_allclose(flight_points.thrust, thrusts + thrusts, rtol=1e-2)
    np.testing.assert_allclose(flight_points.sfc, expected_sfc + expected_sfc, rtol=1e-2)

    solo_flight_point = oad.FlightPoint(
        mach=0.375,
        altitude=2400,
        engine_setting=EngineSetting.CRUISE,
        thrust_is_regulated=False,
        thrust_rate=0.7,
        thrust=0.0,
    )
    engine.compute_flight_points(solo_flight_point)
    np.testing.assert_allclose(solo_flight_point.thrust, 2754.446958660104, rtol=1e-2)
    np.testing.assert_allclose(solo_flight_point.sfc, 1.828453757365673e-05, rtol=1e-2)


def test_engine_weight():
    engine = BasicTPEngineMapped(
        power_design=485.429,
        t_41t_design=1200,
        opr_design=8.0,
        cruise_altitude_propeller=6096.0,
        design_altitude=0.0,
        design_mach=0.2,
        prop_layout=1.0,
        bleed_control=0.0,
        itt_limit=1000.0,
        power_limit=404.524,
        opr_limit=11.0,
        speed_SL=SPEED,
        thrust_SL=THRUST_SL,
        thrust_limit_SL=THRUST_SL_LIMIT,
        efficiency_SL=EFFICIENCY_SL,
        speed_CL=SPEED,
        thrust_CL=THRUST_CL,
        thrust_limit_CL=THRUST_CL_LIMIT,
        efficiency_CL=EFFICIENCY_CL,
        effective_J=1.0,
        effective_efficiency_ls=1.0,
        effective_efficiency_cruise=1.0,
        turbo_mach_SL=MACH_ARRAY,
        turbo_thrust_SL=THRUST_ARRAY_SL,
        turbo_thrust_max_SL=THRUST_MAX_ARRAY_SL,
        turbo_sfc_SL=SFC_SL,
        turbo_mach_CL=MACH_ARRAY,
        turbo_thrust_CL=THRUST_ARRAY_CL,
        turbo_thrust_max_CL=THRUST_MAX_ARRAY_CL,
        turbo_sfc_CL=SFC_CL,
        turbo_mach_IL=MACH_ARRAY,
        turbo_thrust_IL=THRUST_ARRAY_IL,
        turbo_thrust_max_IL=THRUST_MAX_ARRAY_IL,
        turbo_sfc_IL=SFC_IL,
        level_IL=3048,
    )
    np.testing.assert_allclose(engine.compute_weight(), 415, atol=1)


def test_engine_dim():
    engine = BasicTPEngineMapped(
        power_design=485.429,
        t_41t_design=1200,
        opr_design=8.0,
        cruise_altitude_propeller=6096.0,
        design_altitude=0.0,
        design_mach=0.2,
        prop_layout=1.0,
        bleed_control=0.0,
        itt_limit=1000.0,
        power_limit=404.524,
        opr_limit=11.0,
        speed_SL=SPEED,
        thrust_SL=THRUST_SL,
        thrust_limit_SL=THRUST_SL_LIMIT,
        efficiency_SL=EFFICIENCY_SL,
        speed_CL=SPEED,
        thrust_CL=THRUST_CL,
        thrust_limit_CL=THRUST_CL_LIMIT,
        efficiency_CL=EFFICIENCY_CL,
        effective_J=1.0,
        effective_efficiency_ls=1.0,
        effective_efficiency_cruise=1.0,
        turbo_mach_SL=MACH_ARRAY,
        turbo_thrust_SL=THRUST_ARRAY_SL,
        turbo_thrust_max_SL=THRUST_MAX_ARRAY_SL,
        turbo_sfc_SL=SFC_SL,
        turbo_mach_CL=MACH_ARRAY,
        turbo_thrust_CL=THRUST_ARRAY_CL,
        turbo_thrust_max_CL=THRUST_MAX_ARRAY_CL,
        turbo_sfc_CL=SFC_CL,
        turbo_mach_IL=MACH_ARRAY,
        turbo_thrust_IL=THRUST_ARRAY_IL,
        turbo_thrust_max_IL=THRUST_MAX_ARRAY_IL,
        turbo_sfc_IL=SFC_IL,
        level_IL=3048,
    )
    np.testing.assert_allclose(engine.compute_dimensions(), [0.63, 0.60, 3.37, 8.33], atol=1e-2)


def test_compute_max_power():
    engine = BasicTPEngineMapped(
        power_design=485.429,
        t_41t_design=1200,
        opr_design=8.0,
        cruise_altitude_propeller=6096.0,
        design_altitude=0.0,
        design_mach=0.2,
        prop_layout=1.0,
        bleed_control=0.0,
        itt_limit=1000.0,
        power_limit=404.524,
        opr_limit=11.0,
        speed_SL=SPEED,
        thrust_SL=THRUST_SL,
        thrust_limit_SL=THRUST_SL_LIMIT,
        efficiency_SL=EFFICIENCY_SL,
        speed_CL=SPEED,
        thrust_CL=THRUST_CL,
        thrust_limit_CL=THRUST_CL_LIMIT,
        efficiency_CL=EFFICIENCY_CL,
        effective_J=1.0,
        effective_efficiency_ls=1.0,
        effective_efficiency_cruise=1.0,
        turbo_mach_SL=MACH_ARRAY,
        turbo_thrust_SL=THRUST_ARRAY_SL,
        turbo_thrust_max_SL=THRUST_MAX_ARRAY_SL,
        turbo_sfc_SL=SFC_SL,
        turbo_mach_CL=MACH_ARRAY,
        turbo_thrust_CL=THRUST_ARRAY_CL,
        turbo_thrust_max_CL=THRUST_MAX_ARRAY_CL,
        turbo_sfc_CL=SFC_CL,
        turbo_mach_IL=MACH_ARRAY,
        turbo_thrust_IL=THRUST_ARRAY_IL,
        turbo_thrust_max_IL=THRUST_MAX_ARRAY_IL,
        turbo_sfc_IL=SFC_IL,
        level_IL=3048,
    )
    # At design point
    flight_points = oad.FlightPoint(altitude=0, mach=0.5)
    np.testing.assert_allclose(engine.compute_max_power(flight_points), 404.52, atol=1)

    # At higher altitude
    flight_points = oad.FlightPoint(altitude=3000, mach=0.5)
    np.testing.assert_allclose(engine.compute_max_power(flight_points), 404.52, atol=1)

    # At higher altitude
    flight_points = oad.FlightPoint(altitude=6000, mach=0.5)
    np.testing.assert_allclose(engine.compute_max_power(flight_points), 404.52, atol=1)

    # At higher altitude
    flight_points = oad.FlightPoint(altitude=9000, mach=0.5)
    np.testing.assert_allclose(engine.compute_max_power(flight_points), 356.46, atol=1)

    # At higher altitude, higher mach
    flight_points = oad.FlightPoint(altitude=9000, mach=0.8)
    np.testing.assert_allclose(engine.compute_max_power(flight_points), 404.52, atol=1)


def test_compare_with_direct_computation():

    cruise_altitude_propeller = 6096.0
    cruise_speed = 125.524

    mapped_engine = BasicTPEngineMapped(
        power_design=1342.285,
        t_41t_design=1400,
        opr_design=12.0,
        cruise_altitude_propeller=8534.4,
        design_altitude=0.0,
        design_mach=0.0,
        prop_layout=1.0,
        bleed_control=1.0,
        itt_limit=1125.0,
        power_limit=634,
        opr_limit=12.0,
        speed_SL=SPEED,
        thrust_SL=THRUST_SL,
        thrust_limit_SL=THRUST_SL_LIMIT,
        efficiency_SL=EFFICIENCY_SL,
        speed_CL=SPEED,
        thrust_CL=THRUST_CL,
        thrust_limit_CL=THRUST_CL_LIMIT,
        efficiency_CL=EFFICIENCY_CL,
        effective_J=1.0,
        effective_efficiency_ls=1.0,
        effective_efficiency_cruise=1.0,
        turbo_mach_SL=MACH_ARRAY,
        turbo_thrust_SL=THRUST_ARRAY_SL,
        turbo_thrust_max_SL=THRUST_MAX_ARRAY_SL,
        turbo_sfc_SL=SFC_SL,
        turbo_mach_CL=MACH_ARRAY,
        turbo_thrust_CL=THRUST_ARRAY_CL,
        turbo_thrust_max_CL=THRUST_MAX_ARRAY_CL,
        turbo_sfc_CL=SFC_CL,
        turbo_mach_IL=MACH_ARRAY,
        turbo_thrust_IL=THRUST_ARRAY_IL,
        turbo_thrust_max_IL=THRUST_MAX_ARRAY_IL,
        turbo_sfc_IL=SFC_IL,
        level_IL=3048,
    )

    direct_engine = BasicTPEngine(
        power_design=1342.285,
        t_41t_design=1400,
        opr_design=12.0,
        cruise_altitude_propeller=8534.4,
        design_altitude=0.0,
        design_mach=0.0,
        prop_layout=1.0,
        bleed_control=1.0,
        itt_limit=1125.0,
        power_limit=634,
        opr_limit=12.0,
        speed_SL=SPEED_KA,
        thrust_SL=THRUST_SL_KA,
        thrust_limit_SL=THRUST_SL_LIMIT_KA,
        efficiency_SL=EFFICIENCY_SL_KA,
        speed_CL=SPEED_KA,
        thrust_CL=THRUST_CL_KA,
        thrust_limit_CL=THRUST_CL_LIMIT_KA,
        efficiency_CL=EFFICIENCY_CL_KA,
        effective_J=1.0,
        effective_efficiency_ls=1.0,
        effective_efficiency_cruise=1.0,
    )

    atm = Atmosphere(cruise_altitude_propeller, altitude_in_feet=False)
    mach_max = cruise_speed / atm.speed_of_sound

    machs = [
        0.70 * mach_max,
        0.75 * mach_max,
        0.80 * mach_max,
        0.85 * mach_max,
        0.90 * mach_max,
    ]
    altitudes = [
        cruise_altitude_propeller,
        cruise_altitude_propeller,
        cruise_altitude_propeller,
        cruise_altitude_propeller,
        cruise_altitude_propeller,
    ]
    thrust_rates = [0.8, 0.825, 0.85, 0.875, 0.9]
    engine_settings = [
        EngineSetting.CRUISE,
        EngineSetting.CRUISE,
        EngineSetting.CRUISE,
        EngineSetting.CRUISE,
        EngineSetting.CRUISE,
    ]  # mix EngineSetting with integers

    flight_points = oad.FlightPoint(
        mach=machs,
        altitude=altitudes,
        engine_setting=engine_settings + engine_settings,
        thrust_is_regulated=[False] * 5,
        thrust_rate=thrust_rates,
        thrust=[0.0] * 5,
    )

    flight_point_direct = copy.copy(flight_points)
    flight_point_mapped = copy.copy(flight_points)

    direct_engine.compute_flight_points(flight_point_direct)
    mapped_engine.compute_flight_points(flight_point_mapped)

    np.testing.assert_allclose(flight_point_direct.sfc, flight_point_mapped.sfc, rtol=0.05)

    # Test T/O condition

    machs = [
        0.2 * mach_max,
        0.3 * mach_max,
        0.4 * mach_max,
        0.5 * mach_max,
        0.6 * mach_max,
    ]
    altitudes = [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    thrust_rates = [1.0, 1.0, 1.0, 1.0, 1.0]
    engine_settings = [
        EngineSetting.CRUISE,
        EngineSetting.CRUISE,
        EngineSetting.CRUISE,
        EngineSetting.CRUISE,
        EngineSetting.CRUISE,
    ]  # mix EngineSetting with integers

    flight_points = oad.FlightPoint(
        mach=machs,
        altitude=altitudes,
        engine_setting=engine_settings + engine_settings,
        thrust_is_regulated=[False] * 5,
        thrust_rate=thrust_rates,
        thrust=[0.0] * 5,
    )

    flight_point_direct = copy.copy(flight_points)
    flight_point_mapped = copy.copy(flight_points)

    direct_engine.compute_flight_points(flight_point_direct)
    mapped_engine.compute_flight_points(flight_point_mapped)

    np.testing.assert_allclose(flight_point_direct.sfc, flight_point_mapped.sfc, rtol=0.05)

    # Test climb conditions

    machs = [
        0.6 * mach_max,
        0.62 * mach_max,
        0.64 * mach_max,
        0.66 * mach_max,
        0.68 * mach_max,
    ]
    altitudes = [
        1000.0,
        2000.0,
        3000.0,
        4000.0,
        5000.0,
    ]
    thrust_rates = [0.9, 0.9, 0.9, 0.9, 0.9]
    engine_settings = [
        EngineSetting.CRUISE,
        EngineSetting.CRUISE,
        EngineSetting.CRUISE,
        EngineSetting.CRUISE,
        EngineSetting.CRUISE,
    ]  # mix EngineSetting with integers

    flight_points = oad.FlightPoint(
        mach=machs,
        altitude=altitudes,
        engine_setting=engine_settings + engine_settings,
        thrust_is_regulated=[False] * 5,
        thrust_rate=thrust_rates,
        thrust=[0.0] * 5,
    )

    flight_point_direct = copy.copy(flight_points)
    flight_point_mapped = copy.copy(flight_points)

    direct_engine.compute_flight_points(flight_point_direct)
    mapped_engine.compute_flight_points(flight_point_mapped)

    np.testing.assert_allclose(flight_point_direct.sfc, flight_point_mapped.sfc, rtol=0.10)
