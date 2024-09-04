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

import time

import pytest

import numpy as np

import fastoad.api as oad
from fastoad.constants import EngineSetting
from stdatm import Atmosphere

from ..basicTP_engine import BasicTPEngine, CACHE_MAX_SIZE

INVALID_SFC = 0.0

THRUST_SL = np.array(
    [
        1399.3109379,
        2745.06172429,
        4090.81251069,
        5436.56329708,
        6782.31408347,
        8128.06486986,
        9473.81565626,
        10819.56644265,
        12165.31722904,
        13511.06801543,
        14856.81880182,
        16202.56958822,
        17548.32037461,
        18894.071161,
        20239.82194739,
        21585.57273379,
        22931.32352018,
        24277.07430657,
        25622.82509296,
        26968.57587935,
        28314.32666575,
        29660.07745214,
        31005.82823853,
        32351.57902492,
        33697.32981132,
        35043.08059771,
        36388.8313841,
        37734.58217049,
        39080.33295688,
        40426.08374328,
    ]
)
THRUST_SL_LIMIT = np.array(
    [
        11682.74241449,
        14611.57292504,
        17198.82248035,
        19485.15850098,
        21856.80051183,
        24479.74143276,
        27452.71492203,
        30935.42940484,
        35121.2987799,
        40426.08374328,
    ]
)
SPEED = np.array(
    [
        5.0,
        25.02222222,
        45.04444444,
        65.06666667,
        85.08888889,
        105.11111111,
        125.13333333,
        145.15555556,
        165.17777778,
        185.2,
    ]
)
EFFICIENCY_SL = np.array(
    [
        [
            0.15283463,
            0.15452639,
            0.14830252,
            0.135334,
            0.12233365,
            0.11075974,
            0.10053391,
            0.09138763,
            0.08593626,
            0.08593626,
            0.08593626,
            0.08593626,
            0.08593626,
            0.08593626,
            0.08593626,
            0.08593626,
            0.08593626,
            0.08593626,
            0.08593626,
            0.08593626,
            0.08593626,
            0.08593626,
            0.08593626,
            0.08593626,
            0.08593626,
            0.08593626,
            0.08593626,
            0.08593626,
            0.08593626,
            0.08593626,
        ],
        [
            0.46135675,
            0.53695266,
            0.52829611,
            0.49897443,
            0.46634025,
            0.43501997,
            0.40576673,
            0.37901633,
            0.35454218,
            0.33226898,
            0.31335727,
            0.31335727,
            0.31335727,
            0.31335727,
            0.31335727,
            0.31335727,
            0.31335727,
            0.31335727,
            0.31335727,
            0.31335727,
            0.31335727,
            0.31335727,
            0.31335727,
            0.31335727,
            0.31335727,
            0.31335727,
            0.31335727,
            0.31335727,
            0.31335727,
            0.31335727,
        ],
        [
            0.63779021,
            0.70980041,
            0.70362953,
            0.67828143,
            0.64818996,
            0.61743698,
            0.58777702,
            0.55886608,
            0.53155607,
            0.50605109,
            0.48274081,
            0.45956654,
            0.4369637,
            0.4369637,
            0.4369637,
            0.4369637,
            0.4369637,
            0.4369637,
            0.4369637,
            0.4369637,
            0.4369637,
            0.4369637,
            0.4369637,
            0.4369637,
            0.4369637,
            0.4369637,
            0.4369637,
            0.4369637,
            0.4369637,
            0.4369637,
        ],
        [
            0.72944244,
            0.79088962,
            0.78822371,
            0.7689364,
            0.74495648,
            0.71997085,
            0.69468279,
            0.6694675,
            0.64471786,
            0.62028242,
            0.59742973,
            0.57627053,
            0.55560533,
            0.53289965,
            0.51698834,
            0.51698834,
            0.51698834,
            0.51698834,
            0.51698834,
            0.51698834,
            0.51698834,
            0.51698834,
            0.51698834,
            0.51698834,
            0.51698834,
            0.51698834,
            0.51698834,
            0.51698834,
            0.51698834,
            0.51698834,
        ],
        [
            0.76811899,
            0.82810319,
            0.83082104,
            0.81797549,
            0.79963127,
            0.77991319,
            0.75992698,
            0.73907187,
            0.71812541,
            0.69742928,
            0.67629648,
            0.65630614,
            0.63784431,
            0.61974103,
            0.60132763,
            0.57863011,
            0.57032798,
            0.57032798,
            0.57032798,
            0.57032798,
            0.57032798,
            0.57032798,
            0.57032798,
            0.57032798,
            0.57032798,
            0.57032798,
            0.57032798,
            0.57032798,
            0.57032798,
            0.57032798,
        ],
        [
            0.78279891,
            0.84481931,
            0.85321449,
            0.84616679,
            0.83297385,
            0.81729385,
            0.80132825,
            0.78504319,
            0.76756693,
            0.75000684,
            0.73253809,
            0.71458567,
            0.6968801,
            0.68070518,
            0.66492545,
            0.6492067,
            0.63262969,
            0.61113683,
            0.60346574,
            0.60346574,
            0.60346574,
            0.60346574,
            0.60346574,
            0.60346574,
            0.60346574,
            0.60346574,
            0.60346574,
            0.60346574,
            0.60346574,
            0.60346574,
        ],
        [
            0.78294283,
            0.85231102,
            0.86560197,
            0.86322073,
            0.85437618,
            0.84245224,
            0.82914617,
            0.81597218,
            0.80225071,
            0.78741629,
            0.77246842,
            0.75753235,
            0.74241906,
            0.72663768,
            0.71197174,
            0.69814788,
            0.68439461,
            0.67061217,
            0.65604093,
            0.63789473,
            0.62505987,
            0.62505987,
            0.62505987,
            0.62505987,
            0.62505987,
            0.62505987,
            0.62505987,
            0.62505987,
            0.62505987,
            0.62505987,
        ],
        [
            0.78073195,
            0.85415907,
            0.87241362,
            0.87400597,
            0.86865207,
            0.85982485,
            0.84917727,
            0.83769474,
            0.82645019,
            0.81473022,
            0.8020143,
            0.78907884,
            0.77610452,
            0.76317049,
            0.7496935,
            0.73600993,
            0.72336693,
            0.7112559,
            0.69914555,
            0.68701232,
            0.6745434,
            0.66017031,
            0.6383931,
            0.6383931,
            0.6383931,
            0.6383931,
            0.6383931,
            0.6383931,
            0.6383931,
            0.6383931,
        ],
        [
            0.77128239,
            0.8533285,
            0.87597258,
            0.88064439,
            0.87818155,
            0.87189393,
            0.86356588,
            0.85405123,
            0.84391425,
            0.83398151,
            0.82396,
            0.8130076,
            0.80167879,
            0.7903354,
            0.77893959,
            0.76754401,
            0.75564435,
            0.74358209,
            0.73224752,
            0.72146189,
            0.71076397,
            0.70004991,
            0.68925036,
            0.67800994,
            0.66509943,
            0.647162,
            0.64233779,
            0.64233779,
            0.64233779,
            0.64233779,
        ],
        [
            0.758981,
            0.85018925,
            0.87651111,
            0.8842638,
            0.88410527,
            0.87999373,
            0.8736631,
            0.86595054,
            0.85740225,
            0.84837297,
            0.83931893,
            0.83044327,
            0.82116973,
            0.81129194,
            0.80123575,
            0.79116924,
            0.78109535,
            0.77102421,
            0.76071946,
            0.75002658,
            0.73960475,
            0.72967581,
            0.7200989,
            0.71059827,
            0.70111703,
            0.69156965,
            0.68183631,
            0.67139259,
            0.6594064,
            0.63922386,
        ],
    ]
)

THRUST_CL = np.array(
    [
        1099.32399705,
        2206.99792872,
        3314.67186039,
        4422.34579207,
        5530.01972374,
        6637.69365541,
        7745.36758708,
        8853.04151876,
        9960.71545043,
        11068.3893821,
        12176.06331377,
        13283.73724545,
        14391.41117712,
        15499.08510879,
        16606.75904046,
        17714.43297214,
        18822.10690381,
        19929.78083548,
        21037.45476715,
        22145.12869883,
        23252.8026305,
        24360.47656217,
        25468.15049384,
        26575.82442552,
        27683.49835719,
        28791.17228886,
        29898.84622053,
        31006.52015221,
        32114.19408388,
        33221.86801555,
    ]
)
THRUST_CL_LIMIT = np.array(
    [
        9217.56041878,
        11539.96366815,
        13609.23222686,
        15446.62989058,
        17371.45454046,
        19504.33623191,
        21959.93802672,
        24863.95768927,
        28454.47517536,
        33221.86801555,
    ]
)
EFFICIENCY_CL = np.array(
    [
        [
            0.1525017,
            0.15455295,
            0.14719451,
            0.13345396,
            0.12005581,
            0.10825916,
            0.09790557,
            0.08865142,
            0.0857486,
            0.0857486,
            0.0857486,
            0.0857486,
            0.0857486,
            0.0857486,
            0.0857486,
            0.0857486,
            0.0857486,
            0.0857486,
            0.0857486,
            0.0857486,
            0.0857486,
            0.0857486,
            0.0857486,
            0.0857486,
            0.0857486,
            0.0857486,
            0.0857486,
            0.0857486,
            0.0857486,
            0.0857486,
        ],
        [
            0.4609688,
            0.53755163,
            0.52595002,
            0.49438963,
            0.46026748,
            0.42799224,
            0.39807485,
            0.37087108,
            0.34610632,
            0.32319767,
            0.31263268,
            0.31263268,
            0.31263268,
            0.31263268,
            0.31263268,
            0.31263268,
            0.31263268,
            0.31263268,
            0.31263268,
            0.31263268,
            0.31263268,
            0.31263268,
            0.31263268,
            0.31263268,
            0.31263268,
            0.31263268,
            0.31263268,
            0.31263268,
            0.31263268,
            0.31263268,
        ],
        [
            0.63745357,
            0.71030332,
            0.70161376,
            0.67416454,
            0.64233586,
            0.61032813,
            0.57955611,
            0.54981889,
            0.52191414,
            0.49612599,
            0.472251,
            0.44710768,
            0.4358789,
            0.4358789,
            0.4358789,
            0.4358789,
            0.4358789,
            0.4358789,
            0.4358789,
            0.4358789,
            0.4358789,
            0.4358789,
            0.4358789,
            0.4358789,
            0.4358789,
            0.4358789,
            0.4358789,
            0.4358789,
            0.4358789,
            0.4358789,
        ],
        [
            0.72927209,
            0.79130746,
            0.78671589,
            0.76561441,
            0.74023414,
            0.71391805,
            0.68751919,
            0.66135128,
            0.63551825,
            0.61059879,
            0.5874851,
            0.56580739,
            0.54396344,
            0.51562053,
            0.51562053,
            0.51562053,
            0.51562053,
            0.51562053,
            0.51562053,
            0.51562053,
            0.51562053,
            0.51562053,
            0.51562053,
            0.51562053,
            0.51562053,
            0.51562053,
            0.51562053,
            0.51562053,
            0.51562053,
            0.51562053,
        ],
        [
            0.76793801,
            0.82881551,
            0.82991017,
            0.81547853,
            0.79581403,
            0.77509564,
            0.75397934,
            0.73210201,
            0.71035855,
            0.68866829,
            0.66690561,
            0.64682454,
            0.62782948,
            0.60906541,
            0.58867105,
            0.56604834,
            0.56604834,
            0.56604834,
            0.56604834,
            0.56604834,
            0.56604834,
            0.56604834,
            0.56604834,
            0.56604834,
            0.56604834,
            0.56604834,
            0.56604834,
            0.56604834,
            0.56604834,
            0.56604834,
        ],
        [
            0.78212354,
            0.84570749,
            0.85276048,
            0.84434714,
            0.82999362,
            0.81330374,
            0.79654942,
            0.77920051,
            0.76086879,
            0.74261708,
            0.72440042,
            0.70558248,
            0.68780014,
            0.6712616,
            0.65490663,
            0.63844401,
            0.61955778,
            0.60130725,
            0.60130725,
            0.60130725,
            0.60130725,
            0.60130725,
            0.60130725,
            0.60130725,
            0.60130725,
            0.60130725,
            0.60130725,
            0.60130725,
            0.60130725,
            0.60130725,
        ],
        [
            0.782087,
            0.8531705,
            0.86549898,
            0.86201488,
            0.85208882,
            0.83918164,
            0.82509663,
            0.81123192,
            0.79654801,
            0.7809917,
            0.76542586,
            0.74990449,
            0.73389929,
            0.71781824,
            0.70303855,
            0.68874901,
            0.67447937,
            0.66001904,
            0.64359013,
            0.6222916,
            0.6222916,
            0.6222916,
            0.6222916,
            0.6222916,
            0.6222916,
            0.6222916,
            0.6222916,
            0.6222916,
            0.6222916,
            0.6222916,
        ],
        [
            0.77919776,
            0.8553569,
            0.87267871,
            0.87319531,
            0.86685981,
            0.85715348,
            0.84573005,
            0.83358171,
            0.82170864,
            0.80918226,
            0.79581163,
            0.78234797,
            0.76886838,
            0.75536466,
            0.74118215,
            0.72734031,
            0.71449959,
            0.70197214,
            0.68942433,
            0.67679352,
            0.66331158,
            0.64683178,
            0.63468405,
            0.63468405,
            0.63468405,
            0.63468405,
            0.63468405,
            0.63468405,
            0.63468405,
            0.63468405,
        ],
        [
            0.77113386,
            0.85459349,
            0.8763654,
            0.880199,
            0.87679414,
            0.86973444,
            0.8606486,
            0.85046808,
            0.83973112,
            0.82923702,
            0.81852944,
            0.80697934,
            0.79517576,
            0.78337645,
            0.77156959,
            0.75967682,
            0.7472486,
            0.73498721,
            0.72344489,
            0.71233643,
            0.70127338,
            0.69018811,
            0.67895927,
            0.66693785,
            0.65278754,
            0.63700179,
            0.63700179,
            0.63700179,
            0.63700179,
            0.63700179,
        ],
        [
            0.75744225,
            0.85146396,
            0.87724551,
            0.88401495,
            0.88303569,
            0.87818945,
            0.87117639,
            0.86283169,
            0.85369075,
            0.84409768,
            0.83452151,
            0.82507678,
            0.81528304,
            0.80493728,
            0.79445786,
            0.7839991,
            0.77357618,
            0.76315618,
            0.75243645,
            0.74147991,
            0.73083799,
            0.72065277,
            0.71073914,
            0.70091875,
            0.69114959,
            0.68131338,
            0.67123745,
            0.66039378,
            0.64811805,
            0.63000836,
        ],
    ]
)


def test_compute_flight_points_tbm_700():
    engine = BasicTPEngine(
        power_design=745.7,
        t_41t_design=1350,
        opr_design=9.5,
        cruise_altitude_propeller=9200.0,
        design_altitude=0.0,
        design_mach=0.0,
        prop_layout=1.0,
        bleed_control=1.0,
        itt_limit=1100.0,
        power_limit=521.99,
        opr_limit=12.0,
        speed_SL=SPEED,
        thrust_SL=THRUST_SL,
        thrust_limit_SL=THRUST_SL_LIMIT,
        efficiency_SL=EFFICIENCY_SL,
        speed_CL=SPEED,
        thrust_CL=THRUST_CL,
        thrust_limit_CL=THRUST_CL_LIMIT,
        efficiency_CL=EFFICIENCY_CL,
        effective_J=1.0,  # Effective advance ratio factor
        effective_efficiency_ls=1.0,  # Effective efficiency in low speed conditions
        effective_efficiency_cruise=1.0,  # Effective efficiency in cruise conditions
        eta_225=0.85,
        eta_253=0.86,
        eta_445=0.86,
        eta_455=0.86,
        eta_q=43.260e6 * 0.95,
        eta_axe=0.98,
        pi_02=0.8,
        pi_cc=0.95,
        cooling_ratio=0.05,
        hp_shaft_power_out=50 * 745.7,
        gearbox_efficiency=0.98,
        inter_compressor_bleed=0.04,
        exhaust_mach_design=0.4,
        pr_1_ratio_design=0.25,
    )  # load a 1000 kW turboprop gasoline engine

    altitude_array = np.array([0, 5000, 10000, 15000, 20000, 25000, 30000]) * 0.3048  # ft in m
    speed_array = np.array([230, 241, 253, 267, 281, 298, 298]) * 0.5144  # kts in m/s
    atm = Atmosphere(altitude_array, altitude_in_feet=False)
    atm.true_airspeed = speed_array
    mach_array = atm.mach
    thrust_rate = np.full_like(altitude_array, 1.0)
    engine_settings = [
        EngineSetting.TAKEOFF,
        EngineSetting.TAKEOFF,
        EngineSetting.TAKEOFF,
        EngineSetting.TAKEOFF,
        EngineSetting.TAKEOFF,
        EngineSetting.TAKEOFF,
        EngineSetting.TAKEOFF,
    ]

    flight_points = oad.FlightPoint(
        mach=mach_array,
        altitude=altitude_array,
        engine_setting=engine_settings,
        thrust_rate=thrust_rate,
    )

    engine.compute_flight_points(flight_points)
    np.testing.assert_allclose(
        flight_points.thrust / flight_points.thrust_rate,
        [4050.690258, 3885.650828, 3724.108916, 3554.889068, 3408.538184, 3250.217239, 2793.279837],
        rtol=1e-2,
    )
    np.testing.assert_allclose(
        flight_points.sfc * flight_points.thrust * 3600.0,
        [233.831317, 217.645651, 203.416703, 190.898895, 179.773042, 170.306647, 144.282117],
        rtol=1e-2,
    )


def test_compute_flight_points():
    engine = BasicTPEngine(
        power_design=745.7,
        t_41t_design=1350,
        opr_design=9.5,
        cruise_altitude_propeller=9000.0,
        design_altitude=0.0,
        design_mach=0.5,
        prop_layout=1.0,
        bleed_control=1.0,
        itt_limit=1100.0,
        power_limit=521.99,
        opr_limit=12.0,
        speed_SL=SPEED,
        thrust_SL=THRUST_SL,
        thrust_limit_SL=THRUST_SL_LIMIT,
        efficiency_SL=EFFICIENCY_SL,
        speed_CL=SPEED,
        thrust_CL=THRUST_CL,
        thrust_limit_CL=THRUST_CL_LIMIT,
        efficiency_CL=EFFICIENCY_CL,
        effective_J=0.95,  # Effective advance ratio factor
        effective_efficiency_ls=0.97,  # Effective efficiency in low speed conditions
        effective_efficiency_cruise=0.98,  # Effective efficiency in cruise conditions
        eta_225=0.85,
        eta_253=0.86,
        eta_445=0.86,
        eta_455=0.86,
        eta_q=43.260e6 * 0.95,
        eta_axe=0.98,
        pi_02=0.8,
        pi_cc=0.95,
        cooling_ratio=0.05,
        hp_shaft_power_out=50 * 745.7,
        gearbox_efficiency=0.98,
        inter_compressor_bleed=0.04,
        exhaust_mach_design=0.4,
        pr_1_ratio_design=0.25,
    )  # load a 1000 kW turboprop gasoline engine

    # Test full arrays
    # 2D arrays are used, where first line is for thrust rates, and second line
    # is for thrust values.
    # As thrust rates and thrust values match, thrust rate results are 2 equal
    # lines and so are thrust value results.
    machs = [0.3, 0.3, 0.3, 0.4, 0.4]
    altitudes = [0, 0, 0, 1000, 2400]
    thrust_rates = [0.8, 0.5, 0.5, 0.4, 0.7]
    thrusts = [3562.23389399, 2226.39618374, 2226.39618374, 1371.93021539, 2462.66657075]
    engine_settings = [
        EngineSetting.TAKEOFF,
        EngineSetting.TAKEOFF,
        EngineSetting.CLIMB,
        EngineSetting.IDLE,
        EngineSetting.CRUISE,
    ]  # mix EngineSetting with integers
    expected_sfc = [1.469635e-05, 1.828701e-05, 1.828701e-05, 2.614570e-05, 1.757415e-05]

    flight_points = oad.FlightPoint(
        mach=machs + machs,
        altitude=altitudes + altitudes,
        engine_setting=engine_settings + engine_settings,
        thrust_is_regulated=[False] * 5 + [True] * 5,
        thrust_rate=thrust_rates + [0.0] * 5,
        thrust=[0.0] * 5 + thrusts,
    )

    engine.compute_flight_points(flight_points)
    np.testing.assert_allclose(flight_points.sfc, expected_sfc + expected_sfc, rtol=1e-2)


def test_engine_weight():
    _745_kW_engine = BasicTPEngine(
        power_design=745.7,
        t_41t_design=1350,
        opr_design=9.5,
        cruise_altitude_propeller=9000.0,
        design_altitude=0.0,
        design_mach=0.5,
        prop_layout=1.0,
        bleed_control=1.0,
        itt_limit=1100.0,
        power_limit=521.99,
        opr_limit=12.0,
        speed_SL=SPEED,
        thrust_SL=THRUST_SL,
        thrust_limit_SL=THRUST_SL_LIMIT,
        efficiency_SL=EFFICIENCY_SL,
        speed_CL=SPEED,
        thrust_CL=THRUST_CL,
        thrust_limit_CL=THRUST_CL_LIMIT,
        efficiency_CL=EFFICIENCY_CL,
        effective_J=0.95,  # Effective advance ratio factor
        effective_efficiency_ls=0.97,  # Effective efficiency in low speed conditions
        effective_efficiency_cruise=0.98,  # Effective efficiency in cruise conditions
        eta_225=0.85,
        eta_253=0.86,
        eta_445=0.86,
        eta_455=0.86,
        eta_q=43.260e6 * 0.95,
        eta_axe=0.98,
        pi_02=0.8,
        pi_cc=0.95,
        cooling_ratio=0.05,
        hp_shaft_power_out=50 * 745.7,
        gearbox_efficiency=0.98,
        inter_compressor_bleed=0.04,
        exhaust_mach_design=0.4,
        pr_1_ratio_design=0.25,
    )
    np.testing.assert_allclose(_745_kW_engine.compute_weight(), 438, atol=1)
    _1342kw_engine = BasicTPEngine(
        power_design=1342.26,
        t_41t_design=1600.0,
        opr_design=11.4,
        cruise_altitude_propeller=9000.0,
        design_altitude=0.0,
        design_mach=0.5,
        prop_layout=1.0,
        bleed_control=1.0,
        itt_limit=1100.0,
        power_limit=521.99,
        opr_limit=12.0,
        speed_SL=SPEED,
        thrust_SL=THRUST_SL,
        thrust_limit_SL=THRUST_SL_LIMIT,
        efficiency_SL=EFFICIENCY_SL,
        speed_CL=SPEED,
        thrust_CL=THRUST_CL,
        thrust_limit_CL=THRUST_CL_LIMIT,
        efficiency_CL=EFFICIENCY_CL,
        effective_J=0.95,  # Effective advance ratio factor
        effective_efficiency_ls=0.97,  # Effective efficiency in low speed conditions
        effective_efficiency_cruise=0.98,  # Effective efficiency in cruise conditions
        eta_225=0.85,
        eta_253=0.86,
        eta_445=0.86,
        eta_455=0.86,
        eta_q=43.260e6 * 0.95,
        eta_axe=0.98,
        pi_02=0.8,
        pi_cc=0.95,
        cooling_ratio=0.05,
        hp_shaft_power_out=50 * 745.7,
        gearbox_efficiency=0.98,
        inter_compressor_bleed=0.04,
        exhaust_mach_design=0.4,
        pr_1_ratio_design=0.25,
    )
    np.testing.assert_allclose(_1342kw_engine.compute_weight(), 438, atol=1)


def test_engine_dim():
    _745_kW_engine = BasicTPEngine(
        power_design=745.7,
        t_41t_design=1350,
        opr_design=9.5,
        cruise_altitude_propeller=9000.0,
        design_altitude=0.0,
        design_mach=0.5,
        prop_layout=1.0,
        bleed_control=1.0,
        itt_limit=1100.0,
        power_limit=521.99,
        opr_limit=12.0,
        speed_SL=SPEED,
        thrust_SL=THRUST_SL,
        thrust_limit_SL=THRUST_SL_LIMIT,
        efficiency_SL=EFFICIENCY_SL,
        speed_CL=SPEED,
        thrust_CL=THRUST_CL,
        thrust_limit_CL=THRUST_CL_LIMIT,
        efficiency_CL=EFFICIENCY_CL,
        effective_J=0.95,  # Effective advance ratio factor
        effective_efficiency_ls=0.97,  # Effective efficiency in low speed conditions
        effective_efficiency_cruise=0.98,  # Effective efficiency in cruise conditions
        eta_225=0.85,
        eta_253=0.86,
        eta_445=0.86,
        eta_455=0.86,
        eta_q=43.260e6 * 0.95,
        eta_axe=0.98,
        pi_02=0.8,
        pi_cc=0.95,
        cooling_ratio=0.05,
        hp_shaft_power_out=50 * 745.7,
        gearbox_efficiency=0.98,
        inter_compressor_bleed=0.04,
        exhaust_mach_design=0.4,
        pr_1_ratio_design=0.25,
    )
    np.testing.assert_allclose(
        _745_kW_engine.compute_dimensions(), [0.61, 0.54, 3.54, 8.23], atol=1e-2
    )
    _1342kw_engine = BasicTPEngine(
        power_design=1342.26,
        t_41t_design=1600.0,
        opr_design=11.4,
        cruise_altitude_propeller=9000.0,
        design_altitude=0.0,
        design_mach=0.5,
        prop_layout=1.0,
        bleed_control=1.0,
        itt_limit=1100.0,
        power_limit=521.99,
        opr_limit=12.0,
        speed_SL=SPEED,
        thrust_SL=THRUST_SL,
        thrust_limit_SL=THRUST_SL_LIMIT,
        efficiency_SL=EFFICIENCY_SL,
        speed_CL=SPEED,
        thrust_CL=THRUST_CL,
        thrust_limit_CL=THRUST_CL_LIMIT,
        efficiency_CL=EFFICIENCY_CL,
        effective_J=0.95,  # Effective advance ratio factor
        effective_efficiency_ls=0.97,  # Effective efficiency in low speed conditions
        effective_efficiency_cruise=0.98,  # Effective efficiency in cruise conditions
        eta_225=0.85,
        eta_253=0.86,
        eta_445=0.86,
        eta_455=0.86,
        eta_q=43.260e6 * 0.95,
        eta_axe=0.98,
        pi_02=0.8,
        pi_cc=0.95,
        cooling_ratio=0.05,
        hp_shaft_power_out=50 * 745.7,
        gearbox_efficiency=0.98,
        inter_compressor_bleed=0.04,
        exhaust_mach_design=0.4,
        pr_1_ratio_design=0.25,
    )
    np.testing.assert_allclose(
        _1342kw_engine.compute_dimensions(), [0.61, 0.54, 3.77, 8.75], atol=1e-2
    )


def test_compute_max_power():
    _745_kW_engine = BasicTPEngine(
        power_design=745.7,
        t_41t_design=1350,
        opr_design=9.5,
        cruise_altitude_propeller=9000.0,
        design_altitude=0.0,
        design_mach=0.5,
        prop_layout=1.0,
        bleed_control=1.0,
        itt_limit=1100.0,
        power_limit=521.99,
        opr_limit=12.0,
        speed_SL=SPEED,
        thrust_SL=THRUST_SL,
        thrust_limit_SL=THRUST_SL_LIMIT,
        efficiency_SL=EFFICIENCY_SL,
        speed_CL=SPEED,
        thrust_CL=THRUST_CL,
        thrust_limit_CL=THRUST_CL_LIMIT,
        efficiency_CL=EFFICIENCY_CL,
        effective_J=0.95,  # Effective advance ratio factor
        effective_efficiency_ls=0.97,  # Effective efficiency in low speed conditions
        effective_efficiency_cruise=0.98,  # Effective efficiency in cruise conditions
        eta_225=0.85,
        eta_253=0.86,
        eta_445=0.86,
        eta_455=0.86,
        eta_q=43.260e6 * 0.95,
        eta_axe=0.98,
        pi_02=0.8,
        pi_cc=0.95,
        cooling_ratio=0.05,
        hp_shaft_power_out=50 * 745.7,
        gearbox_efficiency=0.98,
        inter_compressor_bleed=0.04,
        exhaust_mach_design=0.4,
        pr_1_ratio_design=0.25,
    )
    # At design point
    flight_points = oad.FlightPoint(altitude=0, mach=0.5)
    np.testing.assert_allclose(_745_kW_engine.compute_max_power(flight_points), 521.99, atol=1)

    # At higher altitude
    flight_points = oad.FlightPoint(altitude=3000, mach=0.5)
    np.testing.assert_allclose(_745_kW_engine.compute_max_power(flight_points), 521.99, atol=1)

    # At higher altitude
    flight_points = oad.FlightPoint(altitude=6000, mach=0.5)
    np.testing.assert_allclose(_745_kW_engine.compute_max_power(flight_points), 521.99, atol=1)

    # At higher altitude
    flight_points = oad.FlightPoint(altitude=9000, mach=0.5)
    np.testing.assert_allclose(_745_kW_engine.compute_max_power(flight_points), 337.57, atol=1)


def test_nested_problem_setup():
    engine = BasicTPEngine(
        power_design=745.7,
        t_41t_design=1350,
        opr_design=9.5,
        cruise_altitude_propeller=9000.0,
        design_altitude=0.0,
        design_mach=0.5,
        prop_layout=1.0,
        bleed_control=1.0,
        itt_limit=1100.0,
        power_limit=521.99,
        opr_limit=12.0,
        speed_SL=SPEED,
        thrust_SL=THRUST_SL,
        thrust_limit_SL=THRUST_SL_LIMIT,
        efficiency_SL=EFFICIENCY_SL,
        speed_CL=SPEED,
        thrust_CL=THRUST_CL,
        thrust_limit_CL=THRUST_CL_LIMIT,
        efficiency_CL=EFFICIENCY_CL,
        effective_J=0.95,  # Effective advance ratio factor
        effective_efficiency_ls=0.97,  # Effective efficiency in low speed conditions
        effective_efficiency_cruise=0.98,  # Effective efficiency in cruise conditions
        eta_225=0.85,
        eta_253=0.86,
        eta_445=0.86,
        eta_455=0.86,
        eta_q=43.260e6 * 0.95,
        eta_axe=0.98,
        pi_02=0.8,
        pi_cc=0.95,
        cooling_ratio=0.05,
        hp_shaft_power_out=50 * 745.7,
        gearbox_efficiency=0.98,
        inter_compressor_bleed=0.04,
        exhaust_mach_design=0.4,
        pr_1_ratio_design=0.25,
    )  # load a 1000 kW turboprop gasoline engine

    # Sizing problem should not be loaded at object instantiation
    assert engine._turboprop_sizing_problem is None
    assert not engine._turboprop_sizing_problem_setup

    assert engine._turboprop_max_thrust_power_limit_problem is None
    assert not engine._turboprop_max_thrust_power_limit_problem_setup

    assert engine._turboprop_max_thrust_opr_limit_problem is None
    assert not engine._turboprop_max_thrust_opr_limit_problem_setup

    assert engine._turboprop_max_thrust_itt_limit_problem is None
    assert not engine._turboprop_max_thrust_itt_limit_problem_setup

    assert engine._turboprop_max_thrust_propeller_thrust_limit_problem is None
    assert not engine._turboprop_max_thrust_propeller_thrust_limit_problem_setup

    assert engine._alpha is None
    assert engine._alpha_p is None
    assert engine._a_41 is None
    assert engine._a_45 is None
    assert engine._a_8 is None
    assert engine._opr_2_opr_1 is None

    # Nor after computing the weight, dimensions or cd0

    _ = engine.compute_weight()
    _, _, _, _ = engine.compute_dimensions()
    _ = engine.compute_drag(mach=0.2, unit_reynolds=5e6, wing_mac=1.0)

    # Sizing problem should not be loaded at object instantiation
    assert engine._turboprop_sizing_problem is None
    assert not engine._turboprop_sizing_problem_setup

    assert engine._turboprop_max_thrust_power_limit_problem is None
    assert not engine._turboprop_max_thrust_power_limit_problem_setup

    assert engine._turboprop_max_thrust_opr_limit_problem is None
    assert not engine._turboprop_max_thrust_opr_limit_problem_setup

    assert engine._turboprop_max_thrust_itt_limit_problem is None
    assert not engine._turboprop_max_thrust_itt_limit_problem_setup

    assert engine._turboprop_max_thrust_propeller_thrust_limit_problem is None
    assert not engine._turboprop_max_thrust_propeller_thrust_limit_problem_setup

    assert engine._alpha is None
    assert engine._alpha_p is None
    assert engine._a_41 is None
    assert engine._a_45 is None
    assert engine._a_8 is None
    assert engine._opr_2_opr_1 is None

    # Then, the first call should take quite a bit of time, while the second is instantaneous
    t1 = time.time()
    _ = engine.turboprop_sizing_problem
    t2 = time.time()

    _ = engine.turboprop_sizing_problem
    t3 = time.time()

    assert t3 - t2 < t2 - t1


def test_access_to_geometry_parameter():
    engine = BasicTPEngine(
        power_design=745.7,
        t_41t_design=1350,
        opr_design=9.5,
        cruise_altitude_propeller=9200.0,
        design_altitude=0.0,
        design_mach=0.0,
        prop_layout=1.0,
        bleed_control=1.0,
        itt_limit=1100.0,
        power_limit=521.99,
        opr_limit=12.0,
        speed_SL=SPEED,
        thrust_SL=THRUST_SL,
        thrust_limit_SL=THRUST_SL_LIMIT,
        efficiency_SL=EFFICIENCY_SL,
        speed_CL=SPEED,
        thrust_CL=THRUST_CL,
        thrust_limit_CL=THRUST_CL_LIMIT,
        efficiency_CL=EFFICIENCY_CL,
        effective_J=1.0,  # Effective advance ratio factor
        effective_efficiency_ls=1.0,  # Effective efficiency in low speed conditions
        effective_efficiency_cruise=1.0,  # Effective efficiency in cruise conditions
        eta_225=0.85,
        eta_253=0.86,
        eta_445=0.86,
        eta_455=0.86,
        eta_q=43.260e6 * 0.95,
        eta_axe=0.98,
        pi_02=0.8,
        pi_cc=0.95,
        cooling_ratio=0.05,
        hp_shaft_power_out=50 * 745.7,
        gearbox_efficiency=0.98,
        inter_compressor_bleed=0.04,
        exhaust_mach_design=0.4,
        pr_1_ratio_design=0.25,
    )  # load a 1000 kW turboprop gasoline engine

    # Sizing problem should not be loaded at object instantiation
    assert engine._turboprop_sizing_problem is None
    assert not engine._turboprop_sizing_problem_setup

    assert engine._alpha is None
    assert engine._alpha_p is None
    assert engine._a_41 is None
    assert engine._a_45 is None
    assert engine._a_8 is None
    assert engine._opr_2_opr_1 is None

    # But accessing alpha should load the sizing problem and run it. Additionally, accessing a
    # geometric parameter should take some time the first time its done but accessing any other
    # after that should be instantaneous
    t1 = time.time()
    assert engine.alpha == pytest.approx(0.798550, rel=1e-2)
    t2 = time.time()

    assert engine.alpha_p == pytest.approx(0.333594, rel=1e-2)
    assert engine.a_41 == pytest.approx(0.004571, rel=1e-2)
    assert engine.a_45 == pytest.approx(0.012201, rel=1e-2)
    assert engine.a_8 == pytest.approx(0.038730, rel=1e-2)
    assert engine.opr_2_opr_1 == pytest.approx(1.684210, rel=1e-2)
    t3 = time.time()

    assert t3 - t2 < t2 - t1

    assert engine._turboprop_sizing_problem is not None
    assert engine._turboprop_sizing_problem_setup


def test_geometry_parameter_not_called_until_compute_flight_point():
    engine = BasicTPEngine(
        power_design=745.7,
        t_41t_design=1350,
        opr_design=9.5,
        cruise_altitude_propeller=9200.0,
        design_altitude=0.0,
        design_mach=0.0,
        prop_layout=1.0,
        bleed_control=1.0,
        itt_limit=1100.0,
        power_limit=521.99,
        opr_limit=12.0,
        speed_SL=SPEED,
        thrust_SL=THRUST_SL,
        thrust_limit_SL=THRUST_SL_LIMIT,
        efficiency_SL=EFFICIENCY_SL,
        speed_CL=SPEED,
        thrust_CL=THRUST_CL,
        thrust_limit_CL=THRUST_CL_LIMIT,
        efficiency_CL=EFFICIENCY_CL,
        effective_J=1.0,  # Effective advance ratio factor
        effective_efficiency_ls=1.0,  # Effective efficiency in low speed conditions
        effective_efficiency_cruise=1.0,  # Effective efficiency in cruise conditions
        eta_225=0.85,
        eta_253=0.86,
        eta_445=0.86,
        eta_455=0.86,
        eta_q=43.260e6 * 0.95,
        eta_axe=0.98,
        pi_02=0.8,
        pi_cc=0.95,
        cooling_ratio=0.05,
        hp_shaft_power_out=50 * 745.7,
        gearbox_efficiency=0.98,
        inter_compressor_bleed=0.04,
        exhaust_mach_design=0.4,
        pr_1_ratio_design=0.25,
    )  # load a 1000 kW turboprop gasoline engine

    assert engine._alpha is None
    assert engine._alpha_p is None
    assert engine._a_41 is None
    assert engine._a_45 is None
    assert engine._a_8 is None
    assert engine._opr_2_opr_1 is None

    flight_points = oad.FlightPoint(
        mach=[0.2],
        altitude=[0],
        engine_setting=[EngineSetting.TAKEOFF],
        thrust_rate=[1.0],
    )

    engine.compute_flight_points(flight_points)

    assert engine._a_41 == pytest.approx(0.004571, rel=1e-2)
    assert engine._a_45 == pytest.approx(0.012201, rel=1e-2)
    assert engine._a_8 == pytest.approx(0.038730, rel=1e-2)


def test_max_thrust_all_engine_limits():
    """
    For this engine the propeller thrust is never limiting so we can't craft a case where it
    is. However we can check that it is always below the max value.
    """

    engine = BasicTPEngine(
        power_design=745.7,
        t_41t_design=1350,
        opr_design=9.5,
        cruise_altitude_propeller=9200.0,
        design_altitude=0.0,
        design_mach=0.0,
        prop_layout=1.0,
        bleed_control=1.0,
        itt_limit=1100.0,
        power_limit=521.99,
        opr_limit=12.0,
        speed_SL=SPEED,
        thrust_SL=THRUST_SL,
        thrust_limit_SL=THRUST_SL_LIMIT,
        efficiency_SL=EFFICIENCY_SL,
        speed_CL=SPEED,
        thrust_CL=THRUST_CL,
        thrust_limit_CL=THRUST_CL_LIMIT,
        efficiency_CL=EFFICIENCY_CL,
        effective_J=1.0,  # Effective advance ratio factor
        effective_efficiency_ls=1.0,  # Effective efficiency in low speed conditions
        effective_efficiency_cruise=1.0,  # Effective efficiency in cruise conditions
        eta_225=0.85,
        eta_253=0.86,
        eta_445=0.86,
        eta_455=0.86,
        eta_q=43.260e6 * 0.95,
        eta_axe=0.98,
        pi_02=0.8,
        pi_cc=0.95,
        cooling_ratio=0.05,
        hp_shaft_power_out=50 * 745.7,
        gearbox_efficiency=0.98,
        inter_compressor_bleed=0.04,
        exhaust_mach_design=0.4,
        pr_1_ratio_design=0.25,
    )  # load a 1000 kW turboprop gasoline engine

    # Not loaded until we try to run the computation of max thrust
    assert engine._a_41 is None
    assert engine._a_45 is None
    assert engine._a_8 is None

    prob_max_thrust_power_limit = engine.turboprop_max_thrust_power_limit_problem

    # Power should be the limit in this case
    prob_max_thrust_power_limit.set_val("altitude", val=0, units="ft")
    prob_max_thrust_power_limit.set_val("mach_0", val=0.34767454)

    prob_max_thrust_power_limit.run_model()

    assert prob_max_thrust_power_limit.get_val("required_thrust", units="kN") == pytest.approx(
        4.050775, rel=1e-2
    )
    assert prob_max_thrust_power_limit.get_val("shaft_power", units="kW") == pytest.approx(
        prob_max_thrust_power_limit.get_val("shaft_power_limit", units="kW"), rel=1e-2
    )
    assert prob_max_thrust_power_limit.get_val("opr") < prob_max_thrust_power_limit.get_val(
        "opr_limit"
    )
    assert prob_max_thrust_power_limit.get_val(
        "total_temperature_45", units="degK"
    ) < prob_max_thrust_power_limit.get_val("itt_limit", units="degK")
    assert prob_max_thrust_power_limit.get_val(
        "propeller_thrust", units="N"
    ) < prob_max_thrust_power_limit.get_val("propeller_max_thrust", units="N")

    # Area should now be loaded
    assert engine._a_41 == pytest.approx(0.004571, rel=1e-2)
    assert engine._a_45 == pytest.approx(0.012201, rel=1e-2)
    assert engine._a_8 == pytest.approx(0.038730, rel=1e-2)

    prob_max_thrust_opr_limit = engine.turboprop_max_thrust_opr_limit_problem

    # OPR should be the limit in this case
    prob_max_thrust_opr_limit.set_val("altitude", val=30000.0, units="ft")
    prob_max_thrust_opr_limit.set_val("mach_0", val=0.50562001)

    prob_max_thrust_opr_limit.run_model()

    assert prob_max_thrust_opr_limit.get_val("required_thrust", units="kN") == pytest.approx(
        2.793539, rel=1e-2
    )
    assert prob_max_thrust_opr_limit.get_val("shaft_power", units="kW") < (
        prob_max_thrust_opr_limit.get_val("shaft_power_limit", units="kW")
    )
    assert prob_max_thrust_opr_limit.get_val("opr") == pytest.approx(
        prob_max_thrust_opr_limit.get_val("opr_limit"), rel=1e-2
    )
    assert prob_max_thrust_opr_limit.get_val(
        "total_temperature_45", units="degK"
    ) < prob_max_thrust_opr_limit.get_val("itt_limit", units="degK")
    assert prob_max_thrust_opr_limit.get_val(
        "propeller_thrust", units="N"
    ) < prob_max_thrust_opr_limit.get_val("propeller_max_thrust", units="N")


def test_max_thrust_private_func():
    """
    Check that the private function indeed does what it is supposed to do
    """

    engine = BasicTPEngine(
        power_design=745.7,
        t_41t_design=1350,
        opr_design=9.5,
        cruise_altitude_propeller=9200.0,
        design_altitude=0.0,
        design_mach=0.0,
        prop_layout=1.0,
        bleed_control=1.0,
        itt_limit=1100.0,
        power_limit=521.99,
        opr_limit=12.0,
        speed_SL=SPEED,
        thrust_SL=THRUST_SL,
        thrust_limit_SL=THRUST_SL_LIMIT,
        efficiency_SL=EFFICIENCY_SL,
        speed_CL=SPEED,
        thrust_CL=THRUST_CL,
        thrust_limit_CL=THRUST_CL_LIMIT,
        efficiency_CL=EFFICIENCY_CL,
        effective_J=1.0,  # Effective advance ratio factor
        effective_efficiency_ls=1.0,  # Effective efficiency in low speed conditions
        effective_efficiency_cruise=1.0,  # Effective efficiency in cruise conditions
        eta_225=0.85,
        eta_253=0.86,
        eta_445=0.86,
        eta_455=0.86,
        eta_q=43.260e6 * 0.95,
        eta_axe=0.98,
        pi_02=0.8,
        pi_cc=0.95,
        cooling_ratio=0.05,
        hp_shaft_power_out=50 * 745.7,
        gearbox_efficiency=0.98,
        inter_compressor_bleed=0.04,
        exhaust_mach_design=0.4,
        pr_1_ratio_design=0.25,
    )  # load a 1000 kW turboprop gasoline engine

    # Here, the power is supposed to be the limit, meaning we should have loaded any other
    # problem. And the first execution for a point where the shaft power is the limit should take
    # longer that the second even if the point is different. The reason being, for the second
    # case, the problem is already warm started
    t1 = time.time()
    assert engine._max_thrust(0, 0.34767454) == pytest.approx(4050.69, rel=1e-2)
    t2 = time.time()

    assert engine._turboprop_max_thrust_opr_limit_problem is None
    assert not engine._turboprop_max_thrust_opr_limit_problem_setup

    assert engine._turboprop_max_thrust_itt_limit_problem is None
    assert not engine._turboprop_max_thrust_itt_limit_problem_setup

    assert engine._turboprop_max_thrust_propeller_thrust_limit_problem is None
    assert not engine._turboprop_max_thrust_propeller_thrust_limit_problem_setup

    t3 = time.time()
    assert engine._max_thrust(5000, 0.37073066) == pytest.approx(3885.731, rel=1e-2)
    t4 = time.time()

    assert t2 - t1 > t4 - t3

    # Here, if we check the cache, there should be 2 items corresponding to the test previously ran
    assert len(engine._cache_max_thrust) == 2
    assert "alt0ft0.348" in engine._cache_max_thrust
    assert "alt5000ft0.371" in engine._cache_max_thrust

    t5 = time.time()
    assert engine._max_thrust(0, 0.34767454) == pytest.approx(4050.775, rel=1e-2)
    t6 = time.time()

    assert t4 - t3 > t6 - t5

    # Now, we will rerun the function for the first case, since it is cached, it should be even
    # quicker than the second case

    # Here, the opr is supposed to be the limit, meaning ITT and propeller thrust should not be
    # loaded
    assert engine._max_thrust(30000.0, 0.50562001) == pytest.approx(2793.539, rel=1e-2)

    assert engine._turboprop_max_thrust_itt_limit_problem is None
    assert not engine._turboprop_max_thrust_itt_limit_problem_setup

    assert engine._turboprop_max_thrust_propeller_thrust_limit_problem is None
    assert not engine._turboprop_max_thrust_propeller_thrust_limit_problem_setup


def test_fuel_problem_not_called_in_max_thrust():
    """
    Really just a test for the sake of a test, but it is to make sure that the problem that will
    compute the fuel are not called when we just run the max thrust function.
    """

    engine = BasicTPEngine(
        power_design=745.7,
        t_41t_design=1350,
        opr_design=9.5,
        cruise_altitude_propeller=9200.0,
        design_altitude=0.0,
        design_mach=0.0,
        prop_layout=1.0,
        bleed_control=1.0,
        itt_limit=1100.0,
        power_limit=521.99,
        opr_limit=12.0,
        speed_SL=SPEED,
        thrust_SL=THRUST_SL,
        thrust_limit_SL=THRUST_SL_LIMIT,
        efficiency_SL=EFFICIENCY_SL,
        speed_CL=SPEED,
        thrust_CL=THRUST_CL,
        thrust_limit_CL=THRUST_CL_LIMIT,
        efficiency_CL=EFFICIENCY_CL,
        effective_J=1.0,  # Effective advance ratio factor
        effective_efficiency_ls=1.0,  # Effective efficiency in low speed conditions
        effective_efficiency_cruise=1.0,  # Effective efficiency in cruise conditions
        eta_225=0.85,
        eta_253=0.86,
        eta_445=0.86,
        eta_455=0.86,
        eta_q=43.260e6 * 0.95,
        eta_axe=0.98,
        pi_02=0.8,
        pi_cc=0.95,
        cooling_ratio=0.05,
        hp_shaft_power_out=50 * 745.7,
        gearbox_efficiency=0.98,
        inter_compressor_bleed=0.04,
        exhaust_mach_design=0.4,
        pr_1_ratio_design=0.25,
    )  # load a 1000 kW turboprop gasoline engine

    # Check that the fuel consumption are not called at object instantiation

    assert engine._turboprop_fuel_problem is None
    assert not engine._turboprop_fuel_problem_setup

    assert engine._turboprop_fuel_problem_ls is None
    assert not engine._turboprop_fuel_problem_ls_setup

    # They should also not be needed to compute max_thrust
    engine._max_thrust(30000.0, 0.50562001)

    assert engine._turboprop_fuel_problem is None
    assert not engine._turboprop_fuel_problem_setup

    assert engine._turboprop_fuel_problem_ls is None
    assert not engine._turboprop_fuel_problem_ls_setup


def test_fuel_consumed_private_func():
    """
    Check that the private function indeed does what it is supposed to do
    """

    engine = BasicTPEngine(
        power_design=745.7,
        t_41t_design=1350,
        opr_design=9.5,
        cruise_altitude_propeller=9200.0,
        design_altitude=0.0,
        design_mach=0.0,
        prop_layout=1.0,
        bleed_control=1.0,
        itt_limit=1100.0,
        power_limit=521.99,
        opr_limit=12.0,
        speed_SL=SPEED,
        thrust_SL=THRUST_SL,
        thrust_limit_SL=THRUST_SL_LIMIT,
        efficiency_SL=EFFICIENCY_SL,
        speed_CL=SPEED,
        thrust_CL=THRUST_CL,
        thrust_limit_CL=THRUST_CL_LIMIT,
        efficiency_CL=EFFICIENCY_CL,
        effective_J=1.0,  # Effective advance ratio factor
        effective_efficiency_ls=1.0,  # Effective efficiency in low speed conditions
        effective_efficiency_cruise=1.0,  # Effective efficiency in cruise conditions
        eta_225=0.85,
        eta_253=0.86,
        eta_445=0.86,
        eta_455=0.86,
        eta_q=43.260e6 * 0.95,
        eta_axe=0.98,
        pi_02=0.8,
        pi_cc=0.95,
        cooling_ratio=0.05,
        hp_shaft_power_out=50 * 745.7,
        gearbox_efficiency=0.98,
        inter_compressor_bleed=0.04,
        exhaust_mach_design=0.4,
        pr_1_ratio_design=0.25,
    )  # load a 1000 kW turboprop gasoline engine

    # Here, the point should be solved without requiring the use of the linesearch problem,
    # we check that and we also check that the geometry has been initialized properly. Since we
    # don't require the max thrust for this function, the max_thrust problem should not have been
    # instantiated. Also the first run should be longer than any subsequent run that don't require
    # the linesearch problem because subsequent run will be "warm started"

    t1 = time.time()
    assert engine._fuel_consumed(0, 0.34767454, 4050.690258)[0] * 3600.0 == pytest.approx(
        233.831317, rel=1e-2
    )
    t2 = time.time()

    # Area should now be loaded
    assert engine._a_41 == pytest.approx(0.004571, rel=1e-2)
    assert engine._a_45 == pytest.approx(0.012201, rel=1e-2)
    assert engine._a_8 == pytest.approx(0.038730, rel=1e-2)

    # No need for linesearch problem
    assert engine._turboprop_fuel_problem_ls is None
    assert not engine._turboprop_fuel_problem_ls_setup

    assert engine._turboprop_max_thrust_opr_limit_problem is None
    assert not engine._turboprop_max_thrust_opr_limit_problem_setup

    assert engine._turboprop_max_thrust_itt_limit_problem is None
    assert not engine._turboprop_max_thrust_itt_limit_problem_setup

    assert engine._turboprop_max_thrust_propeller_thrust_limit_problem is None
    assert not engine._turboprop_max_thrust_propeller_thrust_limit_problem_setup

    t3 = time.time()
    assert engine._fuel_consumed(5000, 0.37073066, 3885.650828)[0] * 3600.0 == pytest.approx(
        217.645651, rel=1e-2
    )
    t4 = time.time()

    assert t4 - t3 < t2 - t1

    # No need for linesearch problem
    assert engine._turboprop_fuel_problem_ls is None
    assert not engine._turboprop_fuel_problem_ls_setup

    assert engine._fuel_consumed(10000, 0.396, 3724.108916)[0] * 3600.0 == pytest.approx(
        203.416703, rel=1e-2
    )


def test_max_thrust_cache_saturation():
    """
    Check that the cache implemented for max thrust works as intended in that it never goes above
    128 elements.
    """

    engine = BasicTPEngine(
        power_design=745.7,
        t_41t_design=1350,
        opr_design=9.5,
        cruise_altitude_propeller=9200.0,
        design_altitude=0.0,
        design_mach=0.0,
        prop_layout=1.0,
        bleed_control=1.0,
        itt_limit=1100.0,
        power_limit=521.99,
        opr_limit=12.0,
        speed_SL=SPEED,
        thrust_SL=THRUST_SL,
        thrust_limit_SL=THRUST_SL_LIMIT,
        efficiency_SL=EFFICIENCY_SL,
        speed_CL=SPEED,
        thrust_CL=THRUST_CL,
        thrust_limit_CL=THRUST_CL_LIMIT,
        efficiency_CL=EFFICIENCY_CL,
        effective_J=1.0,  # Effective advance ratio factor
        effective_efficiency_ls=1.0,  # Effective efficiency in low speed conditions
        effective_efficiency_cruise=1.0,  # Effective efficiency in cruise conditions
        eta_225=0.85,
        eta_253=0.86,
        eta_445=0.86,
        eta_455=0.86,
        eta_q=43.260e6 * 0.95,
        eta_axe=0.98,
        pi_02=0.8,
        pi_cc=0.95,
        cooling_ratio=0.05,
        hp_shaft_power_out=50 * 745.7,
        gearbox_efficiency=0.98,
        inter_compressor_bleed=0.04,
        exhaust_mach_design=0.4,
        pr_1_ratio_design=0.25,
    )  # load a 1000 kW turboprop gasoline engine

    altitude = np.linspace(0.0, 10000.0, CACHE_MAX_SIZE)
    mach = np.linspace(0.3, 0.4, CACHE_MAX_SIZE)
    dummy_thrust = np.linspace(1000.0, 2000.0, CACHE_MAX_SIZE)

    # Artificially fill the cache
    for altitude_loc, mach_loc, dummy_thrust_loc in zip(altitude, mach, dummy_thrust):
        key = "alt" + str(round(altitude_loc)) + "ft" + str(round(mach_loc, 3))
        engine._cache_max_thrust[key] = dummy_thrust_loc

    assert len(engine._cache_max_thrust) == CACHE_MAX_SIZE

    # Now properly add one value and check that it has indeed not changed the size
    engine._max_thrust(15000.0, 0.41)
    assert len(engine._cache_max_thrust) == CACHE_MAX_SIZE
    assert "alt0ft0.3" not in engine._cache_max_thrust.keys()
    assert list(engine._cache_max_thrust.keys())[-2] == "alt10000ft0.4"
    assert list(engine._cache_max_thrust.keys())[-1] == "alt15000ft0.41"
