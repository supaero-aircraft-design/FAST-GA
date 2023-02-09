"""
Test load_analysis module.
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

import numpy as np
import pytest


from ..wing.aerostructural_loads import AerostructuralLoad
from ..wing.structural_loads import StructuralLoads
from ..wing.aerodynamic_loads import AerodynamicLoads
from ..wing.loads import WingLoads

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs


XML_FILE = "daher_tbm900.xml"


def test_compute_shear_stress():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(AerostructuralLoad()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(AerostructuralLoad(), ivc)
    shear_max_mass_condition = problem.get_val("data:loads:max_shear:mass", units="kg")
    assert shear_max_mass_condition == pytest.approx(3360.1, abs=1e-1)
    shear_max_lf_condition = problem.get_val("data:loads:max_shear:load_factor")
    assert shear_max_lf_condition == pytest.approx(3.8, abs=1e-2)
    lift_shear_diagram = problem.get_val("data:loads:max_shear:lift_shear", units="N")
    lift_root_shear = lift_shear_diagram[0]
    assert lift_root_shear == pytest.approx(98313, abs=1)
    weight_shear_diagram = problem.get_val("data:loads:max_shear:weight_shear", units="N")
    weight_root_shear = weight_shear_diagram[0]
    assert weight_root_shear == pytest.approx(-30043, abs=1)


def test_compute_root_bending_moment():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(AerostructuralLoad()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(AerostructuralLoad(), ivc)
    max_rbm_mass_condition = problem.get_val("data:loads:max_rbm:mass", units="kg")
    assert max_rbm_mass_condition == pytest.approx(2764.09, abs=1e-1)
    max_rbm_lf_condition = problem.get_val("data:loads:max_rbm:load_factor")
    assert max_rbm_lf_condition == pytest.approx(3.8, abs=1e-2)
    lift_rbm_diagram = problem.get_val("data:loads:max_rbm:lift_rbm", units="N*m")
    lift_rbm = lift_rbm_diagram[0]
    assert lift_rbm == pytest.approx(215119, abs=1)
    weight_rbm_diagram = problem.get_val("data:loads:max_rbm:weight_rbm", units="N*m")
    weight_rbm = weight_rbm_diagram[0]
    assert weight_rbm == pytest.approx(-34297, abs=1)


def test_compute_mass_distribution():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(StructuralLoads()), __file__, XML_FILE)
    load_factor_shear = 4.0
    ivc.add_output("data:loads:max_shear:load_factor", load_factor_shear)
    load_factor_rbm = 4.0
    ivc.add_output("data:loads:max_rbm:load_factor", load_factor_rbm)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(StructuralLoads(), ivc)
    point_mass_array = problem.get_val(
        "data:loads:structure:ultimate:force_distribution:point_mass", units="N/m"
    )
    point_mass_result = np.array(
        [
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -44813.82752648,
            -44813.82752648,
            -44813.82752648,
            -44813.82752648,
            -44813.82752648,
            -44813.82752648,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    assert np.max(np.abs(point_mass_array - point_mass_result)) <= 1e-1
    wing_mass_array = problem.get_val(
        "data:loads:structure:ultimate:force_distribution:wing", units="N/m"
    )
    wing_mass_result = np.array(
        [
            -1412.66003141,
            -1399.80371957,
            -1399.80371957,
            -1399.80371957,
            -1399.80371957,
            -1399.80371957,
            -1399.80371957,
            -1399.80371957,
            -1391.23831118,
            -1374.05455614,
            -1356.78609991,
            -1339.42235484,
            -1321.98449622,
            -1308.16572847,
            -1308.06074968,
            -1306.77530792,
            -1305.48986617,
            -1304.4831117,
            -1304.20431627,
            -1302.91837568,
            -1302.81335615,
            -1286.91820129,
            -1269.32152792,
            -1251.68250396,
            -1234.03289235,
            -1216.36210545,
            -1198.70190619,
            -1181.04170693,
            -1163.41327062,
            -1145.80600961,
            -1128.25168684,
            -1110.76088997,
            -1093.32303135,
            -1075.95928628,
            -1058.69083005,
            -1041.51766266,
            -1024.45037177,
            -1007.49954502,
            -990.66518242,
            -973.97904691,
            -957.43055084,
            -941.03028186,
            -924.79941528,
            -908.72736344,
            -892.83530163,
            -877.11264222,
            -861.59114815,
            -861.72261916,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    assert np.max(np.abs(wing_mass_result - wing_mass_array)) <= 1e-1
    fuel_mass_array = problem.get_val(
        "data:loads:structure:ultimate:force_distribution:fuel", units="N/m"
    )
    fuel_mass_result = np.array(
        [
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -0.0,
            -1221.0126141,
            -1191.94215652,
            -1159.89107215,
            -1124.85720896,
            -1090.30337158,
            -1056.25423514,
            -1022.73279742,
            -996.64444201,
            -996.44786408,
            -994.04277584,
            -4958.20659062,
            -4948.81518305,
            -4946.21744246,
            -4934.24642256,
            -4933.26957984,
            -4786.80944465,
            -4627.77511951,
            -4471.7672853,
            -4318.89601188,
            -4169.212965,
            -4022.7981801,
            -3879.68817554,
            -3739.95774828,
            -3603.63912663,
            -3470.7578101,
            -3341.34313732,
            -3215.40706498,
            -3092.96537302,
            -2974.01744234,
            -2858.56650296,
            -2746.58202266,
            -2638.05708337,
            -2532.97031311,
            -2431.28726021,
            -2332.96170965,
            -2237.96933044,
            -2146.24953351,
            -2057.75592998,
            -1972.43201003,
            -1890.22692722,
            -1811.0662492,
            -0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    assert np.max(np.abs(fuel_mass_result - fuel_mass_array)) <= 1e-1


def test_compute_structure_shear():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(StructuralLoads()), __file__, XML_FILE)
    ivc.add_output("data:loads:max_shear:load_factor", 4.0)
    ivc.add_output("data:loads:max_rbm:load_factor", 4.0)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(StructuralLoads(), ivc)
    point_mass_array = problem.get_val("data:loads:structure:ultimate:shear:point_mass", units="N")
    point_mass_result = np.array(
        [
            -2239.75490502,
            -2239.75490502,
            -2239.75490502,
            -2239.75490502,
            -2239.75490502,
            -2239.75490502,
            -2239.75490502,
            -2239.75490502,
            -2239.75490502,
            -2239.75490502,
            -2239.75490502,
            -2239.75490502,
            -2239.75490502,
            -2239.75490502,
            -2217.34799126,
            -1668.61272188,
            -1119.87745251,
            -690.10948925,
            -571.14218314,
            -22.40691376,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    assert np.max(np.abs(point_mass_array - point_mass_result)) <= 1e-1
    wing_mass_array = problem.get_val("data:loads:structure:ultimate:shear:wing", units="N")
    wing_mass_result = np.array(
        [
            -7149.39659328,
            -7057.15220202,
            -6873.50675313,
            -6689.84239901,
            -6506.19695013,
            -6322.55150125,
            -6138.88714713,
            -5955.24169824,
            -5750.10666229,
            -5523.87703291,
            -5299.21197281,
            -5076.28546167,
            -4855.26591117,
            -4682.15744399,
            -4680.84933075,
            -4664.84029015,
            -4648.8469895,
            -4636.3320713,
            -4632.86942893,
            -4616.90761255,
            -4615.60474669,
            -4419.62127402,
            -4205.33307564,
            -3993.57306985,
            -3784.5254651,
            -3578.2993672,
            -3375.04673743,
            -3174.86310454,
            -2977.90125253,
            -2784.25827028,
            -2594.02450321,
            -2407.29694299,
            -2224.15210746,
            -2044.67544019,
            -1868.92897957,
            -1696.98250728,
            -1528.85777097,
            -1364.61279959,
            -1204.28613101,
            -1047.89613224,
            -895.44288616,
            -746.9621424,
            -602.4320724,
            -461.85286861,
            -325.20860643,
            -192.49211904,
            -63.65651079,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    assert np.max(np.abs(wing_mass_result - wing_mass_array)) <= 1e-1
    fuel_mass_array = problem.get_val("data:loads:structure:ultimate:shear:fuel", units="N")
    fuel_mass_result = np.array(
        [
            -15029.18144424,
            -15029.18144424,
            -15029.18144424,
            -15029.18144424,
            -15029.18144424,
            -15029.18144424,
            -14949.07860911,
            -14790.79636004,
            -14617.94213978,
            -14431.02604024,
            -14248.78575322,
            -14071.30518885,
            -13898.65335275,
            -13765.74410091,
            -13764.74755476,
            -13752.56099898,
            -13716.11901957,
            -13668.61447907,
            -13655.4803033,
            -13594.98827104,
            -13590.05451304,
            -12854.46675054,
            -12065.24720212,
            -11300.90127962,
            -10561.61024529,
            -9847.2480325,
            -9157.80551653,
            -8493.04969209,
            -7852.90962234,
            -7237.10149556,
            -6645.30068897,
            -6077.18999451,
            -5532.36730162,
            -5010.44092446,
            -4510.94419275,
            -4033.42387169,
            -3577.28703794,
            -3142.04014107,
            -2727.13260907,
            -2331.96679524,
            -1955.90714354,
            -1598.40951768,
            -1258.80110631,
            -936.47380375,
            -630.79359168,
            -341.1587875,
            -66.89794999,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )
    assert np.max(np.abs(fuel_mass_result - fuel_mass_array)) <= 1e-1


def test_compute_structure_bending():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(StructuralLoads()), __file__, XML_FILE)
    ivc.add_output("data:loads:max_shear:load_factor", 4.0)
    ivc.add_output("data:loads:max_rbm:load_factor", 4.0)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(StructuralLoads(), ivc)
    point_mass_array = problem.get_val(
        "data:loads:structure:ultimate:root_bending:point_mass", units="N*m"
    )
    point_mass_result = np.array(
        [
            -4068.55958007,
            -3921.63869782,
            -3627.79693331,
            -3333.9249195,
            -3040.08315499,
            -2746.24139048,
            -2452.36937667,
            -2158.52761216,
            -1829.29415891,
            -1462.82380921,
            -1094.29650665,
            -723.9239964,
            -351.91802358,
            -57.09034808,
            -54.85059318,
            -31.05923439,
            -13.98701593,
            -5.30806074,
            -3.6339378,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )
    assert np.max(np.abs(point_mass_array - point_mass_result)) <= 1e-1
    wing_mass_array = problem.get_val(
        "data:loads:structure:ultimate:root_bending:wing", units="N*m"
    )
    wing_mass_result = np.array(
        [
            -19998.83458005,
            -19532.89593505,
            -18619.08849145,
            -17729.28382566,
            -16863.66511931,
            -16022.13954146,
            -15204.62418244,
            -14411.28734185,
            -13551.01799076,
            -12628.80423634,
            -11738.51034022,
            -10880.76753713,
            -10056.11085205,
            -9428.44659215,
            -9423.76508879,
            -9366.54720007,
            -9309.52524209,
            -9265.00248384,
            -9252.69900577,
            -9196.06833108,
            -9191.45207493,
            -8507.78305977,
            -7784.88144501,
            -7096.31187623,
            -6442.30193765,
            -5822.76919261,
            -5237.69824515,
            -4686.84589592,
            -4170.06558895,
            -3686.99503207,
            -3237.20361383,
            -2820.22865596,
            -2435.50618926,
            -2082.44144553,
            -1760.35013046,
            -1468.51998679,
            -1206.11997488,
            -972.34799757,
            -766.33572918,
            -587.16280279,
            -433.87102957,
            -305.51701656,
            -201.08823211,
            -119.58241496,
            -59.97709064,
            -21.24659705,
            -2.35155117,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    assert np.max(np.abs(wing_mass_result - wing_mass_array)) <= 1e-1
    fuel_mass_array = problem.get_val(
        "data:loads:structure:ultimate:root_bending:fuel", units="N*m"
    )
    fuel_mass_result = np.array(
        [
            -50993.88983427,
            -50008.02276801,
            -48036.28863551,
            -46064.35152441,
            -44092.6173919,
            -42120.8832594,
            -40148.94614829,
            -38198.22891095,
            -36036.92881916,
            -33660.65649788,
            -31301.41486144,
            -28960.10053628,
            -26637.53479177,
            -24816.86152489,
            -24803.09627911,
            -24634.62473277,
            -24466.30222777,
            -24334.9916208,
            -24298.72297653,
            -24131.88548661,
            -24118.29296546,
            -22117.88605975,
            -20029.99746822,
            -18068.38190172,
            -16230.8352247,
            -14514.27879156,
            -12915.84390517,
            -11432.06959805,
            -10059.80256176,
            -8795.36604479,
            -7634.97903188,
            -6574.85799873,
            -5611.04989247,
            -4739.62346691,
            -3956.53089701,
            -3257.76756166,
            -2639.16330111,
            -2096.74101943,
            -1626.49297026,
            -1224.42185391,
            -886.57685099,
            -609.17083842,
            -388.39720568,
            -220.60244984,
            -102.21978591,
            -29.80181402,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    assert np.max(np.abs(fuel_mass_result - fuel_mass_array)) <= 1e-1


def test_compute_lift_distribution():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(AerodynamicLoads()), __file__, XML_FILE)
    ivc.add_output("data:loads:max_shear:load_factor", 4.0)
    ivc.add_output("data:loads:max_shear:mass", 1747.0, units="kg")
    ivc.add_output("data:loads:max_rbm:load_factor", 4.0)
    ivc.add_output("data:loads:max_rbm:mass", 1568.0, units="kg")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(AerodynamicLoads(), ivc)
    lift_array = problem.get_val("data:loads:aerodynamic:ultimate:force_distribution", units="N/m")
    lift_result = np.array(
        [
            5919.39894384,
            5865.94599763,
            5851.16605027,
            5845.78286076,
            5839.14903125,
            5831.31893741,
            5814.95186625,
            5790.482823,
            5738.67581279,
            5684.71508221,
            5662.97277972,
            5625.724133,
            5578.72442532,
            5519.92826961,
            5519.4816443,
            5514.01286522,
            5508.54417415,
            5504.26117172,
            5503.16199136,
            5498.09160799,
            5497.67749154,
            5435.0313161,
            5363.88178001,
            5297.46288024,
            5220.55982652,
            5148.54428618,
            5066.79865581,
            4989.17573646,
            4894.45994953,
            4811.44017913,
            4718.97193641,
            4626.39327636,
            4508.57497962,
            4406.0007625,
            4289.69885708,
            4184.45021623,
            4068.52384535,
            3957.21552704,
            3826.93912295,
            3695.36460433,
            3532.75591626,
            3363.41894544,
            3137.43479404,
            2891.18611534,
            2572.19613879,
            2153.36878189,
            1583.50083153,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    assert np.max(np.abs(lift_array - lift_result)) <= 1e-1


def test_load_group():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(WingLoads()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(WingLoads(), ivc)

    lift_shear_diagram = problem.get_val("data:loads:max_shear:lift_shear", units="N")
    lift_root_shear = lift_shear_diagram[0]
    assert lift_root_shear == pytest.approx(98313, abs=1)
