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
    assert shear_max_mass_condition == pytest.approx(3358.7, abs=1e-1)
    shear_max_lf_condition = problem.get_val("data:loads:max_shear:load_factor")
    assert shear_max_lf_condition == pytest.approx(3.8, abs=1e-2)
    lift_shear_diagram = problem.get_val("data:loads:max_shear:lift_shear", units="N")
    lift_root_shear = lift_shear_diagram[0]
    assert lift_root_shear == pytest.approx(98467.47, abs=1)
    weight_shear_diagram = problem.get_val("data:loads:max_shear:weight_shear", units="N")
    weight_root_shear = weight_shear_diagram[0]
    assert weight_root_shear == pytest.approx(-30016, abs=1)


def test_compute_root_bending_moment():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(AerostructuralLoad()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(AerostructuralLoad(), ivc)
    max_rbm_mass_condition = problem.get_val("data:loads:max_rbm:mass", units="kg")
    assert max_rbm_mass_condition == pytest.approx(2765.07, abs=1e-1)
    max_rbm_lf_condition = problem.get_val("data:loads:max_rbm:load_factor")
    assert max_rbm_lf_condition == pytest.approx(3.8, abs=1e-2)
    lift_rbm_diagram = problem.get_val("data:loads:max_rbm:lift_rbm", units="N*m")
    lift_rbm = lift_rbm_diagram[0]
    assert lift_rbm == pytest.approx(219390, abs=1)
    weight_rbm_diagram = problem.get_val("data:loads:max_rbm:weight_rbm", units="N*m")
    weight_rbm = weight_rbm_diagram[0]
    assert weight_rbm == pytest.approx(-34326, abs=1)


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
            -44953.8885916,
            -44953.8885916,
            -44953.8885916,
            -44953.8885916,
            -44953.8885916,
            -44953.8885916,
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
            -1421.81474618,
            -1408.98015732,
            -1408.98015732,
            -1408.98015732,
            -1408.98015732,
            -1408.98015732,
            -1408.98015732,
            -1408.98015732,
            -1400.3692553,
            -1383.09416597,
            -1365.71250607,
            -1348.25624678,
            -1330.71473103,
            -1316.26821312,
            -1316.16230956,
            -1314.86922381,
            -1313.57613806,
            -1313.10927293,
            -1312.28322839,
            -1310.99041822,
            -1310.88453722,
            -1295.45052956,
            -1277.74915795,
            -1260.00515812,
            -1242.25050123,
            -1224.47453022,
            -1206.69855922,
            -1188.93324527,
            -1171.18924544,
            -1153.47721678,
            -1135.8184734,
            -1118.20235825,
            -1100.65018544,
            -1083.18326909,
            -1065.79095214,
            -1048.50520575,
            -1031.31537288,
            -1014.25342469,
            -997.30870413,
            -980.49186825,
            -963.82423117,
            -947.30579289,
            -930.94721046,
            -914.75914094,
            -898.74158434,
            -882.9051977,
            -867.24998103,
            -867.30699517,
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
            -1221.91029761,
            -1192.80769456,
            -1160.72136278,
            -1125.64921322,
            -1091.05787079,
            -1056.97203701,
            -1023.41473355,
            -996.28226041,
            -996.08503997,
            -993.67893953,
            -4956.38236939,
            -4952.04981366,
            -4944.38820288,
            -4932.41218591,
            -4931.43215531,
            -4789.87338373,
            -4630.66015831,
            -4474.5033243,
            -4321.46282438,
            -4171.63950839,
            -4025.06537541,
            -3881.82399561,
            -3741.95532212,
            -3605.50321082,
            -3472.49314999,
            -3342.94410187,
            -3216.89909099,
            -3094.34277031,
            -2975.28483953,
            -2859.71906733,
            -2747.64308846,
            -2639.02167959,
            -2533.84280973,
            -2432.06341446,
            -2333.66275622,
            -2238.58274225,
            -2146.7876096,
            -2058.22255524,
            -1972.83095937,
            -1890.56186977,
            -1811.34072989,
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
            -2240.50731345,
            -2240.50731345,
            -2240.50731345,
            -2240.50731345,
            -2240.50731345,
            -2240.50731345,
            -2240.50731345,
            -2240.50731345,
            -2240.50731345,
            -2240.50731345,
            -2240.50731345,
            -2240.50731345,
            -2240.50731345,
            -2240.50731345,
            -2218.03036916,
            -1669.14201294,
            -1120.25365673,
            -922.0789954,
            -571.36530051,
            -22.4769443,
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
            -7176.41233792,
            -7083.83016764,
            -6899.50534681,
            -6715.16155081,
            -6530.83672999,
            -6346.51190916,
            -6162.16811316,
            -5977.84329233,
            -5771.94860551,
            -5544.87859213,
            -5319.3771679,
            -5095.61894062,
            -4873.77204998,
            -4693.23191825,
            -4691.91570299,
            -4675.8532152,
            -4659.80651602,
            -4654.01677782,
            -4643.77560477,
            -4627.760479,
            -4626.44954152,
            -4436.49006873,
            -4221.37488662,
            -4008.82984585,
            -3798.97149259,
            -3591.97673979,
            -3387.93257996,
            -3187.00038966,
            -2989.28490566,
            -2794.89972423,
            -2603.9346544,
            -2416.47366889,
            -2232.63976381,
            -2052.47189078,
            -1876.04815728,
            -1703.42525403,
            -1534.66748844,
            -1369.7908632,
            -1208.84721283,
            -1051.84335835,
            -898.81993136,
            -749.75958009,
            -604.68002323,
            -463.56854375,
            -326.40911205,
            -193.19448614,
            -63.8783305,
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
            -14973.49022311,
            -14973.49022311,
            -14973.49022311,
            -14973.49022311,
            -14973.49022311,
            -14973.49022311,
            -14893.5559595,
            -14735.60751042,
            -14563.11948668,
            -14376.60142673,
            -14194.75138627,
            -14017.65304502,
            -13845.37520393,
            -13707.61974082,
            -13706.62355717,
            -13694.47601761,
            -13658.15080222,
            -13636.31064274,
            -13597.70645908,
            -13537.40844149,
            -13532.47651932,
            -12823.9513191,
            -12036.41050044,
            -11273.82393667,
            -10536.12674555,
            -9823.42970984,
            -9135.49442103,
            -8472.30878593,
            -7833.63887636,
            -7219.25522897,
            -6628.83343458,
            -6062.01143988,
            -5518.5226018,
            -4997.8404866,
            -4499.54397225,
            -4023.14178462,
            -3568.15738934,
            -3133.98225464,
            -2720.10505848,
            -2325.89624435,
            -1950.81950096,
            -1594.20299056,
            -1255.46929166,
            -933.97979516,
            -629.10150008,
            -340.2336753,
            -66.70603698,
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
            -4069.92634503,
            -3923.37314236,
            -3630.26673701,
            -3337.13015806,
            -3044.02375271,
            -2750.91734736,
            -2457.78076841,
            -2164.67436306,
            -1836.2648391,
            -1470.71160657,
            -1103.10656885,
            -733.66094119,
            -362.58593883,
            -56.95383312,
            -54.71332581,
            -30.9820789,
            -13.95277563,
            -9.45106739,
            -3.625416,
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
            -20017.93646986,
            -19551.5638139,
            -18636.90373161,
            -17746.26685352,
            -16879.83653258,
            -16037.51985114,
            -15219.23382124,
            -14425.14690116,
            -13564.06149289,
            -12640.96975032,
            -11749.82537698,
            -10891.26021186,
            -10065.81038367,
            -9413.35046995,
            -9408.65789616,
            -9351.46757169,
            -9294.47327429,
            -9273.94375359,
            -9237.67478896,
            -9181.0719674,
            -9176.44486242,
            -8515.98582588,
            -7792.32653233,
            -7103.14502939,
            -6448.44780149,
            -5828.3629864,
            -5242.67646155,
            -4691.33285354,
            -4174.04896588,
            -3690.50782047,
            -3240.27874299,
            -2822.86643151,
            -2437.80373491,
            -2084.39774465,
            -1761.99635884,
            -1469.86345844,
            -1207.23603781,
            -973.24217295,
            -767.03565213,
            -587.68109905,
            -434.26137135,
            -305.77937747,
            -201.25768654,
            -119.6805222,
            -60.02423229,
            -21.26174529,
            -2.35251726,
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
            -50690.25406358,
            -49710.82724617,
            -47751.97361134,
            -45792.91832389,
            -43834.06468907,
            -41875.21105425,
            -39916.1557668,
            -37978.2152874,
            -35831.11033322,
            -33470.49259013,
            -31126.84256096,
            -28801.04983575,
            -26493.92902605,
            -24614.77553902,
            -24601.06841742,
            -24433.78437468,
            -24266.64847441,
            -24206.4860621,
            -24100.25128929,
            -23934.59144374,
            -23921.05650151,
            -22000.85317643,
            -19923.68127568,
            -17972.46996318,
            -16144.39665884,
            -14436.983436,
            -12846.82612899,
            -11370.98981671,
            -10005.9667468,
            -8748.22002095,
            -7593.98624176,
            -6539.41811112,
            -5580.82578141,
            -4714.05222288,
            -3935.15055167,
            -3240.08142353,
            -2624.85719097,
            -2085.35802644,
            -1617.64997936,
            -1217.72435321,
            -881.74368543,
            -605.8247885,
            -386.25877871,
            -219.38450138,
            -101.65335712,
            -29.63561445,
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
            5740.65703753,
            5689.0428984,
            5690.68971056,
            5687.07679594,
            5683.88502038,
            5680.07454519,
            5674.10351466,
            5665.69984208,
            5619.64475768,
            5556.83287018,
            5528.02548714,
            5487.20267503,
            5440.45737811,
            5385.9718833,
            5385.57209372,
            5380.69021526,
            5375.80753182,
            5374.04445435,
            5371.07808116,
            5366.43447897,
            5366.05411192,
            5310.51729086,
            5246.97783152,
            5188.35266825,
            5126.36465834,
            5063.4533161,
            4992.597627,
            4925.62677104,
            4850.2981431,
            4780.79628982,
            4703.86264599,
            4625.29120505,
            4531.13394955,
            4445.99131798,
            4350.68198293,
            4263.8271147,
            4177.46962982,
            4086.55256846,
            3980.86843051,
            3869.11259881,
            3738.51403824,
            3596.81298466,
            3400.85659681,
            3176.57753255,
            2883.06133394,
            2476.07961807,
            1843.17629063,
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
    assert lift_root_shear == pytest.approx(98467, abs=1)
