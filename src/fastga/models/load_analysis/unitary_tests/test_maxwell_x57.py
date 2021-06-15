"""
Test load_analysis module
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

import pandas as pd
import numpy as np
from openmdao.core.component import Component
import pytest
from typing import Union

from fastoad.module_management.service_registry import RegisterPropulsion
from fastoad.model_base import FlightPoint, Atmosphere
from fastoad.model_base.propulsion import IOMPropulsionWrapper
from fastoad.model_base.atmosphere import Atmosphere

from ..aerostructural_loads_x57 import AerostructuralLoadX57
from ..structural_loads_x57 import StructuralLoadsX57
from ..aerodynamic_loads_x57 import AerodynamicLoadsX57

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

from fastga.models.propulsion.fuel_propulsion.base import AbstractFuelPropulsion
from fastga.models.propulsion.propulsion import IPropulsion

from .dummy_engines import ENGINE_WRAPPER_X57 as ENGINE_WRAPPER

XML_FILE = "maxwell_x57.xml"


def test_compute_shear_stress():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(AerostructuralLoadX57()), __file__, XML_FILE)
    y_vector = np.array(
        [0.04619, 0.13856, 0.23093, 0.3233, 0.41567, 0.50804, 0.60041, 0.71159, 0.84215, 0.97345, 1.10541, 1.23795,
         1.37098, 1.50444, 1.63824, 1.77229, 1.90652, 2.04083, 2.17515, 2.3094, 2.44348, 2.57731, 2.71082, 2.84392,
         2.97653, 3.10858, 3.23997, 3.37064, 3.50051, 3.62951, 3.75757, 3.88462, 4.01059, 4.13542, 4.25904, 4.38141,
         4.50245, 4.62213, 4.74039, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    chord_vector = np.array(
        [0.73944, 0.73944, 0.73944, 0.73944, 0.73944, 0.73944, 0.73944, 0.73593, 0.72888, 0.7218, 0.71468, 0.70753,
         0.70035, 0.69315, 0.68593, 0.6787, 0.67146, 0.66421, 0.65696, 0.64972, 0.64249, 0.63527, 0.62806, 0.62088,
         0.61373, 0.6066, 0.59951, 0.59246, 0.58546, 0.5785, 0.57159, 0.56473, 0.55794, 0.5512, 0.54453, 0.53793,
         0.5314, 0.52494, 0.51856, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    cl_vector = np.array(
        [0.13405957102783722, 0.13404954188359933, 0.13398936701817202, 0.13389910472003103, 0.13398936701817202,
         0.1333073852099957, 0.1330767148925243, 0.13374866755646272, 0.13426015391259497, 0.13496219400924706,
         0.13463123224939677, 0.13483181513415454, 0.13492207743229553, 0.13512266031705325, 0.13551379694233084,
         0.13576452554827803, 0.13570435068285072, 0.1357143798270886, 0.1355438843750445, 0.13566423410589917,
         0.13594505014456001, 0.13589490442337057, 0.13572440897132648, 0.1356341466731855, 0.13536335977876254,
         0.13509257288433957, 0.13433035792226017, 0.13376872584493849, 0.13335753093118513, 0.13288616115200444,
         0.13210388790144922, 0.1310909443334226, 0.12974703900554577, 0.12738016096540442, 0.12402039764571221,
         0.11892559237286555, 0.11150402563682907, 0.09912806164727636, 0.07784621757447988, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0])
    y_vector_prop_on = np.array(
        [0.04619, 0.13856, 0.23093, 0.3233, 0.41567, 0.50804, 0.60041, 0.71159, 0.84215, 0.97345, 1.10541, 1.23795,
         1.37098, 1.50444, 1.63824, 1.77229, 1.90652, 2.04083, 2.17515, 2.3094, 2.44348, 2.57731, 2.71082, 2.84392,
         2.97653, 3.10858, 3.23997, 3.37064, 3.50051, 3.62951, 3.75757, 3.88462, 4.01059, 4.13542, 4.25904, 4.38141,
         4.50245, 4.62213, 4.74039, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    cl_vector_only_prop = np.array(
        [-0.0008, -0.0182, -0.0237, -0.0288, -0.0355, -0.0492, -0.0735, -0.0749, -0.0296, 0.1332, 0.122, 0.0184, 0.0133,
         0.1141, 0.1704, 0.0053, -0.0109, 0.0654, 0.1618, 0.0208, -0.0486, -0.0233, 0.1395, 0.05, -0.0616, -0.0225,
         0.173, 0.174, -0.0087, -0.025, -0.0363, 0.1426, 0.1145, -0.0966, -0.1327, -0.1526, -0.1463, -0.121, -0.0979, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    ivc.add_output("data:aerodynamics:wing:low_speed:CL_vector", cl_vector)
    ivc.add_output("data:aerodynamics:wing:low_speed:chord_vector", chord_vector, units="m")
    ivc.add_output("data:aerodynamics:wing:low_speed:Y_vector", y_vector, units="m")
    ivc.add_output("data:aerodynamics:slipstream:wing:prop_on:Y_vector", y_vector_prop_on, units="m")
    ivc.add_output("data:aerodynamics:slipstream:wing:only_prop:CL_vector", cl_vector_only_prop, units="m")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(AerostructuralLoadX57(), ivc)
    shear_max_mass_condition = problem.get_val("data:loads:max_shear:mass", units="kg")
    assert shear_max_mass_condition == pytest.approx(1361., abs=1e-1)
    shear_max_lf_condition = problem.get_val("data:loads:max_shear:load_factor")
    assert shear_max_lf_condition == pytest.approx(3.8, abs=1e-2)
    lift_shear_diagram = problem.get_val("data:loads:max_shear:lift_shear", units="N")
    lift_root_shear = lift_shear_diagram[0]
    assert lift_root_shear == pytest.approx(40261.893, abs=1)
    weight_shear_diagram = problem.get_val("data:loads:max_shear:weight_shear", units="N")
    weight_root_shear = weight_shear_diagram[0]
    assert weight_root_shear == pytest.approx(-9911.28, abs=1)


def test_compute_root_bending_moment():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(AerostructuralLoadX57()), __file__, XML_FILE)
    y_vector = np.array(
        [0.04619, 0.13856, 0.23093, 0.3233, 0.41567, 0.50804, 0.60041, 0.71159, 0.84215, 0.97345, 1.10541, 1.23795,
         1.37098, 1.50444, 1.63824, 1.77229, 1.90652, 2.04083, 2.17515, 2.3094, 2.44348, 2.57731, 2.71082, 2.84392,
         2.97653, 3.10858, 3.23997, 3.37064, 3.50051, 3.62951, 3.75757, 3.88462, 4.01059, 4.13542, 4.25904, 4.38141,
         4.50245, 4.62213, 4.74039, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    chord_vector = np.array(
        [0.73944, 0.73944, 0.73944, 0.73944, 0.73944, 0.73944, 0.73944, 0.73593, 0.72888, 0.7218, 0.71468, 0.70753,
         0.70035, 0.69315, 0.68593, 0.6787, 0.67146, 0.66421, 0.65696, 0.64972, 0.64249, 0.63527, 0.62806, 0.62088,
         0.61373, 0.6066, 0.59951, 0.59246, 0.58546, 0.5785, 0.57159, 0.56473, 0.55794, 0.5512, 0.54453, 0.53793,
         0.5314, 0.52494, 0.51856, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    cl_vector = np.array(
        [0.13405957102783722, 0.13404954188359933, 0.13398936701817202, 0.13389910472003103, 0.13398936701817202,
         0.1333073852099957, 0.1330767148925243, 0.13374866755646272, 0.13426015391259497, 0.13496219400924706,
         0.13463123224939677, 0.13483181513415454, 0.13492207743229553, 0.13512266031705325, 0.13551379694233084,
         0.13576452554827803, 0.13570435068285072, 0.1357143798270886, 0.1355438843750445, 0.13566423410589917,
         0.13594505014456001, 0.13589490442337057, 0.13572440897132648, 0.1356341466731855, 0.13536335977876254,
         0.13509257288433957, 0.13433035792226017, 0.13376872584493849, 0.13335753093118513, 0.13288616115200444,
         0.13210388790144922, 0.1310909443334226, 0.12974703900554577, 0.12738016096540442, 0.12402039764571221,
         0.11892559237286555, 0.11150402563682907, 0.09912806164727636, 0.07784621757447988, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0])
    y_vector_prop_on = np.array(
        [0.04619, 0.13856, 0.23093, 0.3233, 0.41567, 0.50804, 0.60041, 0.71159, 0.84215, 0.97345, 1.10541, 1.23795,
         1.37098, 1.50444, 1.63824, 1.77229, 1.90652, 2.04083, 2.17515, 2.3094, 2.44348, 2.57731, 2.71082, 2.84392,
         2.97653, 3.10858, 3.23997, 3.37064, 3.50051, 3.62951, 3.75757, 3.88462, 4.01059, 4.13542, 4.25904, 4.38141,
         4.50245, 4.62213, 4.74039, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    cl_vector_only_prop = np.array(
        [-0.0008, -0.0182, -0.0237, -0.0288, -0.0355, -0.0492, -0.0735, -0.0749, -0.0296, 0.1332, 0.122, 0.0184, 0.0133,
         0.1141, 0.1704, 0.0053, -0.0109, 0.0654, 0.1618, 0.0208, -0.0486, -0.0233, 0.1395, 0.05, -0.0616, -0.0225,
         0.173, 0.174, -0.0087, -0.025, -0.0363, 0.1426, 0.1145, -0.0966, -0.1327, -0.1526, -0.1463, -0.121, -0.0979, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    ivc.add_output("data:aerodynamics:wing:low_speed:CL_vector", cl_vector)
    ivc.add_output("data:aerodynamics:wing:low_speed:chord_vector", chord_vector, units="m")
    ivc.add_output("data:aerodynamics:wing:low_speed:Y_vector", y_vector, units="m")
    ivc.add_output("data:aerodynamics:slipstream:wing:prop_on:Y_vector", y_vector_prop_on, units="m")
    ivc.add_output("data:aerodynamics:slipstream:wing:only_prop:CL_vector", cl_vector_only_prop, units="m")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(AerostructuralLoadX57(), ivc)
    max_rbm_mass_condition = problem.get_val("data:loads:max_rbm:mass", units="kg")
    assert max_rbm_mass_condition == pytest.approx(1361., abs=1e-1)
    max_rbm_lf_condition = problem.get_val("data:loads:max_rbm:load_factor")
    assert max_rbm_lf_condition == pytest.approx(3.8, abs=1e-2)
    lift_rbm_diagram = problem.get_val("data:loads:max_rbm:lift_rbm", units="N*m")
    lift_rbm = lift_rbm_diagram[0]
    assert lift_rbm == pytest.approx(87644.553, abs=1)
    weight_rbm_diagram = problem.get_val("data:loads:max_rbm:weight_rbm", units="N*m")
    weight_rbm = weight_rbm_diagram[0]
    assert weight_rbm == pytest.approx(-29919.19, abs=1)


def test_compute_mass_distribution_x57():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(StructuralLoadsX57()), __file__, XML_FILE)
    load_factor_shear = 4.0
    ivc.add_output("data:loads:max_shear:load_factor", load_factor_shear)
    load_factor_rbm = 4.0
    ivc.add_output("data:loads:max_rbm:load_factor", load_factor_rbm)
    y_vector = np.array(
        [0.04619, 0.13856, 0.23093, 0.3233, 0.41567, 0.50804, 0.60041, 0.71159, 0.84215, 0.97345, 1.10541, 1.23795,
         1.37098, 1.50444, 1.63824, 1.77229, 1.90652, 2.04083, 2.17515, 2.3094, 2.44348, 2.57731, 2.71082, 2.84392,
         2.97653, 3.10858, 3.23997, 3.37064, 3.50051, 3.62951, 3.75757, 3.88462, 4.01059, 4.13542, 4.25904, 4.38141,
         4.50245, 4.62213, 4.74039, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    chord_vector = np.array(
        [0.73944, 0.73944, 0.73944, 0.73944, 0.73944, 0.73944, 0.73944, 0.73593, 0.72888, 0.7218, 0.71468, 0.70753,
         0.70035, 0.69315, 0.68593, 0.6787, 0.67146, 0.66421, 0.65696, 0.64972, 0.64249, 0.63527, 0.62806, 0.62088,
         0.61373, 0.6066, 0.59951, 0.59246, 0.58546, 0.5785, 0.57159, 0.56473, 0.55794, 0.5512, 0.54453, 0.53793,
         0.5314, 0.52494, 0.51856, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    ivc.add_output("data:aerodynamics:wing:low_speed:chord_vector", chord_vector, units="m")
    ivc.add_output("data:aerodynamics:wing:low_speed:Y_vector", y_vector, units="m")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(StructuralLoadsX57(), ivc)
    point_mass_array = problem.get_val("data:loads:structure:ultimate:force_distribution:point_mass", units="N/m")
    point_mass_result = np.array([-0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0.,
                                  -0., -6772.39, -6772.39, -6772.39, -6772.39, -6772.39, -0., -0., -0., -0., -0., -0.,
                                  -6772.39, -6772.39, -6772.39, -6772.39, -6772.39, -0., -0., -0., -0., -0., -0.,
                                  -6772.39, -6772.39, -6772.39, -6772.39, -6772.39, -6772.39, -0., -0., -0., -0., -0.,
                                  -0., -6772.39, -6772.39, -6772.39, -6772.39, -6772.39, -0., -0., -0., -0., -0., -0.,
                                  -6772.39, -6772.39, -6772.39, -6772.39, -6772.39, -0., -0., -0., -0., -0., -0., -0.,
                                  -6772.39, -6772.39, -6772.39, -6772.39, -6772.39, -0., -0., -0., -0., -0., -0., -0.,
                                  -0., -0., -0., -53114.58, -53114.58, -53114.58, -53114.58, -53114.58, -0., -0., 0.,
                                  0., 0.])
    assert np.max(np.abs(point_mass_array - point_mass_result)) <= 1e-1
    wing_mass_array = problem.get_val("data:loads:structure:ultimate:force_distribution:wing", units="N/m")
    wing_mass_result = np.array([-805.8, -784., -784., -784., -784., -784., -784., -784., -783.92, -783.89, -783.57,
                                 -783.25, -782.92, -782.6, -782.57, -780.28, -772.81, -770.97, -770.92, -770.37,
                                 -769.82, -769.27, -768.72, -768.66, -765.3, -757.75, -750.17, -742.56, -738.91,
                                 -738.85, -738.3, -737.76, -737.21, -736.66, -736.6, -734.92, -727.27, -719.6, -711.93,
                                 -706.17, -706.11, -705.56, -705.01, -704.46, -704.24, -703.91, -703.86, -696.55,
                                 -688.88, -681.21, -673.55, -673.43, -673.37, -672.82, -672.27, -671.72, -671.17,
                                 -671.11, -665.91, -658.3, -650.72, -643.16, -640.67, -640.62, -640.07, -639.52,
                                 -638.97, -638.42, -638.36, -635.64, -628.16, -620.74, -613.36, -606.04, -603.27,
                                 -603.21, -602.66, -602.11, -601.56, -601.02, -600.96, -598.76, -591.56, -584.42,
                                 -577.35, -570.35, -563.42, -556.58, -549.83, -549.81, -549.94, -551.9, -553.85,
                                 -555.81, -557.76, -557.96, -561.94, 0., 0., 0.])
    assert np.max(np.abs(wing_mass_result - wing_mass_array)) <= 1e-1
    fuel_mass_array = problem.get_val("data:loads:structure:ultimate:force_distribution:fuel", units="N/m")
    fuel_mass_result = np.array([-0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0.,
                                 -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0.,
                                 -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0.,
                                 -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0.,
                                 -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0.,
                                 -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., 0., 0., 0.])
    assert np.max(np.abs(fuel_mass_result - fuel_mass_array)) <= 1e-1


def test_compute_structure_shear():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(StructuralLoadsX57()), __file__, XML_FILE)
    load_factor_shear = 4.0
    ivc.add_output("data:loads:max_shear:load_factor", load_factor_shear)
    load_factor_rbm = 4.0
    ivc.add_output("data:loads:max_rbm:load_factor", load_factor_rbm)
    y_vector = np.array(
        [0.04619, 0.13856, 0.23093, 0.3233, 0.41567, 0.50804, 0.60041, 0.71159, 0.84215, 0.97345, 1.10541, 1.23795,
         1.37098, 1.50444, 1.63824, 1.77229, 1.90652, 2.04083, 2.17515, 2.3094, 2.44348, 2.57731, 2.71082, 2.84392,
         2.97653, 3.10858, 3.23997, 3.37064, 3.50051, 3.62951, 3.75757, 3.88462, 4.01059, 4.13542, 4.25904, 4.38141,
         4.50245, 4.62213, 4.74039, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    chord_vector = np.array(
        [0.73944, 0.73944, 0.73944, 0.73944, 0.73944, 0.73944, 0.73944, 0.73593, 0.72888, 0.7218, 0.71468, 0.70753,
         0.70035, 0.69315, 0.68593, 0.6787, 0.67146, 0.66421, 0.65696, 0.64972, 0.64249, 0.63527, 0.62806, 0.62088,
         0.61373, 0.6066, 0.59951, 0.59246, 0.58546, 0.5785, 0.57159, 0.56473, 0.55794, 0.5512, 0.54453, 0.53793,
         0.5314, 0.52494, 0.51856, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    ivc.add_output("data:aerodynamics:wing:low_speed:chord_vector", chord_vector, units="m")
    ivc.add_output("data:aerodynamics:wing:low_speed:Y_vector", y_vector, units="m")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(StructuralLoadsX57(), ivc)
    point_mass_array = problem.get_val("data:loads:structure:ultimate:shear:point_mass", units="N")
    point_mass_result = np.array([-3684.64, -3684.64, -3684.64, -3684.64, -3684.64, -3684.64, -3684.64, -3684.64,
                                  -3684.64, -3684.64, -3684.64, -3684.64, -3684.64, -3684.64, -3684.64, -3684.64,
                                  -3684.64, -3684.64, -3681.25, -3616.23, -3551.22, -3486.21, -3421.19, -3417.8,
                                  -3417.8, -3417.8, -3417.8, -3417.8, -3417.8, -3414.42, -3349.4, -3284.39, -3219.37,
                                  -3154.36, -3150.97, -3150.97, -3150.97, -3150.97, -3150.97, -3150.97, -3147.59,
                                  -3082.57, -3017.56, -2952.54, -2926.12, -2887.53, -2884.14, -2884.14, -2884.14,
                                  -2884.14, -2884.14, -2884.14, -2880.75, -2815.74, -2750.72, -2685.71, -2620.69,
                                  -2617.31, -2617.31, -2617.31, -2617.31, -2617.31, -2617.31, -2613.92, -2548.91,
                                  -2483.89, -2418.88, -2353.86, -2350.48, -2350.48, -2350.48, -2350.48, -2350.48,
                                  -2350.48, -2350.48, -2347.09, -2282.07, -2217.06, -2152.05, -2087.03, -2083.64,
                                  -2083.64, -2083.64, -2083.64, -2083.64, -2083.64, -2083.64, -2083.64, -2083.64,
                                  -2083.64, -2066.16, -1556.26, -1046.36, -536.46, -26.56, 0., 0., 0., 0., 0.])
    assert np.max(np.abs(point_mass_array - point_mass_result)) <= 1e-1
    wing_mass_array = problem.get_val("data:loads:structure:ultimate:shear:wing", units="N")
    wing_mass_result = np.array([-3270.65, -3233.94, -3161.52, -3089.1, -3016.68, -2944.26, -2871.85, -2799.43,
                                 -2797.55, -2796.77, -2789.25, -2781.73, -2774.21, -2766.69, -2765.91, -2712.47,
                                 -2611.08, -2586.34, -2585.57, -2578.17, -2570.78, -2563.39, -2556.01, -2555.24,
                                 -2510.11, -2409.62, -2309.69, -2210.4, -2163.19, -2162.45, -2155.36, -2148.28,
                                 -2141.2, -2134.12, -2133.39, -2111.81, -2013.99, -1917.01, -1820.93, -1749.6,
                                 -1748.89, -1742.11, -1735.34, -1728.58, -1725.83, -1721.82, -1721.11, -1631.75,
                                 -1538.76, -1446.91, -1356.25, -1354.74, -1354.07, -1347.61, -1341.15, -1334.7,
                                 -1328.25, -1327.58, -1266.84, -1178.71, -1091.91, -1006.49, -978.63, -977.99, -971.84,
                                 -965.7, -959.57, -953.43, -952.8, -922.48, -839.91, -758.81, -679.21, -601.13,
                                 -571.92, -571.31, -565.52, -559.74, -553.96, -548.19, -547.59, -524.6, -449.62,
                                 -376.22, -304.41, -234.19, -165.58, -98.56, -33.32, -33.14, -32.77, -27.48, -22.18,
                                 -16.85, -11.51, -10.95, 0., 0., 0., 0.])
    assert np.max(np.abs(wing_mass_result - wing_mass_array)) <= 1e-1
    fuel_mass_array = problem.get_val("data:loads:structure:ultimate:shear:fuel", units="N")
    fuel_mass_result = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    assert np.max(np.abs(fuel_mass_result - fuel_mass_array)) <= 1e-1


def test_compute_structure_bending():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(StructuralLoadsX57()), __file__, XML_FILE)
    load_factor_shear = 4.0
    ivc.add_output("data:loads:max_shear:load_factor", load_factor_shear)
    load_factor_rbm = 4.0
    ivc.add_output("data:loads:max_rbm:load_factor", load_factor_rbm)
    y_vector = np.array(
        [0.04619, 0.13856, 0.23093, 0.3233, 0.41567, 0.50804, 0.60041, 0.71159, 0.84215, 0.97345, 1.10541, 1.23795,
         1.37098, 1.50444, 1.63824, 1.77229, 1.90652, 2.04083, 2.17515, 2.3094, 2.44348, 2.57731, 2.71082, 2.84392,
         2.97653, 3.10858, 3.23997, 3.37064, 3.50051, 3.62951, 3.75757, 3.88462, 4.01059, 4.13542, 4.25904, 4.38141,
         4.50245, 4.62213, 4.74039, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    chord_vector = np.array(
        [0.73944, 0.73944, 0.73944, 0.73944, 0.73944, 0.73944, 0.73944, 0.73593, 0.72888, 0.7218, 0.71468, 0.70753,
         0.70035, 0.69315, 0.68593, 0.6787, 0.67146, 0.66421, 0.65696, 0.64972, 0.64249, 0.63527, 0.62806, 0.62088,
         0.61373, 0.6066, 0.59951, 0.59246, 0.58546, 0.5785, 0.57159, 0.56473, 0.55794, 0.5512, 0.54453, 0.53793,
         0.5314, 0.52494, 0.51856, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    ivc.add_output("data:aerodynamics:wing:low_speed:chord_vector", chord_vector, units="m")
    ivc.add_output("data:aerodynamics:wing:low_speed:Y_vector", y_vector, units="m")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(StructuralLoadsX57(), ivc)
    point_mass_array = problem.get_val("data:loads:structure:ultimate:root_bending:point_mass", units="N*m")
    point_mass_result = np.array([-13647.72, -13477.53, -13137.18, -12796.83, -12456.48, -12116.13,
                                  -11775.78, -11435.43, -11426.62, -11422.94, -11387.57, -11352.2,
                                  -11316.82, -11281.45, -11277.77, -11025.77, -10544.71, -10426.58,
                                  -10422.89, -10387.87, -10353.46, -10319.68, -10286.53, -10283.11,
                                  -10082., -9630.99, -9178., -8723.33, -8505.52, -8502.1,
                                  -8469.64, -8437.79, -8406.58, -8375.98, -8372.83, -8280.4,
                                  -7858.8, -7436.41, -7013.46, -6696.44, -6693.29, -6663.39,
                                  -6634.11, -6605.45, -6593.98, -6577.42, -6574.53, -6206.46,
                                  -5819.26, -5432.56, -5046.57, -5040.11, -5037.22, -5009.88,
                                  -4983.16, -4957.07, -4931.59, -4928.98, -4691.15, -4342.78,
                                  -3995.7, -3650.09, -3536.51, -3533.89, -3509.11, -3484.95,
                                  -3461.42, -3438.51, -3436.16, -3324.29, -3017.15, -2711.89,
                                  -2408.68, -2107.68, -1994.11, -1991.76, -1969.54, -1947.95,
                                  -1926.98, -1906.63, -1904.54, -1824.67, -1562.2, -1302.09,
                                  -1044.51, -789.54, -537.33, -287.96, -42.26, -41.55,
                                  -40.18, -22.79, -10.3, -2.7, 0., 0.,
                                  0., 0., 0., 0.])
    assert np.max(np.abs(point_mass_array - point_mass_result)) <= 1e-1
    wing_mass_array = problem.get_val("data:loads:structure:ultimate:root_bending:wing", units="N*m")
    wing_mass_result = np.array([-7347.99, -7197.77, -6902.4, -6613.72, -6331.72, -6056.41,
                                 -5787.8, -5525.87, -5519.18, -5516.38, -5489.57, -5462.83,
                                 -5436.16, -5409.57, -5406.8, -5219.47, -4871.98, -4788.66,
                                 -4786.08, -4761.29, -4736.58, -4711.93, -4687.36, -4684.81,
                                 -4535.79, -4211.21, -3898.5, -3597.88, -3458.53, -3456.36,
                                 -3435.64, -3414.98, -3394.39, -3373.87, -3371.74, -3309.47,
                                 -3033.49, -2770.05, -2519.21, -2339.62, -2337.87, -2321.11,
                                 -2304.42, -2287.79, -2281.05, -2271.23, -2269.51, -2055.59,
                                 -1842.8, -1642.68, -1455.14, -1452.1, -1450.75, -1437.78,
                                 -1424.87, -1412.03, -1399.25, -1397.92, -1280.06, -1117.34,
                                 -966.82, -828.3, -785.23, -784.26, -774.9, -765.6,
                                 -756.35, -747.17, -746.22, -701.59, -586.48, -482.7,
                                 -389.98, -308.03, -279.69, -279.12, -273.66, -268.26,
                                 -262.91, -257.62, -257.08, -236.53, -175.19, -123.68,
                                 -81.63, -48.7, -24.54, -8.76, -1., -0.99,
                                 -0.97, -0.68, -0.44, -0.25, -0.12, -0.11,
                                 0., 0., 0., 0.])
    assert np.max(np.abs(wing_mass_result - wing_mass_array)) <= 1e-1
    fuel_mass_array = problem.get_val("data:loads:structure:ultimate:root_bending:fuel", units="N*m")
    fuel_mass_result = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    assert np.max(np.abs(fuel_mass_result - fuel_mass_array)) <= 1e-1


def test_compute_lift_distribution():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(AerodynamicLoadsX57()), __file__, XML_FILE)
    load_factor_shear = 4.0
    ivc.add_output("data:loads:max_shear:load_factor", load_factor_shear)
    ivc.add_output("data:loads:max_shear:cg_position", 2.759, units="m")
    ivc.add_output("data:loads:max_shear:mass", 1638.94, units="kg")
    load_factor_rbm = 4.0
    ivc.add_output("data:loads:max_rbm:load_factor", load_factor_rbm)
    ivc.add_output("data:loads:max_rbm:cg_position", 2.759, units="m")
    ivc.add_output("data:loads:max_rbm:mass", 1638.94, units="kg")
    y_vector_prop_on = np.array(
        [0.04619, 0.13856, 0.23093, 0.3233, 0.41567, 0.50804, 0.60041, 0.71159, 0.84215, 0.97345, 1.10541, 1.23795,
         1.37098, 1.50444, 1.63824, 1.77229, 1.90652, 2.04083, 2.17515, 2.3094, 2.44348, 2.57731, 2.71082, 2.84392,
         2.97653, 3.10858, 3.23997, 3.37064, 3.50051, 3.62951, 3.75757, 3.88462, 4.01059, 4.13542, 4.25904, 4.38141,
         4.50245, 4.62213, 4.74039, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    cl_vector_only_prop = np.array(
        [-0.0008, -0.0182, -0.0237, -0.0288, -0.0355, -0.0492, -0.0735, -0.0749, -0.0296, 0.1332, 0.122, 0.0184, 0.0133,
         0.1141, 0.1704, 0.0053, -0.0109, 0.0654, 0.1618, 0.0208, -0.0486, -0.0233, 0.1395, 0.05, -0.0616, -0.0225,
         0.173, 0.174, -0.0087, -0.025, -0.0363, 0.1426, 0.1145, -0.0966, -0.1327, -0.1526, -0.1463, -0.121, -0.0979, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    y_vector = np.array(
        [0.04619, 0.13856, 0.23093, 0.3233, 0.41567, 0.50804, 0.60041, 0.71159, 0.84215, 0.97345, 1.10541, 1.23795,
         1.37098, 1.50444, 1.63824, 1.77229, 1.90652, 2.04083, 2.17515, 2.3094, 2.44348, 2.57731, 2.71082, 2.84392,
         2.97653, 3.10858, 3.23997, 3.37064, 3.50051, 3.62951, 3.75757, 3.88462, 4.01059, 4.13542, 4.25904, 4.38141,
         4.50245, 4.62213, 4.74039, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    chord_vector = np.array(
        [0.73944, 0.73944, 0.73944, 0.73944, 0.73944, 0.73944, 0.73944, 0.73593, 0.72888, 0.7218, 0.71468, 0.70753,
         0.70035, 0.69315, 0.68593, 0.6787, 0.67146, 0.66421, 0.65696, 0.64972, 0.64249, 0.63527, 0.62806, 0.62088,
         0.61373, 0.6066, 0.59951, 0.59246, 0.58546, 0.5785, 0.57159, 0.56473, 0.55794, 0.5512, 0.54453, 0.53793,
         0.5314, 0.52494, 0.51856, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    cl_vector = np.array(
        [0.13405957102783722, 0.13404954188359933, 0.13398936701817202, 0.13389910472003103, 0.13398936701817202,
         0.1333073852099957, 0.1330767148925243, 0.13374866755646272, 0.13426015391259497, 0.13496219400924706,
         0.13463123224939677, 0.13483181513415454, 0.13492207743229553, 0.13512266031705325, 0.13551379694233084,
         0.13576452554827803, 0.13570435068285072, 0.1357143798270886, 0.1355438843750445, 0.13566423410589917,
         0.13594505014456001, 0.13589490442337057, 0.13572440897132648, 0.1356341466731855, 0.13536335977876254,
         0.13509257288433957, 0.13433035792226017, 0.13376872584493849, 0.13335753093118513, 0.13288616115200444,
         0.13210388790144922, 0.1310909443334226, 0.12974703900554577, 0.12738016096540442, 0.12402039764571221,
         0.11892559237286555, 0.11150402563682907, 0.09912806164727636, 0.07784621757447988, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0])
    ivc.add_output("data:aerodynamics:wing:low_speed:CL_vector", cl_vector)
    ivc.add_output("data:aerodynamics:wing:low_speed:chord_vector", chord_vector, units="m")
    ivc.add_output("data:aerodynamics:wing:low_speed:Y_vector", y_vector, units="m")
    ivc.add_output("data:aerodynamics:slipstream:wing:only_prop:CL_vector", cl_vector_only_prop)
    ivc.add_output("data:aerodynamics:slipstream:wing:prop_on:Y_vector", y_vector_prop_on, units="m")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(AerodynamicLoadsX57(), ivc)
    lift_array = problem.get_val("data:loads:aerodynamic:ultimate:force_distribution", units="N/m")
    lift_result = np.array([8341.73, 8116.07, 8077.58, 8061.97, 8045.4, 8036.28, 7965.16,
                            7898.3, 7898.31, 7898.31, 7898.32, 7898.33, 7898.33, 7898.33,
                            7898.33, 7898.27, 7950.34, 8026.78, 8029.16, 8051.95, 8074.71,
                            8097.43, 8120.1, 8122.46, 8260.53, 8136.11, 7850.56, 7765.55,
                            7831.64, 7832.67, 7842.54, 7852.38, 7862.2, 7871.99, 7873.01,
                            7902.78, 7956.12, 7556.33, 7440.4, 7492.69, 7493.2, 7498.09,
                            7502.97, 7507.83, 7509.79, 7513.92, 7514.64, 7605.08, 7257.99,
                            7060.74, 7026.1, 7029.73, 7031.34, 7046.83, 7062.29, 7077.71,
                            7093.1, 7094.7, 7238.59, 6987.66, 6691.96, 6670.58, 6747.23,
                            6748.99, 6765.83, 6782.64, 6799.41, 6816.13, 6817.87, 6900.22,
                            6793.57, 6378.71, 6252.79, 6122.48, 6190.54, 6191.94, 6205.35,
                            6218.73, 6232.08, 6245.39, 6246.77, 6299.56, 6116.26, 5593.01,
                            5317.64, 4997.24, 4623.47, 4074.34, 3158.96, 3156.34, 3122.23,
                            2623.08, 2120.31, 1613.92, 1103.93, 1050.6, 0., 0., 0., 0.])
    assert np.max(np.abs(lift_array - lift_result)) <= 1e-1
