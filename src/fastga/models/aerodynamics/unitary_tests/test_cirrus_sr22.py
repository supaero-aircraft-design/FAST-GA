"""
Test module for aerodynamics groups
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

import os.path as pth
import os
import pandas as pd
import openmdao.api as om
from openmdao.core.component import Component
import numpy as np
from platform import system
import tempfile
from pathlib import Path
from tempfile import TemporaryDirectory
import pytest
from typing import Union
import time

from fastoad.io import VariableIO
from fastoad.module_management.service_registry import RegisterPropulsion
from fastoad.model_base import FlightPoint
from fastoad.model_base.propulsion import IOMPropulsionWrapper

from ..components.cd0 import Cd0
from ..external.xfoil.xfoil_polar import XfoilPolar
from ..external.xfoil import resources
from ..external.vlm import ComputeAEROvlm, ComputeVNvlmNoVH
from ..external.openvsp import ComputeAEROopenvsp, ComputeVNopenvspNoVH
from ..external.openvsp.compute_aero_slipstream import ComputeSlipstreamOpenvsp
from ..components import ComputeExtremeCL, ComputeUnitReynolds, ComputeCnBetaFuselage, ComputeLDMax, \
    ComputeDeltaHighLift, Compute2DHingeMomentsTail, Compute3DHingeMomentsTail, ComputeMachInterpolation
from ..aerodynamics_high_speed import AerodynamicsHighSpeed
from ..aerodynamics_low_speed import AerodynamicsLowSpeed
from ..external.vlm.compute_aero import DEFAULT_WING_AIRFOIL, DEFAULT_HTP_AIRFOIL
from ..constants import SPAN_MESH_POINT, POLAR_POINT_COUNT

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs
from tests.xfoil_exe.get_xfoil import get_xfoil_path

from fastga.models.propulsion.fuel_propulsion.base import AbstractFuelPropulsion
from fastga.models.propulsion.propulsion import IPropulsion

RESULTS_FOLDER = pth.join(pth.dirname(__file__), "results")
xfoil_path = None if system() == "Windows" else get_xfoil_path()

XML_FILE = "cirrus_sr22.xml"
ENGINE_WRAPPER = "test.wrapper.aerodynamics.cirrus.dummy_engine"


class DummyEngine(AbstractFuelPropulsion):

    def __init__(self,
                 max_power: float,
                 design_altitude: float,
                 design_speed: float,
                 fuel_type: float,
                 strokes_nb: float,
                 prop_layout: float,
                 ):
        """
        Dummy engine model returning nacelle aerodynamic drag force.

        """
        super().__init__()
        self.prop_layout = prop_layout
        self.max_power = max_power
        self.design_altitude = design_altitude
        self.design_speed = design_speed
        self.fuel_type = fuel_type
        self.strokes_nb = strokes_nb

    def compute_flight_points(self, flight_points: Union[FlightPoint, pd.DataFrame]):
        flight_points.thrust = 1800.0
        flight_points.sfc = 0.0

    def compute_weight(self) -> float:
        return 0.0

    def compute_dimensions(self) -> (float, float, float, float, float, float):
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def compute_drag(self, mach, unit_reynolds, l0_wing):
        if mach < 0.15:
            return 0.01934377
        else:
            return 0.01771782

    def get_consumed_mass(self, flight_point: FlightPoint, time_step: float) -> float:
        return 0.0


class DummyEngineWrapper(IOMPropulsionWrapper):
    def setup(self, component: Component):
        component.add_input("data:propulsion:IC_engine:max_power", np.nan, units="W")
        component.add_input("data:propulsion:IC_engine:fuel_type", np.nan)
        component.add_input("data:propulsion:IC_engine:strokes_nb", np.nan)
        component.add_input("data:TLAR:v_cruise", np.nan, units="m/s")
        component.add_input("data:mission:sizing:main_route:cruise:altitude", np.nan, units="m")
        component.add_input("data:geometry:propulsion:layout", np.nan)

    @staticmethod
    def get_model(inputs) -> IPropulsion:
        engine_params = {
            "max_power": inputs["data:propulsion:IC_engine:max_power"],
            "design_altitude": inputs["data:mission:sizing:main_route:cruise:altitude"],
            "design_speed": inputs["data:TLAR:v_cruise"],
            "fuel_type": inputs["data:propulsion:IC_engine:fuel_type"],
            "strokes_nb": inputs["data:propulsion:IC_engine:strokes_nb"],
            "prop_layout": inputs["data:geometry:propulsion:layout"]
        }

        return DummyEngine(**engine_params)


RegisterPropulsion(ENGINE_WRAPPER)(DummyEngineWrapper)


def _create_tmp_directory() -> TemporaryDirectory:
    """Provide temporary directory for calculation."""

    for tmp_base_path in [None, pth.join(str(Path.home()), ".fast")]:
        if tmp_base_path is not None:
            os.makedirs(tmp_base_path, exist_ok=True)
        tmp_directory = tempfile.TemporaryDirectory(prefix="x", dir=tmp_base_path)
        break

    return tmp_directory


def reshape_curve(y, cl):
    """ Reshape data from openvsp/vlm lift curve """
    for idx in range(len(y)):
        if np.sum(y[idx:len(y)] == 0) == (len(y) - idx):
            y = y[0:idx]
            cl = cl[0:idx]
            break

    return y, cl


def reshape_polar(cl, cdp):
    """ Reshape data from xfoil polar vectors """
    for idx in range(len(cl)):
        if np.sum(cl[idx:len(cl)] == 0) == (len(cl) - idx):
            cl = cl[0:idx]
            cdp = cdp[0:idx]
            break
    return cl, cdp


def clear_polar_results():
    # Clear saved polar results (for wing and htp airfoils)
    if pth.exists(pth.join(resources.__path__[0], DEFAULT_WING_AIRFOIL.replace('af', 'csv'))):
        os.remove(pth.join(resources.__path__[0], DEFAULT_WING_AIRFOIL.replace('af', 'csv')))
    if pth.exists(pth.join(resources.__path__[0], DEFAULT_WING_AIRFOIL.replace('.af', '_sym.csv'))):
        os.remove(pth.join(resources.__path__[0], DEFAULT_WING_AIRFOIL.replace('.af', '_sym.csv')))
    if pth.exists(pth.join(resources.__path__[0], DEFAULT_HTP_AIRFOIL.replace('af', 'csv'))):
        os.remove(pth.join(resources.__path__[0], DEFAULT_HTP_AIRFOIL.replace('af', 'csv')))


def test_compute_reynolds():
    """ Tests high and low speed reynolds calculation """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeUnitReynolds()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeUnitReynolds(), ivc)
    mach = problem["data:aerodynamics:cruise:mach"]
    assert mach == pytest.approx(0.2488214, abs=1e-4)
    reynolds = problem.get_val("data:aerodynamics:cruise:unit_reynolds", units="m**-1")
    assert reynolds == pytest.approx(4629639, abs=1)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeUnitReynolds(low_speed_aero=True)), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeUnitReynolds(low_speed_aero=True), ivc)
    mach = problem["data:aerodynamics:low_speed:mach"]
    assert mach == pytest.approx(0.1194, abs=1e-4)
    reynolds = problem.get_val("data:aerodynamics:low_speed:unit_reynolds", units="m**-1")
    assert reynolds == pytest.approx(2782216, abs=1)


def test_cd0_high_speed():
    """ Tests drag coefficient @ high speed """

    # Research independent input value in .xml file
    # noinspection PyTypeChecker
    ivc = get_indep_var_comp(list_inputs(Cd0(propulsion_id=ENGINE_WRAPPER)), __file__, XML_FILE)
    ivc.add_output("data:aerodynamics:cruise:mach", 0.2457)
    ivc.add_output("data:aerodynamics:cruise:unit_reynolds", 4571770, units="m**-1")

    # noinspection PyTypeChecker
    problem = run_system(Cd0(propulsion_id=ENGINE_WRAPPER), ivc)
    cd0_wing = problem["data:aerodynamics:wing:cruise:CD0"]
    assert cd0_wing == pytest.approx(0.00525, abs=1e-5)
    cd0_fus = problem["data:aerodynamics:fuselage:cruise:CD0"]
    assert cd0_fus == pytest.approx(0.00675, abs=1e-5)
    cd0_ht = problem["data:aerodynamics:horizontal_tail:cruise:CD0"]
    assert cd0_ht == pytest.approx(0.00143, abs=1e-5)
    cd0_vt = problem["data:aerodynamics:vertical_tail:cruise:CD0"]
    assert cd0_vt == pytest.approx(0.00110, abs=1e-5)
    cd0_nac = problem["data:aerodynamics:nacelles:cruise:CD0"]
    assert cd0_nac == pytest.approx(0.00000, abs=1e-5)
    cd0_lg = problem["data:aerodynamics:landing_gear:cruise:CD0"]
    assert cd0_lg == pytest.approx(0.002899, abs=1e-5)
    cd0_other = problem["data:aerodynamics:other:cruise:CD0"]
    assert cd0_other == pytest.approx(0.00441, abs=1e-5)
    cd0_total = 1.25 * (cd0_wing + cd0_fus + cd0_ht + cd0_vt + cd0_nac + cd0_lg + cd0_other)
    assert cd0_total == pytest.approx(0.02733, abs=1e-5)


def test_cd0_low_speed():
    """ Tests drag coefficient @ low speed """

    # Research independent input value in .xml file
    # noinspection PyTypeChecker
    ivc = get_indep_var_comp(list_inputs(Cd0(propulsion_id=ENGINE_WRAPPER, low_speed_aero=True)), __file__, XML_FILE)
    ivc.add_output("data:aerodynamics:low_speed:mach", 0.1149)  # correction to compensate old version conversion error
    ivc.add_output("data:aerodynamics:low_speed:unit_reynolds", 2613822, units="m**-1")  # correction to ...

    # noinspection PyTypeChecker
    problem = run_system(Cd0(propulsion_id=ENGINE_WRAPPER, low_speed_aero=True), ivc)
    cd0_wing = problem["data:aerodynamics:wing:low_speed:CD0"]
    assert cd0_wing == pytest.approx(0.00578, abs=1e-5)
    cd0_fus = problem["data:aerodynamics:fuselage:low_speed:CD0"]
    assert cd0_fus == pytest.approx(0.00753, abs=1e-5)
    cd0_ht = problem["data:aerodynamics:horizontal_tail:low_speed:CD0"]
    assert cd0_ht == pytest.approx(0.00159, abs=1e-5)
    cd0_vt = problem["data:aerodynamics:vertical_tail:low_speed:CD0"]
    assert cd0_vt == pytest.approx(0.00122, abs=1e-5)
    cd0_nac = problem["data:aerodynamics:nacelles:low_speed:CD0"]
    assert cd0_nac == pytest.approx(0.00000, abs=1e-5)
    cd0_lg = problem["data:aerodynamics:landing_gear:low_speed:CD0"]
    assert cd0_lg == pytest.approx(0.00289, abs=1e-5)
    cd0_other = problem["data:aerodynamics:other:low_speed:CD0"]
    assert cd0_other == pytest.approx(0.00441, abs=1e-5)
    cd0_total = 1.25 * (cd0_wing + cd0_fus + cd0_ht + cd0_vt + cd0_nac + cd0_lg + cd0_other)
    assert cd0_total == pytest.approx(0.029307, abs=1e-5)


def test_polar():
    """ Tests polar execution (XFOIL) @ high and low speed """

    # Clear saved polar results (for wing and htp airfoils)
    clear_polar_results()

    # Define high-speed parameters (with .xml file and additional inputs)
    ivc = get_indep_var_comp(list_inputs(XfoilPolar()), __file__, XML_FILE)
    ivc.add_output("xfoil:mach", 0.245)
    ivc.add_output("xfoil:reynolds", 4571770 * 1.549)

    # Run problem and check obtained value(s) is/(are) correct
    xfoil_comp = XfoilPolar(
        alpha_start=0.0, alpha_end=25.0, iter_limit=20, xfoil_exe_path=xfoil_path
    )
    problem = run_system(xfoil_comp, ivc)
    cl = problem["xfoil:CL"]
    cdp = problem["xfoil:CDp"]
    cl, cdp = reshape_polar(cl, cdp)
    cdp_1 = np.interp(1.0, cl, cdp)
    assert cdp_1 == pytest.approx(0.00464, abs=1e-4)

    # Define low-speed parameters (with .xml file and additional inputs)
    ivc = get_indep_var_comp(list_inputs(XfoilPolar()), __file__, XML_FILE)
    ivc.add_output("xfoil:mach", 0.1179)
    ivc.add_output("xfoil:reynolds", 2746999 * 1.549)

    # Run problem and check obtained value(s) is/(are) correct
    xfoil_comp = XfoilPolar(
        alpha_start=0.0, alpha_end=25.0, iter_limit=20, xfoil_exe_path=xfoil_path
    )
    problem = run_system(xfoil_comp, ivc)
    cl = problem["xfoil:CL"]
    cdp = problem["xfoil:CDp"]
    cl_max_2d = problem["xfoil:CL_max_2D"]
    assert cl_max_2d == pytest.approx(1.6833, abs=1e-4)
    cl, cdp = reshape_polar(cl, cdp)
    cdp_1 = np.interp(1.0, cl, cdp)
    assert cdp_1 == pytest.approx(0.00488, abs=1e-4)


def test_vlm_comp_high_speed():
    """ Tests vlm f @ high speed """

    # Create result temporary directory
    results_folder = _create_tmp_directory()

    # Clear saved polar results (for wing and htp airfoils)
    clear_polar_results()

    # Research independent input value in .xml file
    # noinspection PyTypeChecker
    ivc = get_indep_var_comp(list_inputs(ComputeAEROvlm()), __file__, XML_FILE)

    mach_interpolation = True
    # Run problem and check obtained value(s) is/(are) correct
    start = time.time()
    # noinspection PyTypeChecker
    problem = run_system(ComputeAEROvlm(result_folder_path=results_folder.name,
                                        compute_mach_interpolation=mach_interpolation), ivc)
    stop = time.time()
    duration_1st_run = stop - start
    cl0_wing = problem["data:aerodynamics:wing:cruise:CL0_clean"]
    assert cl0_wing == pytest.approx(0.1367, abs=1e-4)
    cl_alpha_wing = problem.get_val("data:aerodynamics:wing:cruise:CL_alpha", units="rad**-1")
    assert cl_alpha_wing == pytest.approx(5.245, abs=1e-3)
    cm0 = problem["data:aerodynamics:wing:cruise:CM0_clean"]
    assert cm0 == pytest.approx(-0.0510, abs=1e-4)
    coef_k_wing = problem["data:aerodynamics:wing:cruise:induced_drag_coefficient"]
    if mach_interpolation:
        cl_alpha_vector = problem["data:aerodynamics:aircraft:mach_interpolation:CL_alpha_vector"]
        assert cl_alpha_vector == pytest.approx([5.235, 5.235, 5.297, 5.381, 5.484, 5.606], abs=1e-2)
        mach_vector = problem["data:aerodynamics:aircraft:mach_interpolation:mach_vector"]
        assert mach_vector == pytest.approx([0., 0.15, 0.214, 0.275, 0.332, 0.386], abs=1e-2)
    assert coef_k_wing == pytest.approx(0.0412, abs=1e-4)
    cl0_htp = problem["data:aerodynamics:horizontal_tail:cruise:CL0"]
    assert cl0_htp == pytest.approx(-0.0077, abs=1e-4)
    cl_alpha_htp = problem.get_val("data:aerodynamics:horizontal_tail:cruise:CL_alpha", units="rad**-1")
    assert cl_alpha_htp == pytest.approx(0.6020, abs=1e-4)
    cl_alpha_htp_isolated = problem.get_val("data:aerodynamics:horizontal_tail:cruise:CL_alpha_isolated",
                                            units="rad**-1")
    assert cl_alpha_htp_isolated == pytest.approx(0.8974, abs=1e-4)
    coef_k_htp = problem["data:aerodynamics:horizontal_tail:cruise:induced_drag_coefficient"]
    assert coef_k_htp == pytest.approx(0.3225, abs=1e-4)

    # Run problem 2nd time to check time reduction
    start = time.time()
    # noinspection PyTypeChecker
    run_system(ComputeAEROvlm(result_folder_path=results_folder.name), ivc)
    stop = time.time()
    duration_2nd_run = stop - start
    assert (duration_2nd_run / duration_1st_run) <= 0.1

    # Remove existing result files
    results_folder.cleanup()


def test_vlm_comp_low_speed():
    """ Tests vlm components @ low speed """

    # Create result temporary directory
    results_folder = _create_tmp_directory()

    # Clear saved polar results (for wing and htp airfoils)
    clear_polar_results()

    # Research independent input value in .xml file
    # noinspection PyTypeChecker
    ivc = get_indep_var_comp(list_inputs(ComputeAEROvlm(low_speed_aero=True)), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    start = time.time()
    # noinspection PyTypeChecker
    problem = run_system(ComputeAEROvlm(low_speed_aero=True, result_folder_path=results_folder.name), ivc)
    stop = time.time()
    duration_1st_run = stop - start
    cl0_wing = problem["data:aerodynamics:wing:low_speed:CL0_clean"]
    assert cl0_wing == pytest.approx(0.1333, abs=1e-4)
    cl_alpha_wing = problem.get_val("data:aerodynamics:wing:low_speed:CL_alpha", units="rad**-1")
    assert cl_alpha_wing == pytest.approx(5.116, abs=1e-3)
    cm0 = problem["data:aerodynamics:wing:low_speed:CM0_clean"]
    assert cm0 == pytest.approx(-0.0497, abs=1e-4)
    y_vector_wing = np.array(
        [0.105, 0.315, 0.525, 0.853, 1.299, 1.745, 2.192, 2.638, 3.084,
         3.53, 3.902, 4.199, 4.497, 4.794, 5.091, 5.389, 5.686]
    )
    cl_vector_wing = np.array(
        [0.12, 0.122, 0.126, 0.143, 0.15, 0.154, 0.155, 0.156, 0.154,
         0.148, 0.123, 0.114, 0.109, 0.103, 0.096, 0.085, 0.065]
    )
    chord_vector_wing = np.array(
        [1.483, 1.483, 1.483, 1.4512, 1.3876, 1.324, 1.2604, 1.1968, 1.1332, 1.0696, 1.0166, 0.9742, 0.9318, 0.8894,
         0.847, 0.8046, 0.7622, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    )
    y, cl = reshape_curve(problem.get_val("data:aerodynamics:wing:low_speed:Y_vector", "m"),
                          problem["data:aerodynamics:wing:low_speed:CL_vector"])
    chord = problem.get_val("data:aerodynamics:wing:low_speed:chord_vector", "m")
    assert np.max(np.abs(y_vector_wing - y)) <= 1e-3
    assert np.max(np.abs(cl_vector_wing - cl)) <= 1e-3
    assert np.max(np.abs(chord_vector_wing - chord)) <= 1e-3
    coef_k_wing = problem["data:aerodynamics:wing:low_speed:induced_drag_coefficient"]
    assert coef_k_wing == pytest.approx(0.0395, abs=1e-4)
    cl0_htp = problem["data:aerodynamics:horizontal_tail:low_speed:CL0"]
    assert cl0_htp == pytest.approx(-0.0073, abs=1e-4)
    cl_ref_htp = problem["data:aerodynamics:horizontal_tail:low_speed:CL_ref"]
    assert cl_ref_htp == pytest.approx(0.0964, abs=1e-4)
    y_vector_htp = np.array(
        [0.057, 0.171, 0.285, 0.398, 0.512, 0.626, 0.74, 0.854, 0.968,
         1.081, 1.195, 1.309, 1.423, 1.537, 1.65, 1.764, 1.878]
    )
    cl_vector_htp = np.array(
        [0.098, 0.1, 0.101, 0.103, 0.103, 0.104, 0.104, 0.104, 0.104,
         0.103, 0.101, 0.099, 0.095, 0.09, 0.083, 0.072, 0.052]
    )
    y, cl = reshape_curve(problem.get_val("data:aerodynamics:horizontal_tail:low_speed:Y_vector", "m"),
                          problem["data:aerodynamics:horizontal_tail:low_speed:CL_vector"])
    assert np.max(np.abs(y_vector_htp - y)) <= 1e-3
    assert np.max(np.abs(cl_vector_htp - cl)) <= 1e-3
    cl_alpha_htp = problem.get_val("data:aerodynamics:horizontal_tail:low_speed:CL_alpha", units="rad**-1")
    assert cl_alpha_htp == pytest.approx(0.5943, abs=1e-4)
    cl_alpha_htp_isolated = problem.get_val("data:aerodynamics:horizontal_tail:low_speed:CL_alpha_isolated",
                                            units="rad**-1")
    assert cl_alpha_htp_isolated == pytest.approx(0.8754, abs=1e-4)
    coef_k_htp = problem["data:aerodynamics:horizontal_tail:low_speed:induced_drag_coefficient"]
    assert coef_k_htp == pytest.approx(0.3280, abs=1e-4)

    # Run problem 2nd time to check time reduction
    start = time.time()
    # noinspection PyTypeChecker
    run_system(ComputeAEROvlm(low_speed_aero=True, result_folder_path=results_folder.name), ivc)
    stop = time.time()
    duration_2nd_run = stop - start
    assert (duration_2nd_run / duration_1st_run) <= 0.1

    # Remove existing result files
    results_folder.cleanup()


def test_openvsp_comp_high_speed():
    """ Tests openvsp components @ high speed """

    # Create result temporary directory
    results_folder = _create_tmp_directory()

    # Research independent input value in .xml file
    # noinspection PyTypeChecker
    ivc = get_indep_var_comp(list_inputs(ComputeAEROopenvsp()), __file__, XML_FILE)

    mach_interpolation = True
    # Run problem and check obtained value(s) is/(are) correct
    start = time.time()
    # noinspection PyTypeChecker
    problem = run_system(ComputeAEROopenvsp(result_folder_path=results_folder.name,
                                            compute_mach_interpolation=mach_interpolation), ivc)
    stop = time.time()
    duration_1st_run = stop - start
    cl0_wing = problem["data:aerodynamics:wing:cruise:CL0_clean"]
    assert cl0_wing == pytest.approx(0.1269, abs=1e-4)
    cl_alpha_wing = problem.get_val("data:aerodynamics:wing:cruise:CL_alpha", units="rad**-1")
    assert cl_alpha_wing == pytest.approx(5.079, abs=1e-3)
    cm0 = problem["data:aerodynamics:wing:cruise:CM0_clean"]
    assert cm0 == pytest.approx(-0.0272, abs=1e-4)
    coef_k_wing = problem["data:aerodynamics:wing:cruise:induced_drag_coefficient"]
    assert coef_k_wing == pytest.approx(0.0380, abs=1e-4)
    if mach_interpolation:
        cl_alpha_vector = problem["data:aerodynamics:aircraft:mach_interpolation:CL_alpha_vector"]
        assert cl_alpha_vector == pytest.approx([5.51, 5.51, 5.56, 5.63, 5.71, 5.80], abs=1e-2)
        mach_vector = problem["data:aerodynamics:aircraft:mach_interpolation:mach_vector"]
        assert mach_vector == pytest.approx([0., 0.15, 0.21, 0.27, 0.33, 0.39], abs=1e-2)
    cl0_htp = problem["data:aerodynamics:horizontal_tail:cruise:CL0"]
    assert cl0_htp == pytest.approx(-0.0074, abs=1e-4)
    cl_alpha_htp = problem.get_val("data:aerodynamics:horizontal_tail:cruise:CL_alpha", units="rad**-1")
    assert cl_alpha_htp == pytest.approx(0.5179, abs=1e-4)
    cl_alpha_htp_isolated = problem.get_val("data:aerodynamics:horizontal_tail:cruise:CL_alpha_isolated",
                                            units="rad**-1")
    assert cl_alpha_htp_isolated == pytest.approx(0.9493, abs=1e-4)
    coef_k_htp = problem["data:aerodynamics:horizontal_tail:cruise:induced_drag_coefficient"]
    assert coef_k_htp == pytest.approx(0.7119, abs=1e-4)

    # Run problem 2nd time to check time reduction
    start = time.time()
    # noinspection PyTypeChecker
    run_system(ComputeAEROopenvsp(result_folder_path=results_folder.name), ivc)
    stop = time.time()
    duration_2nd_run = stop - start
    assert (duration_2nd_run / duration_1st_run) <= 0.01

    # Remove existing result files
    results_folder.cleanup()


def test_openvsp_comp_low_speed():
    """ Tests openvsp components @ low speed """

    # Create result temporary directory
    results_folder = _create_tmp_directory()

    # Research independent input value in .xml file
    # noinspection PyTypeChecker
    ivc = get_indep_var_comp(list_inputs(ComputeAEROopenvsp(low_speed_aero=True)), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    start = time.time()
    # noinspection PyTypeChecker
    problem = run_system(ComputeAEROopenvsp(low_speed_aero=True, result_folder_path=results_folder.name), ivc)
    stop = time.time()
    duration_1st_run = stop - start
    cl0_wing = problem["data:aerodynamics:wing:low_speed:CL0_clean"]
    assert cl0_wing == pytest.approx(0.1243, abs=1e-4)
    cl_alpha_wing = problem.get_val("data:aerodynamics:wing:low_speed:CL_alpha", units="rad**-1")
    assert cl_alpha_wing == pytest.approx(4.981, abs=1e-3)
    cm0 = problem["data:aerodynamics:wing:low_speed:CM0_clean"]
    assert cm0 == pytest.approx(-0.0266, abs=1e-4)
    y_vector_wing = np.array(
        [0.05, 0.14, 0.23, 0.32, 0.41, 0.5, 0.59, 0.72, 0.88, 1.04, 1.21,
         1.38, 1.54, 1.71, 1.88, 2.05, 2.21, 2.38, 2.55, 2.72, 2.89, 3.06,
         3.22, 3.39, 3.56, 3.72, 3.89, 4.05, 4.21, 4.37, 4.53, 4.69, 4.85,
         5.01, 5.16, 5.32, 5.47, 5.62, 5.77]
    )
    cl_vector_wing = np.array(
        [0.123, 0.123, 0.123, 0.123, 0.122, 0.122, 0.122, 0.122, 0.122,
         0.124, 0.125, 0.125, 0.126, 0.126, 0.127, 0.128, 0.128, 0.129,
         0.129, 0.129, 0.129, 0.129, 0.13, 0.13, 0.13, 0.13, 0.129,
         0.129, 0.128, 0.128, 0.127, 0.127, 0.126, 0.124, 0.122, 0.117,
         0.111, 0.098, 0.077]
    )
    y, cl = reshape_curve(problem.get_val("data:aerodynamics:wing:low_speed:Y_vector", "m"),
                          problem["data:aerodynamics:wing:low_speed:CL_vector"])
    assert np.max(np.abs(y_vector_wing - y)) <= 1e-2
    assert np.max(np.abs(cl_vector_wing - cl)) <= 1e-3
    coef_k_wing = problem["data:aerodynamics:wing:low_speed:induced_drag_coefficient"]
    assert coef_k_wing == pytest.approx(0.0380, abs=1e-4)
    cl0_htp = problem["data:aerodynamics:horizontal_tail:low_speed:CL0"]
    assert cl0_htp == pytest.approx(-0.0072, abs=1e-4)
    cl_ref_htp = problem["data:aerodynamics:horizontal_tail:low_speed:CL_ref"]
    assert cl_ref_htp == pytest.approx(0.08245, abs=1e-4)
    y_vector_htp = np.array(
        [0.04, 0.12, 0.2, 0.28, 0.36, 0.44, 0.52, 0.6, 0.69, 0.77, 0.85,
         0.93, 1.01, 1.09, 1.17, 1.25, 1.33, 1.41, 1.49, 1.57, 1.65, 1.73,
         1.81, 1.89]
    )
    cl_vector_htp = np.array(
        [0.079, 0.082, 0.083, 0.084, 0.087, 0.088, 0.085, 0.084, 0.087,
         0.087, 0.088, 0.088, 0.087, 0.087, 0.088, 0.088, 0.087, 0.085,
         0.083, 0.081, 0.078, 0.072, 0.061, 0.049]
    )
    y, cl = reshape_curve(problem.get_val("data:aerodynamics:horizontal_tail:low_speed:Y_vector", "m"),
                          problem["data:aerodynamics:horizontal_tail:low_speed:CL_vector"])
    assert np.max(np.abs(y_vector_htp - y)) <= 1e-2
    assert np.max(np.abs(cl_vector_htp - cl)) <= 1e-3
    cl_alpha_htp = problem.get_val("data:aerodynamics:horizontal_tail:low_speed:CL_alpha", units="rad**-1")
    assert cl_alpha_htp == pytest.approx(0.5137, abs=1e-4)
    cl_alpha_htp_isolated = problem.get_val("data:aerodynamics:horizontal_tail:low_speed:CL_alpha_isolated",
                                            units="rad**-1")
    assert cl_alpha_htp_isolated == pytest.approx(0.9321, abs=1e-4)
    coef_k_htp = problem["data:aerodynamics:horizontal_tail:low_speed:induced_drag_coefficient"]
    assert coef_k_htp == pytest.approx(0.7075, abs=1e-4)

    # Run problem 2nd time to check time reduction
    start = time.time()
    # noinspection PyTypeChecker
    run_system(ComputeAEROopenvsp(low_speed_aero=True, result_folder_path=results_folder.name), ivc)
    stop = time.time()
    duration_2nd_run = stop - start
    assert (duration_2nd_run / duration_1st_run) <= 0.1

    # Remove existing result files
    results_folder.cleanup()


def test_2d_hinge_moment():
    """ Tests tail hinge-moments """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(Compute2DHingeMomentsTail()), __file__, XML_FILE)
    ivc.add_output("data:aerodynamics:horizontal_tail:cruise:CL_alpha", 0.6826, units="rad**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(Compute2DHingeMomentsTail(), ivc)
    ch_alpha_2d = problem.get_val("data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_alpha_2D", units="rad**-1")
    assert ch_alpha_2d == pytest.approx(-0.3334, abs=1e-4)
    ch_delta_2d = problem.get_val("data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_delta_2D", units="rad**-1")
    assert ch_delta_2d == pytest.approx(-0.6347, abs=1e-4)


def test_3d_hinge_moment():
    """ Tests tail hinge-moments """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(Compute3DHingeMomentsTail()), __file__, XML_FILE)
    ivc.add_output("data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_alpha_2D", -0.3334, units="rad**-1")
    ivc.add_output("data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_delta_2D", -0.6347, units="rad**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(Compute3DHingeMomentsTail(), ivc)
    ch_alpha = problem.get_val("data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_alpha", units="rad**-1")
    assert ch_alpha == pytest.approx(-0.2437, abs=1e-4)
    ch_delta = problem.get_val("data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_delta", units="rad**-1")
    assert ch_delta == pytest.approx(-0.6777, abs=1e-4)


def test_high_lift():
    """ Tests high-lift contribution """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeDeltaHighLift()), __file__, XML_FILE)
    ivc.add_output("data:aerodynamics:low_speed:mach", 0.1149)
    ivc.add_output("data:aerodynamics:wing:low_speed:CL_alpha", 4.981, units="rad**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeDeltaHighLift(), ivc)
    delta_cl0_landing = problem["data:aerodynamics:flaps:landing:CL"]
    assert delta_cl0_landing == pytest.approx(0.7224, abs=1e-4)
    delta_clmax_landing = problem["data:aerodynamics:flaps:landing:CL_max"]
    assert delta_clmax_landing == pytest.approx(0.4650, abs=1e-4)
    delta_cm_landing = problem["data:aerodynamics:flaps:landing:CM"]
    assert delta_cm_landing == pytest.approx(-0.1228, abs=1e-4)
    delta_cd_landing = problem["data:aerodynamics:flaps:landing:CD"]
    assert delta_cd_landing == pytest.approx(0.01511, abs=1e-4)
    delta_cl0_takeoff = problem["data:aerodynamics:flaps:takeoff:CL"]
    assert delta_cl0_takeoff == pytest.approx(0.2694, abs=1e-4)
    delta_clmax_takeoff = problem["data:aerodynamics:flaps:takeoff:CL_max"]
    assert delta_clmax_takeoff == pytest.approx(0.09522, abs=1e-4)
    delta_cm_takeoff = problem["data:aerodynamics:flaps:takeoff:CM"]
    assert delta_cm_takeoff == pytest.approx(-0.0458, abs=1e-4)
    delta_cd_takeoff = problem["data:aerodynamics:flaps:takeoff:CD"]
    assert delta_cd_takeoff == pytest.approx(0.001221, abs=1e-4)
    cl_delta_elev = problem.get_val("data:aerodynamics:elevator:low_speed:CL_delta", units="rad**-1")
    assert cl_delta_elev == pytest.approx(0.5456, abs=1e-4)


def test_extreme_cl():
    """ Tests maximum/minimum cl component with default result cl=f(y) curve"""

    # Clear saved polar results (for wing and htp airfoils)
    clear_polar_results()

    # Research independent input value in .xml file for Openvsp test
    ivc = get_indep_var_comp(list_inputs(ComputeExtremeCL()), __file__, XML_FILE)
    y_vector_wing = np.zeros(SPAN_MESH_POINT)
    cl_vector_wing = np.zeros(SPAN_MESH_POINT)
    y_vector_htp = np.zeros(SPAN_MESH_POINT)
    cl_vector_htp = np.zeros(SPAN_MESH_POINT)
    y_vector_wing[0:39] = [0.05, 0.14, 0.23, 0.32, 0.41, 0.5, 0.59, 0.72, 0.88, 1.04, 1.21,
                           1.38, 1.54, 1.71, 1.88, 2.05, 2.21, 2.38, 2.55, 2.72, 2.89, 3.06,
                           3.22, 3.39, 3.56, 3.72, 3.89, 4.05, 4.21, 4.37, 4.53, 4.69, 4.85,
                           5.01, 5.16, 5.32, 5.47, 5.62, 5.77]
    cl_vector_wing[0:39] = [0.123, 0.123, 0.123, 0.123, 0.122, 0.122, 0.122, 0.122, 0.122,
                            0.124, 0.125, 0.125, 0.126, 0.126, 0.127, 0.128, 0.128, 0.129,
                            0.129, 0.129, 0.129, 0.129, 0.13, 0.13, 0.13, 0.13, 0.129,
                            0.129, 0.128, 0.128, 0.127, 0.127, 0.126, 0.124, 0.122, 0.117,
                            0.111, 0.098, 0.077]
    y_vector_htp[0:24] = [0.04, 0.12, 0.2, 0.28, 0.36, 0.44, 0.52, 0.6, 0.69, 0.77, 0.85,
                          0.93, 1.01, 1.09, 1.17, 1.25, 1.33, 1.41, 1.49, 1.57, 1.65, 1.73,
                          1.81, 1.89]
    cl_vector_htp[0:24] = [0.079, 0.082, 0.083, 0.084, 0.087, 0.088, 0.085, 0.084, 0.087,
                           0.087, 0.088, 0.088, 0.087, 0.087, 0.088, 0.088, 0.087, 0.085,
                           0.083, 0.081, 0.078, 0.072, 0.061, 0.049]
    ivc.add_output("data:aerodynamics:flaps:landing:CL_max", 0.4677)
    ivc.add_output("data:aerodynamics:flaps:takeoff:CL_max", 0.0984)
    ivc.add_output("data:aerodynamics:wing:low_speed:Y_vector", y_vector_wing, units="m")
    ivc.add_output("data:aerodynamics:wing:low_speed:CL_vector", cl_vector_wing)
    ivc.add_output("data:aerodynamics:wing:low_speed:CL0_clean", 0.1243)
    ivc.add_output("data:aerodynamics:horizontal_tail:low_speed:Y_vector", y_vector_htp, units="m")
    ivc.add_output("data:aerodynamics:horizontal_tail:low_speed:CL_vector", cl_vector_htp)
    ivc.add_output("data:aerodynamics:horizontal_tail:low_speed:CL_ref", 0.08245)
    ivc.add_output("data:aerodynamics:horizontal_tail:low_speed:CL_alpha", 0.5137, units="rad**-1")
    ivc.add_output("data:aerodynamics:low_speed:mach", 0.1149)
    ivc.add_output("data:aerodynamics:low_speed:unit_reynolds", 2782216, units="m**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeExtremeCL(), ivc)
    cl_max_clean_wing = problem["data:aerodynamics:wing:low_speed:CL_max_clean"]
    assert cl_max_clean_wing == pytest.approx(1.49, abs=1e-2)
    cl_min_clean_wing = problem["data:aerodynamics:wing:low_speed:CL_min_clean"]
    assert cl_min_clean_wing == pytest.approx(-1.26, abs=1e-2)
    cl_max_takeoff_wing = problem["data:aerodynamics:aircraft:takeoff:CL_max"]
    assert cl_max_takeoff_wing == pytest.approx(1.59, abs=1e-2)
    cl_max_landing_wing = problem["data:aerodynamics:aircraft:landing:CL_max"]
    assert cl_max_landing_wing == pytest.approx(1.96, abs=1e-2)
    cl_max_clean_htp = problem["data:aerodynamics:horizontal_tail:low_speed:CL_max_clean"]
    assert cl_max_clean_htp == pytest.approx(1.38, abs=1e-2)
    cl_min_clean_htp = problem["data:aerodynamics:horizontal_tail:low_speed:CL_min_clean"]
    assert cl_min_clean_htp == pytest.approx(-1.40, abs=1e-2)
    alpha_max_clean_htp = problem["data:aerodynamics:horizontal_tail:low_speed:clean:alpha_aircraft_max"]
    assert alpha_max_clean_htp == pytest.approx(32., abs=1)
    alpha_min_clean_htp = problem["data:aerodynamics:horizontal_tail:low_speed:clean:alpha_aircraft_min"]
    assert alpha_min_clean_htp == pytest.approx(-32., abs=1)


def test_l_d_max():
    """ Tests best lift/drag component """

    # Define independent input value (openVSP)
    ivc = om.IndepVarComp()
    ivc.add_output("data:aerodynamics:wing:cruise:CL0_clean", 0.0906)
    ivc.add_output("data:aerodynamics:wing:cruise:CL_alpha", 4.650, units="rad**-1")
    ivc.add_output("data:aerodynamics:aircraft:cruise:CD0", 0.01603)
    ivc.add_output("data:aerodynamics:wing:cruise:induced_drag_coefficient", 0.0480)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeLDMax(), ivc)
    l_d_max = problem["data:aerodynamics:aircraft:cruise:L_D_max"]
    assert l_d_max == pytest.approx(18.0, abs=1e-1)
    optimal_cl = problem["data:aerodynamics:aircraft:cruise:optimal_CL"]
    assert optimal_cl == pytest.approx(0.5778, abs=1e-4)
    optimal_cd = problem["data:aerodynamics:aircraft:cruise:optimal_CD"]
    assert optimal_cd == pytest.approx(0.0320, abs=1e-4)
    optimal_alpha = problem.get_val("data:aerodynamics:aircraft:cruise:optimal_alpha", units="deg")
    assert optimal_alpha == pytest.approx(6.00, abs=1e-2)


def test_cnbeta():
    """ Tests cn beta fuselage """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeCnBetaFuselage()), __file__, XML_FILE)
    ivc.add_output("data:aerodynamics:cruise:mach", 0.2488)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCnBetaFuselage(), ivc)
    cn_beta_fus = problem["data:aerodynamics:fuselage:cruise:CnBeta"]
    assert cn_beta_fus == pytest.approx(-0.0636, abs=1e-4)


def test_slipstream_openvsp_cruise():

    # Create result temporary directory
    results_folder = _create_tmp_directory()

    # Research independent input value in .xml file
    # noinspection PyTypeChecker
    ivc = get_indep_var_comp(list_inputs(ComputeSlipstreamOpenvsp(
        propulsion_id=ENGINE_WRAPPER,
        result_folder_path=results_folder.name,
    )), __file__, XML_FILE)
    ivc.add_output("data:aerodynamics:wing:cruise:CL0_clean", val=0.1269)
    ivc.add_output("data:aerodynamics:wing:cruise:CL_alpha", val=5.079, units="rad**-1")
    ivc.add_output("data:aerodynamics:wing:low_speed:CL_max_clean", val=1.4465)
    # Run problem and check obtained value(s) is/(are) correct
    # noinspection PyTypeChecker
    problem = run_system(ComputeSlipstreamOpenvsp(propulsion_id=ENGINE_WRAPPER,
                                                  result_folder_path=results_folder.name,
                                                  low_speed_aero=False
                                                  ), ivc)
    y_vector_prop_on = problem.get_val("data:aerodynamics:slipstream:wing:cruise:prop_on:Y_vector", units="m")
    y_result_prop_on = np.array([0.045, 0.136, 0.227, 0.318, 0.408, 0.499, 0.59, 0.716, 0.88,
                                 1.045, 1.21, 1.376, 1.543, 1.71, 1.878, 2.046, 2.214, 2.383,
                                 2.551, 2.719, 2.887, 3.055, 3.222, 3.389, 3.556, 3.721, 3.886,
                                 4.05, 4.212, 4.374, 4.535, 4.694, 4.852, 5.008, 5.163, 5.317,
                                 5.468, 5.618, 5.766, 0., 0., 0., 0., 0., 0.,
                                 0., 0., 0., 0., 0.])
    assert np.max(np.abs(y_vector_prop_on - y_result_prop_on)) <= 1e-2
    cl_vector_prop_on = problem.get_val("data:aerodynamics:slipstream:wing:cruise:prop_on:CL_vector")
    cl_result_prop_on = np.array([1.415, 1.416, 1.414, 1.412, 1.41, 1.406, 1.404, 1.403, 1.414,
                                  1.434, 1.451, 1.465, 1.468, 1.479, 1.491, 1.502, 1.508, 1.517,
                                  1.522, 1.528, 1.525, 1.53, 1.531, 1.536, 1.531, 1.532, 1.528,
                                  1.525, 1.513, 1.51, 1.497, 1.488, 1.463, 1.437, 1.386, 1.322,
                                  1.22, 1.056, 0.864, 0., 0., 0., 0., 0., 0.,
                                  0., 0., 0., 0., 0.])
    assert np.max(np.abs(cl_vector_prop_on - cl_result_prop_on)) <= 1e-2
    ct = problem.get_val("data:aerodynamics:slipstream:wing:cruise:prop_on:CT_ref")
    assert ct == pytest.approx(0.07245, abs=1e-4)
    delta_cl = problem.get_val("data:aerodynamics:slipstream:wing:cruise:prop_on:CL") - \
               problem.get_val("data:aerodynamics:slipstream:wing:cruise:prop_off:CL")
    assert delta_cl == pytest.approx(-0.0003, abs=1e-4)


def test_slipstream_openvsp_low_speed():

    # Create result temporary directory
    results_folder = _create_tmp_directory()

    # Research independent input value in .xml file
    # noinspection PyTypeChecker
    ivc = get_indep_var_comp(list_inputs(ComputeSlipstreamOpenvsp(
        propulsion_id=ENGINE_WRAPPER,
        result_folder_path=results_folder.name,
        low_speed_aero=True,
    )), __file__, XML_FILE)
    ivc.add_output("data:aerodynamics:wing:low_speed:CL0_clean", val=0.1243)
    ivc.add_output("data:aerodynamics:wing:low_speed:CL_alpha", val=4.981, units="rad**-1")
    ivc.add_output("data:aerodynamics:wing:low_speed:CL_max_clean", val=1.4465)
    # Run problem and check obtained value(s) is/(are) correct
    # noinspection PyTypeChecker
    problem = run_system(ComputeSlipstreamOpenvsp(propulsion_id=ENGINE_WRAPPER,
                                                  result_folder_path=results_folder.name,
                                                  low_speed_aero=True
                                                  ), ivc)
    y_vector_prop_on = problem.get_val("data:aerodynamics:slipstream:wing:low_speed:prop_on:Y_vector", units="m")
    y_result_prop_on = np.array([0.045, 0.136, 0.227, 0.318, 0.408, 0.499, 0.59, 0.716, 0.88,
                                 1.045, 1.21, 1.376, 1.543, 1.71, 1.878, 2.046, 2.214, 2.383,
                                 2.551, 2.719, 2.887, 3.055, 3.222, 3.389, 3.556, 3.721, 3.886,
                                 4.05, 4.212, 4.374, 4.535, 4.694, 4.852, 5.008, 5.163, 5.317,
                                 5.468, 5.618, 5.766, 0., 0., 0., 0., 0., 0.,
                                 0., 0., 0., 0., 0.])
    assert np.max(np.abs(y_vector_prop_on - y_result_prop_on)) <= 1e-2
    cl_vector_prop_on = problem.get_val("data:aerodynamics:slipstream:wing:low_speed:prop_on:CL_vector")
    cl_result_prop_on = np.array([1.413, 1.415, 1.413, 1.412, 1.409, 1.405, 1.403, 1.402, 1.414,
                                  1.433, 1.45, 1.464, 1.467, 1.478, 1.491, 1.501, 1.507, 1.517,
                                  1.521, 1.527, 1.524, 1.529, 1.531, 1.535, 1.53, 1.532, 1.527,
                                  1.525, 1.513, 1.509, 1.496, 1.487, 1.463, 1.438, 1.388, 1.325,
                                  1.224, 1.061, 0.873, 0., 0., 0., 0., 0., 0.,
                                  0., 0., 0., 0., 0.])
    assert np.max(np.abs(cl_vector_prop_on - cl_result_prop_on)) <= 1e-2
    ct = problem.get_val("data:aerodynamics:slipstream:wing:low_speed:prop_on:CT_ref")
    assert ct == pytest.approx(0.05694, abs=1e-4)
    delta_cl = problem.get_val("data:aerodynamics:slipstream:wing:low_speed:prop_on:CL") - \
               problem.get_val("data:aerodynamics:slipstream:wing:low_speed:prop_off:CL")
    assert delta_cl == pytest.approx(-0.00062, abs=1e-4)


def test_compute_mach_interpolation_roskam():
    """ Tests computation of the mach interpolation vector using Roskam's approach """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeMachInterpolation()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeMachInterpolation(), ivc)
    cl_alpha_vector = problem["data:aerodynamics:aircraft:mach_interpolation:CL_alpha_vector"]
    cl_alpha_result = np.array([5.50, 5.52, 5.60, 5.73, 5.93, 6.20])
    assert np.max(np.abs(cl_alpha_vector - cl_alpha_result)) <= 1e-2
    mach_vector = problem["data:aerodynamics:aircraft:mach_interpolation:mach_vector"]
    mach_result = np.array([0., 0.08, 0.15, 0.23 , 0.31, 0.39])
    assert np.max(np.abs(mach_vector - mach_result)) <= 1e-2


def test_high_speed_connection():
    """ Tests high speed components connection """

    # Clear saved polar results (for wing and htp airfoils)
    clear_polar_results()

    # load all inputs
    reader = VariableIO(pth.join(pth.dirname(__file__), "data", XML_FILE))
    reader.path_separator = ":"
    input_vars = reader.read().to_ivc()

    # Run problem with VLM
    # noinspection PyTypeChecker
    run_system(AerodynamicsHighSpeed(propulsion_id=ENGINE_WRAPPER, use_openvsp=False), input_vars)

    # Run problem with OPENVSP
    # noinspection PyTypeChecker
    run_system(AerodynamicsHighSpeed(propulsion_id=ENGINE_WRAPPER, use_openvsp=True), input_vars)


def test_low_speed_connection():
    """ Tests low speed components connection """

    # Clear saved polar results (for wing and htp airfoils)
    clear_polar_results()

    # load all inputs
    reader = VariableIO(pth.join(pth.dirname(__file__), "data", XML_FILE))
    reader.path_separator = ":"
    input_vars = reader.read().to_ivc()

    # Run problem with VLM
    # noinspection PyTypeChecker
    run_system(AerodynamicsLowSpeed(propulsion_id=ENGINE_WRAPPER, use_openvsp=False), input_vars)

    # Run problem with OPENVSP
    # noinspection PyTypeChecker
    run_system(AerodynamicsLowSpeed(propulsion_id=ENGINE_WRAPPER, use_openvsp=True), input_vars)


def test_v_n_diagram_vlm():

    # load all inputs
    reader = VariableIO(pth.join(pth.dirname(__file__), "data", XML_FILE))
    reader.path_separator = ":"
    input_vars = reader.read().to_ivc()
    cl_wing_airfoil = np.zeros(POLAR_POINT_COUNT)
    cdp_wing_airfoil = np.zeros(POLAR_POINT_COUNT)
    cl_htp_airfoil = np.zeros(POLAR_POINT_COUNT)
    cdp_htp_airfoil = np.zeros(POLAR_POINT_COUNT)
    cl_wing_airfoil[0:38] = np.array(
        [0.1391, 0.1988, 0.2581, 0.3177, 0.377, 0.4903, 0.5477, 0.6062,
         0.6647, 0.7226, 0.7807, 0.838, 0.8939, 0.9473, 1.1335, 1.1968,
         1.2451, 1.296, 1.3424, 1.4014, 1.4597, 1.5118, 1.5575, 1.6006,
         1.6383, 1.664, 1.6845, 1.7023, 1.7152, 1.7196, 1.7121, 1.6871,
         1.6386, 1.563, 1.4764, 1.3993, 1.3418, 1.2981]
    )
    cdp_wing_airfoil[0:38] = np.array(
        [0.00143, 0.00147, 0.00154, 0.00163, 0.00173, 0.00196, 0.00214,
         0.00235, 0.0026, 0.00287, 0.00317, 0.00349, 0.00385, 0.00424,
         0.00572, 0.00636, 0.00701, 0.00777, 0.00908, 0.00913, 0.00923,
         0.00982, 0.01098, 0.01221, 0.01357, 0.01508, 0.01715, 0.01974,
         0.02318, 0.02804, 0.035, 0.04486, 0.05824, 0.07544, 0.09465,
         0.1133, 0.1299, 0.14507]
    )
    cl_htp_airfoil[0:41] = np.array(
        [-0., 0.0582, 0.117, 0.1751, 0.2333, 0.291, 0.3486,
         0.4064, 0.4641, 0.5216, 0.5789, 0.6356, 0.6923, 0.747,
         0.8027, 0.8632, 0.9254, 0.9935, 1.0611, 1.127, 1.1796,
         1.227, 1.2762, 1.3255, 1.3756, 1.4232, 1.4658, 1.5084,
         1.5413, 1.5655, 1.5848, 1.5975, 1.6002, 1.5894, 1.5613,
         1.5147, 1.4515, 1.3761, 1.2892, 1.1988, 1.1276]
    )
    cdp_htp_airfoil[0:41] = np.array(
        [0.00074, 0.00075, 0.00078, 0.00086, 0.00095, 0.00109, 0.00126,
         0.00145, 0.00167, 0.00191, 0.00218, 0.00249, 0.00283, 0.00324,
         0.00365, 0.00405, 0.00453, 0.00508, 0.00559, 0.00624, 0.00679,
         0.0074, 0.00813, 0.00905, 0.01, 0.01111, 0.0126, 0.01393,
         0.0155, 0.01743, 0.01993, 0.02332, 0.0282, 0.03541, 0.04577,
         0.05938, 0.07576, 0.0944, 0.11556, 0.13878, 0.16068]
    )
    input_vars.add_output("data:aerodynamics:wing:cruise:CL", cl_wing_airfoil)
    input_vars.add_output("data:aerodynamics:wing:cruise:CDp", cdp_wing_airfoil)
    input_vars.add_output("data:aerodynamics:horizontal_tail:cruise:CL", cl_htp_airfoil)
    input_vars.add_output("data:aerodynamics:horizontal_tail:cruise:CDp", cdp_htp_airfoil)
    input_vars.add_output("data:aerodynamics:aircraft:landing:CL_max", 1.949)
    input_vars.add_output("data:aerodynamics:wing:low_speed:CL_max_clean", 1.4818)
    input_vars.add_output("data:aerodynamics:wing:low_speed:CL_min_clean", -1.1614)
    input_vars.add_output("data:weight:aircraft:MTOW", 1633.0, units="kg")
    input_vars.add_output("data:aerodynamics:cruise:mach", 0.2488)
    input_vars.add_output("data:aerodynamics:wing:cruise:induced_drag_coefficient", 0.048)
    input_vars.add_output("data:aerodynamics:aircraft:cruise:CD0", 0.02733)
    input_vars.add_output("data:aerodynamics:aircraft:mach_interpolation:CL_alpha_vector",
                          [5.235, 5.235, 5.297, 5.381, 5.484, 5.606])
    input_vars.add_output("data:aerodynamics:aircraft:mach_interpolation:mach_vector",
                          [0., 0.15, 0.214, 0.275, 0.332, 0.386])

    # Run problem with VLM and check obtained value(s) is/(are) correct
    # noinspection PyTypeChecker
    problem = run_system(ComputeVNvlmNoVH(propulsion_id=ENGINE_WRAPPER, compute_cl_alpha=True), input_vars)
    velocity_vect = np.array(
        [36.206, 40.897, 70.579, 50.421, 0., 0., 75.561,
         75.561, 75.561, 105.556, 105.556, 105.556, 105.556, 95.001,
         83.942, 0., 31.57, 44.647, 56.826]
    )
    load_factor_vect = np.array(
        [1., -1., 3.8, -1.52, 0., 0., -1.52, 3.8,
         -1.52, 3.8, 0., 2.764, -0.764, 0., 0., 0.,
         1., 2., 2.]
    )
    velocity_array = problem.get_val("data:flight_domain:velocity", units="m/s")
    load_factor_array = problem["data:flight_domain:load_factor"]
    assert np.max(np.abs(velocity_vect - velocity_array)) <= 1e-3
    assert np.max(np.abs(load_factor_vect - load_factor_array)) <= 1e-3


def test_v_n_diagram_openvsp():

    # load all inputs
    reader = VariableIO(pth.join(pth.dirname(__file__), "data", XML_FILE))
    reader.path_separator = ":"
    input_vars = reader.read().to_ivc()
    input_vars.add_output("data:aerodynamics:aircraft:landing:CL_max", 1.9)
    input_vars.add_output("data:aerodynamics:wing:low_speed:CL_max_clean", 1.5)
    input_vars.add_output("data:aerodynamics:wing:low_speed:CL_min_clean", -1.5)
    input_vars.add_output("data:aerodynamics:aircraft:mach_interpolation:CL_alpha_vector",
                          [5.51, 5.51, 5.56, 5.63, 5.71, 5.80])
    input_vars.add_output("data:aerodynamics:aircraft:mach_interpolation:mach_vector",
                          [0., 0.15, 0.21, 0.27, 0.33, 0.39])
    input_vars.add_output("data:weight:aircraft:MTOW", 1633.0, units="kg")
    input_vars.add_output("data:aerodynamics:cruise:mach", 0.2488)
    input_vars.add_output("data:aerodynamics:wing:cruise:induced_drag_coefficient", 0.048)
    input_vars.add_output("data:aerodynamics:aircraft:cruise:CD0", 0.02733)

    # Run problem with VLM and check obtained value(s) is/(are) correct
    # noinspection PyTypeChecker
    problem = run_system(ComputeVNopenvspNoVH(propulsion_id=ENGINE_WRAPPER, compute_cl_alpha=True), input_vars)
    velocity_vect = np.array(
        [35.986, 35.986, 70.15, 44.367, 0., 0., 83.942,
         83.942, 83.942, 117.265, 117.265, 117.265, 117.265, 105.538,
         83.942, 0., 31.974, 45.219, 57.554]
    )
    load_factor_vect = np.array(
        [1., -1., 3.8, -1.52, 0., 0., -1.52, 3.874, -1.874, 3.8, 0., 3.05, -1.05, 0., 0., 0., 1., 2., 2.]
    )
    velocity_array = problem.get_val("data:flight_domain:velocity", units="m/s")
    load_factor_array = problem["data:flight_domain:load_factor"]
    assert np.max(np.abs(velocity_vect - velocity_array)) <= 1e-3
    assert np.max(np.abs(load_factor_vect - load_factor_array)) <= 1e-3
