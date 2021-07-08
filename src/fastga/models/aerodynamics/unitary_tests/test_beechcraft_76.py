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
import glob
import shutil
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
    ComputeDeltaHighLift, Compute2DHingeMomentsTail, Compute3DHingeMomentsTail, ComputeMachInterpolation, \
    ComputeCyDeltaRudder, ComputeAirfoilLiftCurveSlope, ComputeClalphaVT
from ..aerodynamics_high_speed import AerodynamicsHighSpeed
from ..aerodynamics_low_speed import AerodynamicsLowSpeed
from ..external.vlm.compute_aero import DEFAULT_WING_AIRFOIL, DEFAULT_HTP_AIRFOIL
from ..constants import SPAN_MESH_POINT, POLAR_POINT_COUNT
from ..components.compute_propeller_aero import ComputePropellerPerformance

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs
from tests.xfoil_exe.get_xfoil import get_xfoil_path

from fastga.models.propulsion.fuel_propulsion.base import AbstractFuelPropulsion
from fastga.models.propulsion.propulsion import IPropulsion

from .dummy_engines import ENGINE_WRAPPER_BE76 as ENGINE_WRAPPER

RESULTS_FOLDER = pth.join(pth.dirname(__file__), "results")
xfoil_path = None if system() == "Windows" else get_xfoil_path()

XML_FILE = "beechcraft_76.xml"

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


def polar_result_transfer():
    # Put saved polar results in a temporary folder to activate Xfoil run and have repeatable results  [need writting
    # permission]

    tmp_folder = _create_tmp_directory()

    files = glob.iglob(pth.join(resources.__path__[0], "*.csv"))

    for file in files:
        if os.path.isfile(file):
            shutil.copy(file, tmp_folder.name)
            try:
                os.remove(file)
            except:
                pass

    return tmp_folder


def polar_result_retrieve(tmp_folder):
    # Retrieve the polar results set aside during the test duration if there are some [need writting permission]

    files = glob.iglob(pth.join(tmp_folder.name, "*.csv"))

    for file in files:
        if os.path.isfile(file):
            try:
                shutil.copy(file, resources.__path__[0])
            except:
                pass

    tmp_folder.cleanup()


def _test_compute_reynolds():
    """ Tests high and low speed reynolds calculation """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeUnitReynolds()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeUnitReynolds(), ivc)
    mach = problem["data:aerodynamics:cruise:mach"]
    assert mach == pytest.approx(0.255041, abs=1e-4)
    reynolds = problem.get_val("data:aerodynamics:cruise:unit_reynolds", units="m**-1")
    assert reynolds == pytest.approx(4745380, abs=1)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeUnitReynolds(low_speed_aero=True)), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeUnitReynolds(low_speed_aero=True), ivc)
    mach = problem["data:aerodynamics:low_speed:mach"]
    assert mach == pytest.approx(0.1179, abs=1e-4)
    reynolds = problem.get_val("data:aerodynamics:low_speed:unit_reynolds", units="m**-1")
    assert reynolds == pytest.approx(2746999, abs=1)


def _test_cd0_high_speed():
    """ Tests drag coefficient @ high speed """

    # Research independent input value in .xml file
    # noinspection PyTypeChecker
    ivc = get_indep_var_comp(list_inputs(Cd0(propulsion_id=ENGINE_WRAPPER)), __file__, XML_FILE)
    ivc.add_output("data:aerodynamics:cruise:mach", 0.2457)
    ivc.add_output("data:aerodynamics:cruise:unit_reynolds", 4571770, units="m**-1")

    # noinspection PyTypeChecker
    problem = run_system(Cd0(propulsion_id=ENGINE_WRAPPER), ivc)
    cd0_wing = problem["data:aerodynamics:wing:cruise:CD0"]
    assert cd0_wing == pytest.approx(0.00503, abs=1e-5)
    cd0_fus = problem["data:aerodynamics:fuselage:cruise:CD0"]
    assert cd0_fus == pytest.approx(0.00490, abs=1e-5)
    cd0_ht = problem["data:aerodynamics:horizontal_tail:cruise:CD0"]
    assert cd0_ht == pytest.approx(0.00123, abs=1e-5)
    cd0_vt = problem["data:aerodynamics:vertical_tail:cruise:CD0"]
    assert cd0_vt == pytest.approx(0.00077, abs=1e-5)
    cd0_nac = problem["data:aerodynamics:nacelles:cruise:CD0"]
    assert cd0_nac == pytest.approx(0.00185, abs=1e-5)
    cd0_lg = problem["data:aerodynamics:landing_gear:cruise:CD0"]
    assert cd0_lg == pytest.approx(0.0, abs=1e-5)
    cd0_other = problem["data:aerodynamics:other:cruise:CD0"]
    assert cd0_other == pytest.approx(0.00187, abs=1e-5)
    cd0_total = 1.25 * (cd0_wing + cd0_fus + cd0_ht + cd0_vt + cd0_nac + cd0_lg + cd0_other)
    assert cd0_total == pytest.approx(0.01958, abs=1e-5)


def _test_cd0_low_speed():
    """ Tests drag coefficient @ low speed """

    # Research independent input value in .xml file
    # noinspection PyTypeChecker
    ivc = get_indep_var_comp(list_inputs(Cd0(propulsion_id=ENGINE_WRAPPER, low_speed_aero=True)), __file__, XML_FILE)
    ivc.add_output("data:aerodynamics:low_speed:mach", 0.1149)  # correction to compensate old version conversion error
    ivc.add_output("data:aerodynamics:low_speed:unit_reynolds", 2613822, units="m**-1")  # correction to ...

    # noinspection PyTypeChecker
    problem = run_system(Cd0(propulsion_id=ENGINE_WRAPPER, low_speed_aero=True), ivc)
    cd0_wing = problem["data:aerodynamics:wing:low_speed:CD0"]
    assert cd0_wing == pytest.approx(0.00552, abs=1e-5)
    cd0_fus = problem["data:aerodynamics:fuselage:low_speed:CD0"]
    assert cd0_fus == pytest.approx(0.00547, abs=1e-5)
    cd0_ht = problem["data:aerodynamics:horizontal_tail:low_speed:CD0"]
    assert cd0_ht == pytest.approx(0.00135, abs=1e-5)
    cd0_vt = problem["data:aerodynamics:vertical_tail:low_speed:CD0"]
    assert cd0_vt == pytest.approx(0.00086, abs=1e-5)
    cd0_nac = problem["data:aerodynamics:nacelles:low_speed:CD0"]
    assert cd0_nac == pytest.approx(0.00202, abs=1e-5)
    cd0_lg = problem["data:aerodynamics:landing_gear:low_speed:CD0"]
    assert cd0_lg == pytest.approx(0.01900, abs=1e-5)
    cd0_other = problem["data:aerodynamics:other:low_speed:CD0"]
    assert cd0_other == pytest.approx(0.00187, abs=1e-5)
    cd0_total = 1.25 * (cd0_wing + cd0_fus + cd0_ht + cd0_vt + cd0_nac + cd0_lg + cd0_other)
    assert cd0_total == pytest.approx(0.04513, abs=1e-5)


def _test_polar():
    """ Tests polar execution (XFOIL) @ high and low speed """

    # Transfer saved polar results to temporary folder
    tmp_folder = polar_result_transfer()

    # Define high-speed parameters (with .xml file and additional inputs)
    ivc = get_indep_var_comp(list_inputs(XfoilPolar()), __file__, XML_FILE)
    ivc.add_output("xfoil:mach", 0.245)
    ivc.add_output("xfoil:reynolds", 4571770 * 1.549)

    # Run problem
    xfoil_comp = XfoilPolar(
        alpha_start=0.0, alpha_end=25.0, iter_limit=20, xfoil_exe_path=xfoil_path
    )
    problem = run_system(xfoil_comp, ivc)

    # Retrieve polar results from temporary folder
    polar_result_retrieve(tmp_folder)

    # Check obtained value(s) is/(are) correct
    cl = problem["xfoil:CL"]
    cdp = problem["xfoil:CDp"]
    cl, cdp = reshape_polar(cl, cdp)
    cdp_1 = np.interp(1.0, cl, cdp)
    assert cdp_1 == pytest.approx(0.0046, abs=1e-4)

    # Transfer saved polar results to temporary folder
    tmp_folder = polar_result_transfer()

    # Define low-speed parameters (with .xml file and additional inputs)
    ivc = get_indep_var_comp(list_inputs(XfoilPolar()), __file__, XML_FILE)
    ivc.add_output("xfoil:mach", 0.1179)
    ivc.add_output("xfoil:reynolds", 2746999 * 1.549)

    # Run problem
    xfoil_comp = XfoilPolar(
        alpha_start=0.0, alpha_end=25.0, iter_limit=20, xfoil_exe_path=xfoil_path
    )
    problem = run_system(xfoil_comp, ivc)

    # Retrieve polar results from temporary folder
    polar_result_retrieve(tmp_folder)

    # Check obtained value(s) is/(are) correct
    cl = problem["xfoil:CL"]
    cdp = problem["xfoil:CDp"]
    cl_max_2d = problem["xfoil:CL_max_2D"]
    assert cl_max_2d == pytest.approx(1.6965, abs=1e-4)
    cl, cdp = reshape_polar(cl, cdp)
    cdp_1 = np.interp(1.0, cl, cdp)
    assert cdp_1 == pytest.approx(0.0049, abs=1e-4)


def _test_airfoil_slope():
    """ Tests polar execution (XFOIL) @ high and low speed """

    # Transfer saved polar results to temporary folder
    tmp_folder = polar_result_transfer()

    # Define high-speed parameters (with .xml file and additional inputs)
    ivc = get_indep_var_comp(list_inputs(ComputeAirfoilLiftCurveSlope(wing_airfoil_file="naca63_415.af",
                                                                      htp_airfoil_file="naca0012.af",
                                                                      vtp_airfoil_file="naca0012.af")),
                             __file__, XML_FILE)
    ivc.add_output('data:aerodynamics:low_speed:unit_reynolds', val=2782216)
    ivc.add_output("data:aerodynamics:low_speed:mach", val=0.1194)

    # Run problem
    problem = run_system(ComputeAirfoilLiftCurveSlope(wing_airfoil_file="naca63_415.af",
                                                      htp_airfoil_file="naca0012.af",
                                                      vtp_airfoil_file="naca0012.af"), ivc)

    # Retrieve polar results from temporary folder
    polar_result_retrieve(tmp_folder)

    # Check obtained value(s) is/(are) correct
    cl_alpha_wing = problem.get_val("data:aerodynamics:wing:airfoil:CL_alpha", units="rad**-1")
    assert cl_alpha_wing == pytest.approx(6.4810, abs=1e-4)
    cl_alpha_htp = problem.get_val("data:aerodynamics:horizontal_tail:airfoil:CL_alpha", units="rad**-1")
    assert cl_alpha_htp == pytest.approx(6.3081, abs=1e-4)
    cl_alpha_vtp = problem.get_val("data:aerodynamics:vertical_tail:airfoil:CL_alpha", units="rad**-1")
    assert cl_alpha_vtp == pytest.approx(6.3081, abs=1e-4)


def _test_vlm_comp_high_speed():
    """ Tests vlm components @ high speed """

    for mach_interpolation in [True, False]:

        # Create result temporary directory
        results_folder = _create_tmp_directory()

        # Transfer saved polar results to temporary folder
        tmp_folder = polar_result_transfer()

        # Research independent input value in .xml file
        # noinspection PyTypeChecker
        ivc = get_indep_var_comp(list_inputs(ComputeAEROvlm()), __file__, XML_FILE)

        # Run problem twice
        start = time.time()
        # noinspection PyTypeChecker
        problem = run_system(ComputeAEROvlm(result_folder_path=results_folder.name,
                                            compute_mach_interpolation=mach_interpolation), ivc)
        stop = time.time()
        duration_1st_run = stop - start
        start = time.time()
        # noinspection PyTypeChecker
        run_system(ComputeAEROvlm(result_folder_path=results_folder.name,
                                  compute_mach_interpolation=mach_interpolation), ivc)
        stop = time.time()
        duration_2nd_run = stop - start

        # Retrieve polar results from temporary folder
        polar_result_retrieve(tmp_folder)

        # Check obtained value(s) is/(are) correct
        cl0_wing = problem["data:aerodynamics:wing:cruise:CL0_clean"]
        assert cl0_wing == pytest.approx(0.1511, abs=1e-4)
        cl_alpha_wing = problem.get_val("data:aerodynamics:wing:cruise:CL_alpha", units="rad**-1")
        assert cl_alpha_wing == pytest.approx(4.832, abs=1e-3)
        cm0 = problem["data:aerodynamics:wing:cruise:CM0_clean"]
        assert cm0 == pytest.approx(-0.0563, abs=1e-4)
        coef_k_wing = problem["data:aerodynamics:wing:cruise:induced_drag_coefficient"]
        assert coef_k_wing == pytest.approx(0.05219, abs=1e-4)
        cl0_htp = problem["data:aerodynamics:horizontal_tail:cruise:CL0"]
        assert cl0_htp == pytest.approx(-0.0122, abs=1e-4)
        cl_alpha_htp = problem.get_val("data:aerodynamics:horizontal_tail:cruise:CL_alpha", units="rad**-1")
        assert cl_alpha_htp == pytest.approx(0.6266, abs=1e-4)
        cl_alpha_htp_isolated = problem.get_val("data:aerodynamics:horizontal_tail:cruise:CL_alpha_isolated",
                                                units="rad**-1")
        assert cl_alpha_htp_isolated == pytest.approx(1.0182, abs=1e-4)
        coef_k_htp = problem["data:aerodynamics:horizontal_tail:cruise:induced_drag_coefficient"]
        assert coef_k_htp == pytest.approx(0.2759, abs=1e-4)
        if mach_interpolation:
            cl_alpha_vector = problem["data:aerodynamics:aircraft:mach_interpolation:CL_alpha_vector"]
            assert cl_alpha_vector == pytest.approx([4.823, 4.823, 4.884, 4.964, 5.064, 5.184], abs=1e-2)
            mach_vector = problem["data:aerodynamics:aircraft:mach_interpolation:mach_vector"]
            assert mach_vector == pytest.approx([0., 0.15, 0.217, 0.28, 0.339, 0.395], abs=1e-2)

        # Run problem 2nd time to check time reduction

        assert (duration_2nd_run / duration_1st_run) <= 0.1

        # Remove existing result files
        results_folder.cleanup()


def _test_vlm_comp_low_speed():
    """ Tests vlm components @ low speed """

    # Create result temporary directory
    results_folder = _create_tmp_directory()

    # Transfer saved polar results to temporary folder
    tmp_folder = polar_result_transfer()

    # Research independent input value in .xml file
    # noinspection PyTypeChecker
    ivc = get_indep_var_comp(list_inputs(ComputeAEROvlm(low_speed_aero=True)), __file__, XML_FILE)

    # Run problem twice
    start = time.time()
    # noinspection PyTypeChecker
    problem = run_system(ComputeAEROvlm(low_speed_aero=True, result_folder_path=results_folder.name), ivc)
    stop = time.time()
    duration_1st_run = stop - start
    start = time.time()
    # noinspection PyTypeChecker
    run_system(ComputeAEROvlm(low_speed_aero=True, result_folder_path=results_folder.name), ivc)
    stop = time.time()
    duration_2nd_run = stop - start

    # Retrieve polar results from temporary folder
    polar_result_retrieve(tmp_folder)

    # Check obtained value(s) is/(are) correct
    cl0_wing = problem["data:aerodynamics:wing:low_speed:CL0_clean"]
    assert cl0_wing == pytest.approx(0.1471, abs=1e-4)
    cl_alpha_wing = problem.get_val("data:aerodynamics:wing:low_speed:CL_alpha", units="rad**-1")
    assert cl_alpha_wing == pytest.approx(4.705, abs=1e-3)
    cm0 = problem["data:aerodynamics:wing:low_speed:CM0_clean"]
    assert cm0 == pytest.approx(-0.0548, abs=1e-4)
    y_vector_wing = np.array(
        [0.09983333, 0.2995, 0.49916667, 0.918, 1.556,
         2.194, 2.832, 3.47, 4.108, 4.746,
         5.14475, 5.30425, 5.46375, 5.62325, 5.78275,
         5.94225, 6.10175]
    )
    cl_vector_wing = np.array(
        [0.14401017, 0.1463924, 0.15226592, 0.16893604, 0.17265623,
         0.17246168, 0.16990552, 0.16492115, 0.15646116, 0.14158877,
         0.11655005, 0.10320515, 0.09218582, 0.08158985, 0.07026192,
         0.05685002, 0.03852811]
    )
    chord_vector_wing = np.array(
        [1.549, 1.549, 1.549, 1.549, 1.549, 1.549, 1.549, 1.549, 1.549, 1.549, 1.549, 1.549, 1.549, 1.549, 1.549,
         1.549, 1.549, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0.]
    )
    y, cl = reshape_curve(problem.get_val("data:aerodynamics:wing:low_speed:Y_vector", "m"),
                          problem["data:aerodynamics:wing:low_speed:CL_vector"])
    chord = problem.get_val("data:aerodynamics:wing:low_speed:chord_vector", "m")
    assert np.max(np.abs(y_vector_wing - y)) <= 1e-3
    assert np.max(np.abs(cl_vector_wing - cl)) <= 1e-3
    assert np.max(np.abs(chord_vector_wing - chord)) <= 1e-3
    coef_k_wing = problem["data:aerodynamics:wing:low_speed:induced_drag_coefficient"]
    assert coef_k_wing == pytest.approx(0.04978358, abs=1e-4)
    cl0_htp = problem["data:aerodynamics:horizontal_tail:low_speed:CL0"]
    assert cl0_htp == pytest.approx(-0.0116, abs=1e-4)
    cl_ref_htp = problem["data:aerodynamics:horizontal_tail:low_speed:CL_ref"]
    assert cl_ref_htp == pytest.approx(0.0966, abs=1e-4)
    y_vector_htp = np.array(
        [0.07492647, 0.22477941, 0.37463235, 0.52448529, 0.67433824,
         0.82419118, 0.97404412, 1.12389706, 1.27375, 1.42360294,
         1.57345588, 1.72330882, 1.87316176, 2.02301471, 2.17286765,
         2.32272059, 2.47257353]
    )
    cl_vector_htp = np.array(
        [0.11302182, 0.11283237, 0.11244797, 0.11185721, 0.11104205,
         0.10997647, 0.10862448, 0.10693711, 0.10484791, 0.10226597,
         0.09906504, 0.09506558, 0.09000346, 0.08347052, 0.07478576,
         0.0626507, 0.04381395]
    )
    y, cl = reshape_curve(problem.get_val("data:aerodynamics:horizontal_tail:low_speed:Y_vector", "m"),
                          problem["data:aerodynamics:horizontal_tail:low_speed:CL_vector"])
    assert np.max(np.abs(y_vector_htp - y)) <= 1e-3
    assert np.max(np.abs(cl_vector_htp - cl)) <= 1e-3
    cl_alpha_htp = problem.get_val("data:aerodynamics:horizontal_tail:low_speed:CL_alpha", units="rad**-1")
    assert cl_alpha_htp == pytest.approx(0.6202, abs=1e-4)
    cl_alpha_htp_isolated = problem.get_val("data:aerodynamics:horizontal_tail:low_speed:CL_alpha_isolated",
                                            units="rad**-1")
    assert cl_alpha_htp_isolated == pytest.approx(0.9915, abs=1e-4)
    coef_k_htp = problem["data:aerodynamics:horizontal_tail:low_speed:induced_drag_coefficient"]
    assert coef_k_htp == pytest.approx(0.2800, abs=1e-4)
    assert (duration_2nd_run / duration_1st_run) <= 0.1

    # Remove existing result files
    results_folder.cleanup()


def _test_openvsp_comp_high_speed():
    """ Tests openvsp components @ high speed """

    for mach_interpolation in [True, False]:

        # Create result temporary directory
        results_folder = _create_tmp_directory()

        # Research independent input value in .xml file
        # noinspection PyTypeChecker
        ivc = get_indep_var_comp(list_inputs(ComputeAEROopenvsp()), __file__, XML_FILE)

        # Run problem twice
        start = time.time()
        # noinspection PyTypeChecker
        problem = run_system(ComputeAEROopenvsp(result_folder_path=results_folder.name,
                                                compute_mach_interpolation=mach_interpolation), ivc)
        stop = time.time()
        duration_1st_run = stop - start
        start = time.time()
        # noinspection PyTypeChecker
        run_system(ComputeAEROopenvsp(result_folder_path=results_folder.name,
                                      compute_mach_interpolation=mach_interpolation), ivc)
        stop = time.time()
        duration_2nd_run = stop - start

        # Check obtained value(s) is/(are) correct
        cl0_wing = problem["data:aerodynamics:wing:cruise:CL0_clean"]
        assert cl0_wing == pytest.approx(0.1171, abs=1e-4)
        cl_alpha_wing = problem.get_val("data:aerodynamics:wing:cruise:CL_alpha", units="rad**-1")
        assert cl_alpha_wing == pytest.approx(4.595, abs=1e-3)
        cm0 = problem["data:aerodynamics:wing:cruise:CM0_clean"]
        assert cm0 == pytest.approx(-0.0265, abs=1e-4)
        coef_k_wing = problem["data:aerodynamics:wing:cruise:induced_drag_coefficient"]
        assert coef_k_wing == pytest.approx(0.0482, abs=1e-4)
        if mach_interpolation:
            cl_alpha_vector = problem["data:aerodynamics:aircraft:mach_interpolation:CL_alpha_vector"]
            assert cl_alpha_vector == pytest.approx([5.20, 5.20, 5.24, 5.30, 5.37, 5.45], abs=1e-2)
            mach_vector = problem["data:aerodynamics:aircraft:mach_interpolation:mach_vector"]
            assert mach_vector == pytest.approx([0., 0.15, 0.21, 0.28, 0.34, 0.39], abs=1e-2)
        cl0_htp = problem["data:aerodynamics:horizontal_tail:cruise:CL0"]
        assert cl0_htp == pytest.approx(-0.0058, abs=1e-4)
        cl_alpha_htp = problem.get_val("data:aerodynamics:horizontal_tail:cruise:CL_alpha", units="rad**-1")
        assert cl_alpha_htp == pytest.approx(0.6826, abs=1e-4)
        cl_alpha_htp_isolated = problem.get_val("data:aerodynamics:horizontal_tail:cruise:CL_alpha_isolated",
                                                units="rad**-1")
        assert cl_alpha_htp_isolated == pytest.approx(1.0387, abs=1e-4)
        coef_k_htp = problem["data:aerodynamics:horizontal_tail:cruise:induced_drag_coefficient"]
        assert coef_k_htp == pytest.approx(0.4605, abs=1e-4)
        assert (duration_2nd_run / duration_1st_run) <= 0.01

        # Remove existing result files
        results_folder.cleanup()


def _test_openvsp_comp_low_speed():
    """ Tests openvsp components @ low speed """

    # Create result temporary directory
    results_folder = _create_tmp_directory()

    # Research independent input value in .xml file
    # noinspection PyTypeChecker
    ivc = get_indep_var_comp(list_inputs(ComputeAEROopenvsp(low_speed_aero=True)), __file__, XML_FILE)

    # Run problem twice
    start = time.time()
    # noinspection PyTypeChecker
    problem = run_system(ComputeAEROopenvsp(low_speed_aero=True, result_folder_path=results_folder.name), ivc)
    stop = time.time()
    duration_1st_run = stop - start
    start = time.time()
    # noinspection PyTypeChecker
    run_system(ComputeAEROopenvsp(low_speed_aero=True, result_folder_path=results_folder.name), ivc)
    stop = time.time()
    duration_2nd_run = stop - start

    # Check obtained value(s) is/(are) correct
    cl0_wing = problem["data:aerodynamics:wing:low_speed:CL0_clean"]
    assert cl0_wing == pytest.approx(0.1147, abs=1e-4)
    cl_alpha_wing = problem.get_val("data:aerodynamics:wing:low_speed:CL_alpha", units="rad**-1")
    assert cl_alpha_wing == pytest.approx(4.509, abs=1e-3)
    cm0 = problem["data:aerodynamics:wing:low_speed:CM0_clean"]
    assert cm0 == pytest.approx(-0.0258, abs=1e-4)
    y_vector_wing = np.array(
        [0.04279, 0.12836, 0.21393, 0.2995, 0.38507, 0.47064, 0.55621,
         0.68649, 0.862, 1.0385, 1.21588, 1.39404, 1.57287, 1.75226,
         1.93212, 2.11231, 2.29274, 2.47328, 2.65384, 2.83429, 3.01452,
         3.19443, 3.37389, 3.5528, 3.73106, 3.90855, 4.08517, 4.26082,
         4.43539, 4.60879, 4.78093, 4.9517, 5.12103, 5.28882, 5.45499,
         5.61947, 5.78218, 5.94305, 6.10201]
    )
    cl_vector_wing = np.array(
        [0.12714757, 0.1275284, 0.12742818, 0.12739811, 0.12730792,
         0.12730792, 0.1272077, 0.12667654, 0.12628568, 0.12660638,
         0.126887, 0.12680682, 0.12623558, 0.12584472, 0.12580463,
         0.12553404, 0.12503295, 0.12456192, 0.12390048, 0.12320897,
         0.12222682, 0.12144512, 0.12058324, 0.11971133, 0.11849869,
         0.11724595, 0.11562241, 0.11403895, 0.11208468, 0.11031081,
         0.10819619, 0.10589116, 0.10301488, 0.10000832, 0.09569891,
         0.09022697, 0.08247003, 0.07037363, 0.05267499]
    )
    y, cl = reshape_curve(problem.get_val("data:aerodynamics:wing:low_speed:Y_vector", "m"),
                          problem["data:aerodynamics:wing:low_speed:CL_vector"])
    assert np.max(np.abs(y_vector_wing - y)) <= 1e-3
    assert np.max(np.abs(cl_vector_wing - cl)) <= 1e-3
    coef_k_wing = problem["data:aerodynamics:wing:low_speed:induced_drag_coefficient"]
    assert coef_k_wing == pytest.approx(0.0482, abs=1e-4)
    cl0_htp = problem["data:aerodynamics:horizontal_tail:low_speed:CL0"]
    assert cl0_htp == pytest.approx(-0.0055, abs=1e-4)
    cl_ref_htp = problem["data:aerodynamics:horizontal_tail:low_speed:CL_ref"]
    assert cl_ref_htp == pytest.approx(0.1124, abs=1e-4)
    y_vector_htp = np.array(
        [0.05307, 0.15922, 0.26536, 0.37151, 0.47766, 0.5838, 0.68995,
         0.79609, 0.90224, 1.00839, 1.11453, 1.22068, 1.32682, 1.43297,
         1.53911, 1.64526, 1.75141, 1.85755, 1.9637, 2.06984, 2.17599,
         2.28214, 2.38828, 2.49443]
    )
    cl_vector_htp = np.array(
        [0.12706265, 0.12950803, 0.12983592, 0.12961425, 0.12981745,
         0.12923324, 0.12780388, 0.12690332, 0.12606048, 0.12506524,
         0.12356892, 0.12197331, 0.11966648, 0.11778684, 0.11551696,
         0.11321936, 0.11022672, 0.10689925, 0.1019831, 0.0968037,
         0.08927591, 0.08074826, 0.06745917, 0.05250057]
    )
    y, cl = reshape_curve(problem.get_val("data:aerodynamics:horizontal_tail:low_speed:Y_vector", "m"),
                          problem["data:aerodynamics:horizontal_tail:low_speed:CL_vector"])
    assert np.max(np.abs(y_vector_htp - y)) <= 1e-3
    assert np.max(np.abs(cl_vector_htp - cl)) <= 1e-3
    cl_alpha_htp = problem.get_val("data:aerodynamics:horizontal_tail:low_speed:CL_alpha", units="rad**-1")
    assert cl_alpha_htp == pytest.approx(0.6760, abs=1e-4)
    cl_alpha_htp_isolated = problem.get_val("data:aerodynamics:horizontal_tail:low_speed:CL_alpha_isolated",
                                            units="rad**-1")
    assert cl_alpha_htp_isolated == pytest.approx(1.0198, abs=1e-4)
    coef_k_htp = problem["data:aerodynamics:horizontal_tail:low_speed:induced_drag_coefficient"]
    assert coef_k_htp == pytest.approx(0.4587, abs=1e-4)
    assert (duration_2nd_run / duration_1st_run) <= 0.1

    # Remove existing result files
    results_folder.cleanup()


def _test_2d_hinge_moment():
    """ Tests tail hinge-moments """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(Compute2DHingeMomentsTail()), __file__, XML_FILE)
    ivc.add_output("data:aerodynamics:horizontal_tail:cruise:CL_alpha", 0.6826, units="rad**-1")
    ivc.add_output("data:aerodynamics:horizontal_tail:airfoil:CL_alpha", 6.3090, units="rad**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(Compute2DHingeMomentsTail(), ivc)
    ch_alpha_2d = problem.get_val("data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_alpha_2D", units="rad**-1")
    assert ch_alpha_2d == pytest.approx(-0.3557, abs=1e-4)
    ch_delta_2d = problem.get_val("data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_delta_2D", units="rad**-1")
    assert ch_delta_2d == pytest.approx(-0.5752, abs=1e-4)


def _test_3d_hinge_moment():
    """ Tests tail hinge-moments """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(Compute3DHingeMomentsTail()), __file__, XML_FILE)
    ivc.add_output("data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_alpha_2D", -0.3339, units="rad**-1")
    ivc.add_output("data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_delta_2D", -0.6358, units="rad**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(Compute3DHingeMomentsTail(), ivc)
    ch_alpha = problem.get_val("data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_alpha", units="rad**-1")
    assert ch_alpha == pytest.approx(-0.2486, abs=1e-4)
    ch_delta = problem.get_val("data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_delta", units="rad**-1")
    assert ch_delta == pytest.approx(-0.6765, abs=1e-4)


def _test_high_lift():
    """ Tests high-lift contribution """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeDeltaHighLift()), __file__, XML_FILE)
    ivc.add_output("data:aerodynamics:low_speed:mach", 0.1149)
    ivc.add_output("data:aerodynamics:wing:low_speed:CL_alpha", 4.569, units="rad**-1")
    ivc.add_output("data:aerodynamics:horizontal_tail:airfoil:CL_alpha", 6.3090, units="rad**-1")
    ivc.add_output("data:aerodynamics:wing:airfoil:CL_alpha", 6.4810, units="rad**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeDeltaHighLift(), ivc)
    delta_cl0_landing = problem["data:aerodynamics:flaps:landing:CL"]
    assert delta_cl0_landing == pytest.approx(0.7145, abs=1e-4)
    delta_clmax_landing = problem["data:aerodynamics:flaps:landing:CL_max"]
    assert delta_clmax_landing == pytest.approx(0.5258, abs=1e-4)
    delta_cm_landing = problem["data:aerodynamics:flaps:landing:CM"]
    assert delta_cm_landing == pytest.approx(-0.0964, abs=1e-4)
    delta_cd_landing = problem["data:aerodynamics:flaps:landing:CD"]
    assert delta_cd_landing == pytest.approx(0.01383, abs=1e-4)
    delta_cl0_takeoff = problem["data:aerodynamics:flaps:takeoff:CL"]
    assert delta_cl0_takeoff == pytest.approx(0.2735, abs=1e-4)
    delta_clmax_takeoff = problem["data:aerodynamics:flaps:takeoff:CL_max"]
    assert delta_clmax_takeoff == pytest.approx(0.1076, abs=1e-4)
    delta_cm_takeoff = problem["data:aerodynamics:flaps:takeoff:CM"]
    assert delta_cm_takeoff == pytest.approx(-0.0369, abs=1e-4)
    delta_cd_takeoff = problem["data:aerodynamics:flaps:takeoff:CD"]
    assert delta_cd_takeoff == pytest.approx(0.00111811, abs=1e-4)
    cl_delta_elev = problem.get_val("data:aerodynamics:elevator:low_speed:CL_delta", units="rad**-1")
    assert cl_delta_elev == pytest.approx(0.5424, abs=1e-4)


def _test_extreme_cl():
    """ Tests maximum/minimum cl component with default result cl=f(y) curve"""

    # Transfer saved polar results to temporary folder
    tmp_folder = polar_result_transfer()

    # Research independent input value in .xml file for Openvsp test
    ivc = get_indep_var_comp(list_inputs(ComputeExtremeCL()), __file__, XML_FILE)
    y_vector_wing = np.zeros(SPAN_MESH_POINT)
    cl_vector_wing = np.zeros(SPAN_MESH_POINT)
    y_vector_htp = np.zeros(SPAN_MESH_POINT)
    cl_vector_htp = np.zeros(SPAN_MESH_POINT)
    y_vector_wing[0:39] = [0.04279, 0.12836, 0.21393, 0.2995, 0.38507, 0.47064, 0.55621,
                           0.68649, 0.862, 1.0385, 1.21588, 1.39404, 1.57287, 1.75226,
                           1.93212, 2.11231, 2.29274, 2.47328, 2.65384, 2.83429, 3.01452,
                           3.19443, 3.37389, 3.5528, 3.73106, 3.90855, 4.08517, 4.26082,
                           4.43539, 4.60879, 4.78093, 4.9517, 5.12103, 5.28882, 5.45499,
                           5.61947, 5.78218, 5.94305, 6.10201]
    cl_vector_wing[0:39] = [0.0989, 0.09908, 0.09901, 0.09898, 0.09892, 0.09888, 0.09887,
                            0.09871, 0.09823, 0.09859, 0.09894, 0.09888, 0.09837, 0.098,
                            0.0979, 0.09763, 0.09716, 0.09671, 0.0961, 0.09545, 0.09454,
                            0.09377, 0.09295, 0.09209, 0.09087, 0.08965, 0.08812, 0.0866,
                            0.08465, 0.08284, 0.08059, 0.07817, 0.07494, 0.07178, 0.06773,
                            0.06279, 0.05602, 0.04639, 0.03265]
    y_vector_htp[0:24] = [0.05307, 0.15922, 0.26536, 0.37151, 0.47766, 0.5838, 0.68995,
                          0.79609, 0.90224, 1.00839, 1.11453, 1.22068, 1.32682, 1.43297,
                          1.53911, 1.64526, 1.75141, 1.85755, 1.9637, 2.06984, 2.17599,
                          2.28214, 2.38828, 2.49443]
    cl_vector_htp[0:24] = [0.12706265, 0.12950803, 0.12983592, 0.12961425, 0.12981745,
                           0.12923324, 0.12780388, 0.12690332, 0.12606048, 0.12506524,
                           0.12356892, 0.12197331, 0.11966648, 0.11778684, 0.11551696,
                           0.11321936, 0.11022672, 0.10689925, 0.1019831, 0.0968037,
                           0.08927591, 0.08074826, 0.06745917, 0.05250057]
    ivc.add_output("data:aerodynamics:flaps:landing:CL_max", 0.5788)
    ivc.add_output("data:aerodynamics:flaps:takeoff:CL_max", 0.1218)
    ivc.add_output("data:aerodynamics:wing:low_speed:Y_vector", y_vector_wing, units="m")
    ivc.add_output("data:aerodynamics:wing:low_speed:CL_vector", cl_vector_wing)
    ivc.add_output("data:aerodynamics:wing:low_speed:CL0_clean", 0.0877)
    ivc.add_output("data:aerodynamics:horizontal_tail:low_speed:Y_vector", y_vector_htp, units="m")
    ivc.add_output("data:aerodynamics:horizontal_tail:low_speed:CL_vector", cl_vector_htp)
    ivc.add_output("data:aerodynamics:horizontal_tail:low_speed:CL_ref", 0.11245)
    ivc.add_output("data:aerodynamics:horizontal_tail:low_speed:CL_alpha", 0.6760, units="rad**-1")
    ivc.add_output("data:aerodynamics:low_speed:mach", 0.1149)
    ivc.add_output("data:aerodynamics:low_speed:unit_reynolds", 2613822, units="m**-1")

    # Run problem
    problem = run_system(ComputeExtremeCL(), ivc)

    # Retrieve polar results from temporary folder
    polar_result_retrieve(tmp_folder)

    # Check obtained value(s) is/(are) correct
    cl_max_clean_wing = problem["data:aerodynamics:wing:low_speed:CL_max_clean"]
    assert cl_max_clean_wing == pytest.approx(1.49, abs=1e-2)
    cl_min_clean_wing = problem["data:aerodynamics:wing:low_speed:CL_min_clean"]
    assert cl_min_clean_wing == pytest.approx(-1.19, abs=1e-2)
    cl_max_takeoff_wing = problem["data:aerodynamics:aircraft:takeoff:CL_max"]
    assert cl_max_takeoff_wing == pytest.approx(1.618, abs=1e-2)
    cl_max_landing_wing = problem["data:aerodynamics:aircraft:landing:CL_max"]
    assert cl_max_landing_wing == pytest.approx(2.07, abs=1e-2)
    cl_max_clean_htp = problem["data:aerodynamics:horizontal_tail:low_speed:CL_max_clean"]
    assert cl_max_clean_htp == pytest.approx(1.36, abs=1e-2)
    cl_min_clean_htp = problem["data:aerodynamics:horizontal_tail:low_speed:CL_min_clean"]
    assert cl_min_clean_htp == pytest.approx(-1.36, abs=1e-2)
    alpha_max_clean_htp = problem["data:aerodynamics:horizontal_tail:low_speed:clean:alpha_aircraft_max"]
    assert alpha_max_clean_htp == pytest.approx(26.64, abs=1e-2)
    alpha_min_clean_htp = problem["data:aerodynamics:horizontal_tail:low_speed:clean:alpha_aircraft_min"]
    assert alpha_min_clean_htp == pytest.approx(-26.62, abs=1e-2)


def _test_l_d_max():
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


def _test_cnbeta():
    """ Tests cn beta fuselage """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeCnBetaFuselage()), __file__, XML_FILE)
    ivc.add_output("data:aerodynamics:cruise:mach", 0.245)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCnBetaFuselage(), ivc)
    cn_beta_fus = problem["data:aerodynamics:fuselage:cruise:CnBeta"]
    assert cn_beta_fus == pytest.approx(-0.0599, abs=1e-4)


def _test_slipstream_openvsp_cruise():

    # Create result temporary directory
    results_folder = _create_tmp_directory()

    # Research independent input value in .xml file
    # noinspection PyTypeChecker
    ivc = get_indep_var_comp(list_inputs(ComputeSlipstreamOpenvsp(
        propulsion_id=ENGINE_WRAPPER,
        result_folder_path=results_folder.name,
    )), __file__, XML_FILE)
    ivc.add_output("data:aerodynamics:wing:cruise:CL0_clean", val=0.1173)
    ivc.add_output("data:aerodynamics:wing:cruise:CL_alpha", val=4.5996, units="rad**-1")
    ivc.add_output("data:aerodynamics:wing:low_speed:CL_max_clean", val=1.4465)
    # Run problem and check obtained value(s) is/(are) correct
    # start = time.time()
    # noinspection PyTypeChecker
    problem = run_system(ComputeSlipstreamOpenvsp(propulsion_id=ENGINE_WRAPPER,
                                                  result_folder_path=results_folder.name,
                                                  low_speed_aero=False
                                                  ), ivc)
    # stop = time.time()
    # duration_1st_run = stop - start
    y_vector_prop_on = problem.get_val("data:aerodynamics:slipstream:wing:cruise:prop_on:Y_vector", units="m")
    y_result_prop_on = np.array([0.04, 0.13, 0.21, 0.3, 0.39, 0.47, 0.56, 0.69, 0.86, 1.04, 1.22,
                                 1.39, 1.57, 1.75, 1.93, 2.11, 2.29, 2.47, 2.65, 2.83, 3.01, 3.19,
                                 3.37, 3.55, 3.73, 3.91, 4.09, 4.26, 4.44, 4.61, 4.78, 4.95, 5.12,
                                 5.29, 5.45, 5.62, 5.78, 5.94, 6.1, 0., 0., 0., 0., 0.,
                                 0., 0., 0., 0., 0., 0.])
    assert np.max(np.abs(y_vector_prop_on - y_result_prop_on)) <= 1e-2
    cl_vector_prop_on = problem.get_val("data:aerodynamics:slipstream:wing:cruise:prop_on:CL_vector")
    cl_result_prop_on = np.array([1.657, 1.655, 1.655, 1.655, 1.654, 1.653, 1.658, 1.649, 1.645,
                                  1.65, 1.659, 1.714, 1.737, 1.752, 1.737, 1.658, 1.627, 1.606,
                                  1.594, 1.571, 1.535, 1.529, 1.518, 1.506, 1.484, 1.465, 1.439,
                                  1.414, 1.378, 1.348, 1.309, 1.271, 1.22, 1.171, 1.1, 1.024,
                                  0.923, 0.793, 0.758, 0., 0., 0., 0., 0., 0.,
                                  0., 0., 0., 0., 0.])
    assert np.max(np.abs(cl_vector_prop_on - cl_result_prop_on)) <= 1e-2
    ct = problem.get_val("data:aerodynamics:slipstream:wing:cruise:prop_on:CT_ref")
    assert ct == pytest.approx(0.0483, abs=1e-4)
    delta_cl = problem.get_val("data:aerodynamics:slipstream:wing:cruise:prop_on:CL") - \
               problem.get_val("data:aerodynamics:slipstream:wing:cruise:prop_off:CL")
    assert delta_cl == pytest.approx(0.00565, abs=1e-4)


def _test_slipstream_openvsp_low_speed():

    # Create result temporary directory
    results_folder = _create_tmp_directory()

    # Research independent input value in .xml file
    # noinspection PyTypeChecker
    ivc = get_indep_var_comp(list_inputs(ComputeSlipstreamOpenvsp(
        propulsion_id=ENGINE_WRAPPER,
        result_folder_path=results_folder.name,
        low_speed_aero=True
    )), __file__, XML_FILE)
    ivc.add_output("data:aerodynamics:wing:low_speed:CL0_clean", val=0.1147)
    ivc.add_output("data:aerodynamics:wing:low_speed:CL_alpha", val=4.509, units="rad**-1")
    ivc.add_output("data:aerodynamics:wing:low_speed:CL_max_clean", val=1.4465)
    # Run problem and check obtained value(s) is/(are) correct
    # start = time.time()
    # noinspection PyTypeChecker
    problem = run_system(ComputeSlipstreamOpenvsp(propulsion_id=ENGINE_WRAPPER,
                                                  result_folder_path=results_folder.name,
                                                  low_speed_aero=True
                                                  ), ivc)
    # stop = time.time()
    # duration_1st_run = stop - start
    y_vector_prop_on = problem.get_val("data:aerodynamics:slipstream:wing:low_speed:prop_on:Y_vector", units="m")
    y_result_prop_on = np.array([0.04, 0.13, 0.21, 0.3, 0.39, 0.47, 0.56, 0.69, 0.86, 1.04, 1.22,
                                 1.39, 1.57, 1.75, 1.93, 2.11, 2.29, 2.47, 2.65, 2.83, 3.01, 3.19,
                                 3.37, 3.55, 3.73, 3.91, 4.09, 4.26, 4.44, 4.61, 4.78, 4.95, 5.12,
                                 5.29, 5.45, 5.62, 5.78, 5.94, 6.1, 0., 0., 0., 0., 0.,
                                 0., 0., 0., 0., 0., 0.])
    assert np.max(np.abs(y_vector_prop_on - y_result_prop_on)) <= 1e-2
    cl_vector_prop_on = problem.get_val("data:aerodynamics:slipstream:wing:low_speed:prop_on:CL_vector")
    cl_result_prop_on = np.array([1.691, 1.653, 1.653, 1.653, 1.653, 1.652, 1.658, 1.648, 1.645,
                                  1.652, 1.663, 1.83, 1.926, 1.96, 1.939, 1.798, 1.751, 1.713,
                                  1.668, 1.617, 1.513, 1.513, 1.505, 1.496, 1.477, 1.459, 1.435,
                                  1.411, 1.375, 1.346, 1.309, 1.272, 1.221, 1.173, 1.103, 1.028,
                                  0.928, 0.799, 0.767, 0., 0., 0., 0., 0., 0.,
                                  0., 0., 0., 0., 0.])
    assert np.max(np.abs(cl_vector_prop_on - cl_result_prop_on)) <= 1e-2
    ct = problem.get_val("data:aerodynamics:slipstream:wing:low_speed:prop_on:CT_ref")
    assert ct == pytest.approx(0.03796, abs=1e-4)
    delta_cl = problem.get_val("data:aerodynamics:slipstream:wing:low_speed:prop_on:CL") - \
               problem.get_val("data:aerodynamics:slipstream:wing:low_speed:prop_off:CL")
    assert delta_cl == pytest.approx(0.02196, abs=1e-4)


def _test_compute_mach_interpolation_roskam():
    """ Tests computation of the mach interpolation vector using Roskam's approach """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeMachInterpolation()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeMachInterpolation(), ivc)
    cl_alpha_vector = problem["data:aerodynamics:aircraft:mach_interpolation:CL_alpha_vector"]
    cl_alpha_result = np.array([5.45, 5.48, 5.55, 5.68, 5.87, 6.14])
    assert np.max(np.abs(cl_alpha_vector - cl_alpha_result)) <= 1e-2
    mach_vector = problem["data:aerodynamics:aircraft:mach_interpolation:mach_vector"]
    mach_result = np.array([0., 0.08, 0.16, 0.24, 0.32, 0.40])
    assert np.max(np.abs(mach_vector - mach_result)) <= 1e-2


def _test_cl_alpha_vt():
    """ Tests Cl alpha vt """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeClalphaVT(low_speed_aero=True)), __file__, XML_FILE)
    ivc.add_output("data:aerodynamics:low_speed:mach", 0.119)
    ivc.add_output("data:aerodynamics:vertical_tail:airfoil:CL_alpha", 6.4038, units="rad**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeClalphaVT(low_speed_aero=True), ivc)
    cl_alpha_vt_ls = problem.get_val("data:aerodynamics:vertical_tail:low_speed:CL_alpha", units="rad**-1")
    assert cl_alpha_vt_ls == pytest.approx(2.92419, abs=1e-4)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeClalphaVT()), __file__, XML_FILE)
    ivc.add_output("data:aerodynamics:cruise:mach", 0.248)
    ivc.add_output("data:aerodynamics:vertical_tail:airfoil:CL_alpha", 6.4038, units="rad**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeClalphaVT(), ivc)
    cl_alpha_vt_cruise = problem.get_val("data:aerodynamics:vertical_tail:cruise:CL_alpha", units="rad**-1")
    assert cl_alpha_vt_cruise == pytest.approx(3.0044, abs=1e-4)


def _test_cy_delta_r():
    """ Tests cy delta of the rudder """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeCyDeltaRudder()), __file__, XML_FILE)
    ivc.add_output("data:aerodynamics:vertical_tail:low_speed:CL_alpha", val=1.94358, units="rad**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCyDeltaRudder(), ivc)
    cy_delta_r = problem.get_val("data:aerodynamics:rudder:low_speed:Cy_delta_r", units="rad**-1")
    assert cy_delta_r == pytest.approx(1.2506, abs=1e-4)


def _test_high_speed_connection():
    """ Tests high speed components connection """

    # load all inputs
    reader = VariableIO(pth.join(pth.dirname(__file__), "data", XML_FILE))
    reader.path_separator = ":"
    input_vars = reader.read().to_ivc()
    input_vars.add_output("data:aerodynamics:vertical_tail:airfoil:CL_alpha", 6.3090, units="rad**-1")
    input_vars.add_output("data:aerodynamics:horizontal_tail:airfoil:CL_alpha", 6.3090, units="rad**-1")

    # Run problem with VLM
    # noinspection PyTypeChecker
    run_system(AerodynamicsHighSpeed(propulsion_id=ENGINE_WRAPPER, use_openvsp=False), input_vars)

    # Run problem with OPENVSP
    # noinspection PyTypeChecker
    run_system(AerodynamicsHighSpeed(propulsion_id=ENGINE_WRAPPER, use_openvsp=True), input_vars)


def _test_low_speed_connection():
    """ Tests low speed components connection """

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


def _test_v_n_diagram_vlm():

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
    input_vars.add_output("data:aerodynamics:aircraft:landing:CL_max", 1.9)
    input_vars.add_output("data:aerodynamics:wing:low_speed:CL_max_clean", 1.5)
    input_vars.add_output("data:aerodynamics:wing:low_speed:CL_min_clean", -1.5)
    input_vars.add_output("data:weight:aircraft:MTOW", 1700.0, units="kg")
    input_vars.add_output("data:aerodynamics:cruise:mach", 0.21)
    input_vars.add_output("data:aerodynamics:wing:cruise:induced_drag_coefficient", 0.048)
    input_vars.add_output("data:aerodynamics:aircraft:cruise:CD0", 0.016)
    input_vars.add_output("data:aerodynamics:aircraft:mach_interpolation:CL_alpha_vector",
                          [4.823, 4.823, 4.884, 4.964, 5.064, 5.184])
    input_vars.add_output("data:aerodynamics:aircraft:mach_interpolation:mach_vector",
                          [0., 0.15, 0.217, 0.28, 0.339, 0.395])

    # Run problem with VLM and check obtained value(s) is/(are) correct
    # noinspection PyTypeChecker
    problem = run_system(ComputeVNvlmNoVH(propulsion_id=ENGINE_WRAPPER, compute_cl_alpha=True), input_vars)
    velocity_vect = np.array(
        [30.782, 30.782, 60.006, 37.951, 0., 0., 74.799,
         74.799, 74.799, 101.345, 101.345, 101.345, 101.345, 91.211,
         74.799, 0., 27.351, 38.68, 49.232]
    )
    load_factor_vect = np.array(
        [1., -1., 3.8, -1.52, 0., 0., -1.52, 3.836,
         -1.836, 3.8, 0., 2.957, -0.957, 0., 0., 0.,
         1., 2., 2.]
    )
    velocity_array = problem.get_val("data:flight_domain:velocity", units="m/s")
    load_factor_array = problem["data:flight_domain:load_factor"]
    assert np.max(np.abs(velocity_vect - velocity_array)) <= 1e-3
    assert np.max(np.abs(load_factor_vect - load_factor_array)) <= 1e-3


def _test_v_n_diagram_openvsp():

    # load all inputs
    reader = VariableIO(pth.join(pth.dirname(__file__), "data", XML_FILE))
    reader.path_separator = ":"
    input_vars = reader.read().to_ivc()
    input_vars.add_output("data:aerodynamics:aircraft:landing:CL_max", 1.9)
    input_vars.add_output("data:aerodynamics:wing:low_speed:CL_max_clean", 1.5)
    input_vars.add_output("data:aerodynamics:wing:low_speed:CL_min_clean", -1.5)
    input_vars.add_output("data:aerodynamics:aircraft:mach_interpolation:CL_alpha_vector",
                          [5.20, 5.20, 5.24, 5.30, 5.37, 5.45])
    input_vars.add_output("data:aerodynamics:aircraft:mach_interpolation:mach_vector",
                          [0., 0.15, 0.21, 0.28, 0.34, 0.39])
    input_vars.add_output("data:weight:aircraft:MTOW", 1700.0, units="kg")
    input_vars.add_output("data:aerodynamics:cruise:mach", 0.21)
    input_vars.add_output("data:aerodynamics:wing:cruise:induced_drag_coefficient", 0.048)
    input_vars.add_output("data:aerodynamics:aircraft:cruise:CD0", 0.016)

    # Run problem with VLM and check obtained value(s) is/(are) correct
    # noinspection PyTypeChecker
    problem = run_system(ComputeVNopenvspNoVH(propulsion_id=ENGINE_WRAPPER, compute_cl_alpha=True), input_vars)
    velocity_vect = np.array(
        [30.782, 30.782, 60.006, 37.951, 0., 0., 74.799, 74.799, 74.799, 101.345, 101.345, 101.345, 101.345, 91.210,
         74.799, 0., 27.350, 38.680, 49.231]
    )
    load_factor_vect = np.array(
        [1., -1., 3.8, -1.52, 0., 0., -1.52, 4.004, -2.004, 3.8, 0., 3.059, -1.059, 0., 0., 0., 1., 2., 2.]
    )
    velocity_array = problem.get_val("data:flight_domain:velocity", units="m/s")
    load_factor_array = problem["data:flight_domain:load_factor"]
    assert np.max(np.abs(velocity_vect - velocity_array)) <= 1e-3
    assert np.max(np.abs(load_factor_vect - load_factor_array)) <= 1e-3


def test_propeller():

    # Transfer saved polar results to temporary folder
    tmp_folder = polar_result_transfer()

    # load all inputs and add missing ones
    reader = VariableIO(pth.join(pth.dirname(__file__), "data", XML_FILE))
    reader.path_separator = ":"
    ivc = reader.read().to_ivc()
    ivc.add_output("data:geometry:propeller:diameter", 2.0 * 0.965, units="m")
    ivc.add_output("data:geometry:propeller:hub_diameter", 2.0 * 0.965 * 0.18, units="m")
    ivc.add_output("data:geometry:propeller:blades_number", 2)
    twist_law = lambda x: 25.0 / (x + 0.75)**1.5 + 46.3917 - 15.0
    radius_ratio = [0.165, 0.3, 0.45, 0.655, 0.835, 0.975, 1.0]
    chord = list(np.array([0.11568421, 0.16431579, 0.16844211, 0.21957895, 0.19231579, 0.11568421, 0.11568421]) * 0.965)
    sweep = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    radius_ratio_vect = np.linspace(radius_ratio[0], radius_ratio[-1], 100)
    chord_vect = np.interp(radius_ratio_vect, radius_ratio, chord)
    sweep_vect = np.interp(radius_ratio_vect, radius_ratio, sweep)
    twist_vect = twist_law(radius_ratio_vect)
    ivc.add_output("data:geometry:propeller:radius_ratio_vect", radius_ratio, shape=7)
    ivc.add_output("data:geometry:propeller:chord_vect", chord, units="m", shape=7)
    ivc.add_output("data:geometry:propeller:twist_vect", twist_law(np.array(radius_ratio)), units="deg", shape=7)
    ivc.add_output("data:geometry:propeller:sweep_vect", sweep, units="deg", shape=7)


    # Run problem
    problem = run_system(ComputePropellerPerformance(), ivc)

    # Retrieve polar results from temporary folder
    polar_result_retrieve(tmp_folder)

    # Check obtained value(s) is/(are) correct
    thrust_SL = np.array(
        [ 165.55463516,  363.74098297,  561.92733079,  760.11367861,
          958.30002643, 1156.48637424, 1354.67272206, 1552.85906988,
          1751.04541769, 1949.23176551, 2147.41811333, 2345.60446114,
          2543.79080896, 2741.97715678, 2940.16350459, 3138.34985241,
          3336.53620023, 3534.72254804, 3732.90889586, 3931.09524368,
          4129.2815915 , 4327.46793931, 4525.65428713, 4723.84063495,
          4922.02698276, 5120.21333058, 5318.3996784 , 5516.58602621,
          5714.77237403, 5912.95872185]
    )
    trust_SL_limit = np.array(
        [3992.47453905, 4354.01018556, 4627.19987747, 4851.0636332,
         5044.6557686 , 5220.57194688, 5390.01724447, 5560.09096074,
         5735.92640037, 5912.95872185]
    )
    speed = np.array(
        [  5.,  15.69362963,  26.38725926,  37.08088889,
           47.77451852,  58.46814815,  69.16177778,  79.85540741,
           90.54903704, 101.24266667]
    )
    efficiency_SL = np.array([[0.10624897, 0.18013462, 0.21651981, 0.22807006, 0.22721517,
        0.22083446, 0.21221014, 0.20288339, 0.19375159, 0.18521768,
        0.17695007, 0.16861719, 0.16072916, 0.15320241, 0.14590992,
        0.13921122, 0.1334496 , 0.12795466, 0.12241574, 0.11577645,
        0.11271665, 0.11271665, 0.11271665, 0.11271665, 0.11271665,
        0.11271665, 0.11271665, 0.11271665, 0.11271665, 0.11271665],
       [0.27467896, 0.4344713 , 0.50019948, 0.52210089, 0.52331264,
        0.51450441, 0.50108722, 0.48583145, 0.47029896, 0.45541008,
        0.44104008, 0.42584508, 0.41088675, 0.3965363 , 0.3823869 ,
        0.36810154, 0.35501908, 0.34370124, 0.33279269, 0.32200437,
        0.31026176, 0.29307527, 0.28733591, 0.28733591, 0.28733591,
        0.28733591, 0.28733591, 0.28733591, 0.28733591, 0.28733591],
       [0.39000222, 0.57044741, 0.63699228, 0.66146452, 0.66549523,
        0.66001213, 0.64930646, 0.63598725, 0.62156858, 0.60764123,
        0.59469412, 0.57995927, 0.56458934, 0.54967715, 0.53514192,
        0.52045092, 0.50471251, 0.49125795, 0.47913675, 0.46718454,
        0.45538279, 0.44289265, 0.42604793, 0.40954706, 0.40954706,
        0.40954706, 0.40954706, 0.40954706, 0.40954706, 0.40954706],
       [0.45696153, 0.63245969, 0.69803342, 0.72446018, 0.73281662,
        0.73178886, 0.72538282, 0.71598956, 0.70509116, 0.69372407,
        0.6836798 , 0.67181609, 0.65838315, 0.64512732, 0.63194556,
        0.61905296, 0.60503622, 0.59093406, 0.57890255, 0.56763984,
        0.55636957, 0.54503321, 0.53306416, 0.51715615, 0.49662567,
        0.49662567, 0.49662567, 0.49662567, 0.49662567, 0.49662567],
       [0.47572647, 0.65142816, 0.71998696, 0.75036676, 0.76306201,
        0.76643601, 0.76440373, 0.75915986, 0.75198267, 0.74433328,
        0.73669419, 0.72755218, 0.71693665, 0.7058908 , 0.69470578,
        0.68367562, 0.67205808, 0.65958055, 0.64809168, 0.63756928,
        0.627485  , 0.61741439, 0.60709627, 0.59588947, 0.58106966,
        0.56259329, 0.56259329, 0.56259329, 0.56259329, 0.56259329],
       [0.46755441, 0.64686877, 0.72044175, 0.7562094 , 0.77394705,
        0.78185907, 0.78396766, 0.78252236, 0.77890522, 0.77465356,
        0.76942822, 0.76247608, 0.7543838 , 0.74564022, 0.7366293 ,
        0.72748983, 0.71776748, 0.70736368, 0.69732461, 0.68787651,
        0.67878305, 0.66980521, 0.66084096, 0.65160745, 0.64117342,
        0.62731614, 0.61203777, 0.61203777, 0.61203777, 0.61203777],
       [0.44964411, 0.62979713, 0.71092595, 0.75259286, 0.77515639,
        0.78705554, 0.79282371, 0.79475707, 0.79457721, 0.7930026 ,
        0.7898828 , 0.78503361, 0.77907539, 0.77244989, 0.76540475,
        0.75804781, 0.75010758, 0.74151402, 0.73300844, 0.72486176,
        0.71688625, 0.70899465, 0.70108463, 0.69306823, 0.68471333,
        0.67504824, 0.66188233, 0.64998888, 0.64998888, 0.64998888],
       [0.41597783, 0.6106596 , 0.69887042, 0.74406057, 0.77069263,
        0.78662226, 0.79581753, 0.80082936, 0.80345225, 0.80415093,
        0.8028508 , 0.79994596, 0.79585717, 0.79108738, 0.78578353,
        0.77999311, 0.77361158, 0.76666696, 0.75959304, 0.75263267,
        0.74580499, 0.73897916, 0.73207307, 0.72509322, 0.71792275,
        0.71028881, 0.70132514, 0.68838938, 0.67973214, 0.67973214],
       [0.39388674, 0.59361286, 0.68254224, 0.73406677, 0.76407991,
        0.78323456, 0.79552201, 0.80329205, 0.80823338, 0.81082741,
        0.81117791, 0.80987937, 0.80751355, 0.80434656, 0.80055104,
        0.79619451, 0.79120453, 0.78566054, 0.77985385, 0.77403874,
        0.76825303, 0.76237137, 0.75641685, 0.7503678 , 0.74418538,
        0.73774801, 0.73069332, 0.72233462, 0.70917405, 0.70352754],
       [0.37986866, 0.57166131, 0.66845157, 0.72246303, 0.75663953,
        0.77839387, 0.79349744, 0.80353619, 0.81049222, 0.81452268,
        0.81642655, 0.81673453, 0.81581215, 0.81398968, 0.81151883,
        0.80843284, 0.80466699, 0.80034478, 0.79568565, 0.79085532,
        0.78600398, 0.78104705, 0.77593328, 0.77067985, 0.76533983,
        0.759812  , 0.75399333, 0.74748603, 0.73970246, 0.72610238]])
    assert np.sum(
        np.abs(thrust_SL - problem.get_val("data:aerodynamics:propeller:sea_level:thrust", units="N"))
        < 1) == np.size(problem["data:aerodynamics:propeller:sea_level:thrust"])
    assert np.sum(
        np.abs(trust_SL_limit - problem.get_val("data:aerodynamics:propeller:sea_level:thrust_limit", units="N"))
        < 1) == np.size(problem["data:aerodynamics:propeller:sea_level:thrust_limit"])
    assert np.sum(
        np.abs(speed - problem.get_val("data:aerodynamics:propeller:sea_level:speed", units="m/s"))
        < 1e-2) == np.size(problem["data:aerodynamics:propeller:sea_level:speed"])
    assert np.sum(
        np.abs(efficiency_SL - problem["data:aerodynamics:propeller:sea_level:efficiency"])
        < 1e-5) == np.size(problem["data:aerodynamics:propeller:sea_level:efficiency"])
    thrust_CL = np.array(
        [ 130.02473826,  286.71392915,  443.40312004,  600.09231092,
          756.78150181,  913.47069269, 1070.15988358, 1226.84907447,
          1383.53826535, 1540.22745624, 1696.91664713, 1853.60583801,
          2010.2950289 , 2166.98421978, 2323.67341067, 2480.36260156,
          2637.05179244, 2793.74098333, 2950.43017421, 3107.1193651 ,
          3263.80855599, 3420.49774687, 3577.18693776, 3733.87612864,
          3890.56531953, 4047.25451042, 4203.9437013 , 4360.63289219,
          4517.32208307, 4674.01127396]
    )
    trust_CL_limit = np.array(
        [3143.48677725, 3428.45974096, 3644.13600906, 3821.24591387,
         3974.64021074, 4114.38712534, 4249.22691158, 4384.69046768,
         4524.95060164, 4674.01127396]
    )
    efficiency_CL = np.array([[0.10608816, 0.18034658, 0.21668281, 0.22807476, 0.22706235,
        0.22055878, 0.21184539, 0.20245902, 0.19328207, 0.1847079 ,
        0.17642882, 0.16809556, 0.16020603, 0.15268613, 0.145397  ,
        0.13871568, 0.13296437, 0.12745844, 0.12187893, 0.11501007,
        0.11261989, 0.11261989, 0.11261989, 0.11261989, 0.11261989,
        0.11261989, 0.11261989, 0.11261989, 0.11261989, 0.11261989],
       [0.27437851, 0.43483937, 0.50046579, 0.52210109, 0.5230512 ,
        0.51403102, 0.50044969, 0.48506752, 0.46943296, 0.45444031,
        0.44001314, 0.42481021, 0.40983547, 0.39547941, 0.38130631,
        0.36702687, 0.35398164, 0.3426632 , 0.33173048, 0.32089245,
        0.30894696, 0.29021217, 0.28715199, 0.28715199, 0.28715199,
        0.28715199, 0.28715199, 0.28715199, 0.28715199, 0.28715199],
       [0.38955417, 0.57085698, 0.63719913, 0.66140144, 0.66521047,
        0.65951903, 0.64865822, 0.63520988, 0.62067866, 0.60662068,
        0.59357506, 0.57882788, 0.56343443, 0.54851317, 0.53395338,
        0.51923439, 0.50350864, 0.49005808, 0.47792195, 0.46594175,
        0.45409059, 0.44139855, 0.42397755, 0.40932274, 0.40932274,
        0.40932274, 0.40932274, 0.40932274, 0.40932274, 0.40932274],
       [0.45561907, 0.63276693, 0.69812978, 0.72431866, 0.73248072,
        0.73128936, 0.72476624, 0.71525397, 0.7042555 , 0.69273891,
        0.68260341, 0.67074194, 0.65728513, 0.64400762, 0.6308024 ,
        0.61789298, 0.60385179, 0.58971891, 0.57769924, 0.56641794,
        0.55510551, 0.54372095, 0.53160721, 0.51513089, 0.49638912,
        0.49638912, 0.49638912, 0.49638912, 0.49638912, 0.49638912],
       [0.47585855, 0.65158176, 0.71994647, 0.75014468, 0.76268261,
        0.76592657, 0.76379348, 0.7584549 , 0.7511925 , 0.74336715,
        0.73568604, 0.72656528, 0.71592465, 0.70486546, 0.6936497 ,
        0.68259358, 0.670996  , 0.65846917, 0.64695721, 0.6364223 ,
        0.62631211, 0.61620639, 0.60583408, 0.59445407, 0.57916424,
        0.56234249, 0.56234249, 0.56234249, 0.56234249, 0.56234249],
       [0.46767189, 0.64593106, 0.71993781, 0.75574295, 0.77343038,
        0.78128505, 0.78332623, 0.7818116 , 0.77808668, 0.7737207 ,
        0.7684629 , 0.76155754, 0.75344859, 0.74469707, 0.7356597 ,
        0.72649847, 0.71678839, 0.70636234, 0.6962931 , 0.68681983,
        0.67770403, 0.66870286, 0.65970997, 0.65041342, 0.63981451,
        0.6254655 , 0.61178354, 0.61178354, 0.61178354, 0.61178354],
       [0.44513886, 0.62874897, 0.71032027, 0.75204099, 0.77458361,
        0.78635328, 0.79210351, 0.79400371, 0.7936974 , 0.79206531,
        0.78898049, 0.78414413, 0.77820458, 0.77156226, 0.7644998 ,
        0.75713663, 0.74921362, 0.74061084, 0.7320752 , 0.72389855,
        0.71589941, 0.70798877, 0.70005254, 0.69200947, 0.68360997,
        0.6737865 , 0.66000601, 0.64973358, 0.64973358, 0.64973358],
       [0.41286626, 0.60976727, 0.69812203, 0.7431451 , 0.76990787,
        0.78586351, 0.7949721 , 0.79993854, 0.80250421, 0.80319149,
        0.80196185, 0.79909761, 0.79500433, 0.79022682, 0.78491469,
        0.77912962, 0.77277299, 0.76583082, 0.75873595, 0.751745  ,
        0.74489843, 0.73805432, 0.73113139, 0.72413379, 0.71693525,
        0.7092516 , 0.70015481, 0.6863676 , 0.67947196, 0.67947196],
       [0.39274883, 0.59105147, 0.68133411, 0.73315527, 0.76300913,
        0.78228744, 0.79447731, 0.80224646, 0.80717386, 0.80985899,
        0.81021727, 0.80899167, 0.80664047, 0.80348132, 0.79968891,
        0.79534928, 0.79039593, 0.7848619 , 0.77904345, 0.77320609,
        0.76740301, 0.76150944, 0.75554648, 0.74948449, 0.74328696,
        0.73682998, 0.7297194 , 0.72125358, 0.70668174, 0.70325717],
       [0.37372212, 0.57015445, 0.6664333 , 0.72116761, 0.75521021,
        0.77720271, 0.792199  , 0.80231108, 0.80932662, 0.81341586,
        0.81541422, 0.81579073, 0.81487394, 0.81308074, 0.81063329,
        0.80757868, 0.80384315, 0.79954793, 0.7948973 , 0.7900533 ,
        0.78518887, 0.78022594, 0.77511   , 0.76985526, 0.76450674,
        0.75896819, 0.75313416, 0.7465912 , 0.73871676, 0.72264884]])
    assert np.sum(
        np.abs(thrust_CL - problem.get_val("data:aerodynamics:propeller:cruise_level:thrust", units="N"))
        < 1) == np.size(problem["data:aerodynamics:propeller:cruise_level:thrust"])
    assert np.sum(
        np.abs(trust_CL_limit - problem.get_val("data:aerodynamics:propeller:cruise_level:thrust_limit", units="N"))
        < 1) == np.size(problem["data:aerodynamics:propeller:cruise_level:thrust_limit"])
    assert np.sum(
        np.abs(speed - problem.get_val("data:aerodynamics:propeller:cruise_level:speed", units="m/s"))
        < 1e-2) == np.size(problem["data:aerodynamics:propeller:cruise_level:speed"])
    assert np.sum(
        np.abs(efficiency_CL - problem["data:aerodynamics:propeller:cruise_level:efficiency"])
        < 1e-5) == np.size(problem["data:aerodynamics:propeller:cruise_level:efficiency"])
