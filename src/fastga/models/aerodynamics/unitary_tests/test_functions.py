"""Test module for aerodynamics groups."""
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

import glob
import logging
import os
import os.path as pth
import shutil
import tempfile
import time
from pathlib import Path
from platform import system
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from fastga.models.aerodynamics.aerodynamics_high_speed import AerodynamicsHighSpeed
from fastga.models.aerodynamics.aerodynamics_low_speed import AerodynamicsLowSpeed
from fastga.models.aerodynamics.components import (
    ComputeAircraftMaxCl,
    ComputeUnitReynolds,
    ComputeLDMax,
    ComputeDeltaHighLift,
    ComputeDeltaElevator,
    Compute2DHingeMomentsTail,
    Compute3DHingeMomentsTail,
    ComputeHingeMomentsTail,
    ComputeMachInterpolation,
    ComputeCyDeltaRudder,
    ComputeAirfoilLiftCurveSlope,
    ComputeVNAndVH,
    ComputeEquilibratedPolar,
    ComputeNonEquilibratedPolar,
    ComputeExtremeCLWing,
    ComputeExtremeCLHtp,
    ComputeEffectiveEfficiencyPropeller,
    ComputeCLAlphaDotAircraft,
    ComputeCLPitchVelocityAircraft,
    ComputeCYBetaAircraft,
    ComputeCyYawRateAircraft,
    ComputeCyRollRateAircraft,
    ComputeClBetaAircraft,
    ComputeClRollRateAircraft,
    ComputeClYawRateAircraft,
    ComputeClDeltaAileron,
    ComputeClDeltaRudder,
    ComputeCMPitchVelocityAircraft,
    ComputeCMAlphaDotAircraft,
    ComputeCnBetaAircraft,
    ComputeCnDeltaAileron,
    ComputeCnDeltaRudder,
    ComputeCnRollRateAircraft,
)
from fastga.models.aerodynamics.components.cd0 import Cd0
from fastga.models.aerodynamics.components.compute_cn_yaw_rate import ComputeCnYawRateAircraft
from fastga.models.aerodynamics.components.compute_equilibrated_polar import FIRST_INVALID_COEFF
from fastga.models.aerodynamics.components.fuselage import (
    ComputeCyBetaFuselage,
    ComputeCnBetaFuselage,
    ComputeCmAlphaFuselage,
)
from fastga.models.aerodynamics.components.ht import (
    DownWashGradientComputation,
    ComputeCLPitchVelocityHorizontalTail,
    ComputeClBetaHorizontalTail,
    ComputeClRollRateHorizontalTail,
    ComputeCMPitchVelocityHorizontalTail,
)
from fastga.models.aerodynamics.components.wing import (
    ComputeCLPitchVelocityWing,
    ComputeCyBetaWing,
    ComputeClBetaWing,
    ComputeClRollRateWing,
    ComputeClYawRateWing,
    ComputeCMPitchVelocityWing,
    ComputeCnRollRateWing,
    ComputeCnYawRateWing,
)
from fastga.models.aerodynamics.components.vt import (
    ComputeClAlphaVerticalTail,
    ComputeCyBetaVerticalTail,
    ComputeClBetaVerticalTail,
    ComputeClRollRateVerticalTail,
    ComputeClYawRateVerticalTail,
    ComputeCnBetaVerticalTail,
    ComputeCnRollRateVerticalTail,
    ComputeCnYawRateVerticalTail,
)
from fastga.models.aerodynamics.external.propeller_code.compute_propeller_aero import (
    ComputePropellerPerformance,
)
from fastga.models.aerodynamics.external.openvsp import ComputeAEROopenvsp
from fastga.models.aerodynamics.external.openvsp.compute_aero_slipstream import (
    ComputeSlipstreamOpenvsp,
)
from fastga.models.aerodynamics.external.vlm import ComputeAEROvlm
from fastga.models.aerodynamics.external.xfoil import resources
from fastga.models.aerodynamics.external.xfoil.xfoil_polar import XfoilPolar
from fastga.models.aerodynamics.load_factor import LoadFactor
from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs
from tests.xfoil_exe.get_xfoil import get_xfoil_path

RESULTS_FOLDER = pth.join(pth.dirname(__file__), "results")
DATA_FOLDER = pth.join(pth.dirname(__file__), "data")
TMP_SAVE_FOLDER = "test_save"
xfoil_path = None if system() == "Windows" else get_xfoil_path()

_LOGGER = logging.getLogger(__name__)


def _create_tmp_directory() -> TemporaryDirectory:
    """Provide temporary directory for calculation!"""
    for tmp_base_path in [None, pth.join(str(Path.home()), ".fast")]:
        if tmp_base_path is not None:
            os.makedirs(tmp_base_path, exist_ok=True)
        tmp_directory = tempfile.TemporaryDirectory(prefix="x", dir=tmp_base_path)
        break

    return tmp_directory


def reshape_curve(y, cl):
    """Reshape data from openvsp/vlm lift curve!"""
    for idx in range(len(y)):
        if np.sum(y[idx : len(y)] == 0) == (len(y) - idx):
            y = y[0:idx]
            cl = cl[0:idx]
            break

    return y, cl


def reshape_polar(cl, cdp):
    """Reshape data from xfoil polar vectors!"""
    for idx in range(len(cl)):
        if np.sum(cl[idx : len(cl)] == 0) == (len(cl) - idx):
            cl = cl[0:idx]
            cdp = cdp[0:idx]
            break
    return cl, cdp


def polar_result_transfer():
    # Put saved polar results in a temporary folder to activate Xfoil run and have repeatable
    # results [need writing permission]

    tmp_folder = _create_tmp_directory()

    files = glob.iglob(pth.join(resources.__path__[0], "*.csv"))

    for file in files:
        if os.path.isfile(file):
            shutil.copy(file, tmp_folder.name)
            # noinspection PyBroadException
            try:
                os.remove(file)
            except:
                _LOGGER.info("Cannot remove %s file!" % file)

    return tmp_folder


def polar_result_retrieve(tmp_folder):
    # Retrieve the polar results set aside during the test duration if there are some [need
    # writing permission]

    files = glob.iglob(pth.join(tmp_folder.name, "*.csv"))

    for file in files:
        if os.path.isfile(file):
            # noinspection PyBroadException
            try:
                shutil.copy(file, resources.__path__[0])
            except (OSError, shutil.SameFileError) as e:
                if isinstance(e, OSError):
                    _LOGGER.info(
                        "Cannot copy %s file to %s! Likely due to permission error"
                        % (file, tmp_folder.name)
                    )
                else:
                    _LOGGER.info(
                        "Cannot copy %s file to %s! Likely because the file already exists in the "
                        "target directory " % (file, tmp_folder.name)
                    )

    tmp_folder.cleanup()


def compute_reynolds(
    XML_FILE: str,
    mach_low_speed: float,
    reynolds_low_speed: float,
    mach_high_speed: float,
    reynolds_high_speed: float,
):
    """Tests high and low speed reynolds calculation!"""
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeUnitReynolds(low_speed_aero=True)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeUnitReynolds(low_speed_aero=True), ivc)
    assert problem["data:aerodynamics:low_speed:mach"] == pytest.approx(mach_low_speed, abs=1e-4)
    assert problem.get_val(
        "data:aerodynamics:low_speed:unit_reynolds", units="m**-1"
    ) == pytest.approx(reynolds_low_speed, abs=1)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeUnitReynolds(low_speed_aero=False)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeUnitReynolds(low_speed_aero=False), ivc)
    assert problem["data:aerodynamics:cruise:mach"] == pytest.approx(mach_high_speed, abs=1e-4)
    assert problem.get_val(
        "data:aerodynamics:cruise:unit_reynolds", units="m**-1"
    ) == pytest.approx(reynolds_high_speed, abs=1)


def cd0_high_speed(
    XML_FILE: str,
    ENGINE_WRAPPER: str,
    cd0_wing: float,
    cd0_fus: float,
    cd0_ht: float,
    cd0_vt: float,
    cd0_nac: float,
    cd0_lg: float,
    cd0_other: float,
    cd0_total: float,
):
    """Tests drag coefficient @ high speed!"""
    # Research independent input value in .xml file
    # noinspection PyTypeChecker
    ivc = get_indep_var_comp(list_inputs(Cd0(propulsion_id=ENGINE_WRAPPER)), __file__, XML_FILE)

    # noinspection PyTypeChecker
    problem = run_system(Cd0(propulsion_id=ENGINE_WRAPPER), ivc)
    assert problem["data:aerodynamics:wing:cruise:CD0"] == pytest.approx(cd0_wing, abs=1e-5)
    assert problem["data:aerodynamics:fuselage:cruise:CD0"] == pytest.approx(cd0_fus, abs=1e-5)
    assert problem["data:aerodynamics:horizontal_tail:cruise:CD0"] == pytest.approx(
        cd0_ht, abs=1e-5
    )
    assert problem["data:aerodynamics:vertical_tail:cruise:CD0"] == pytest.approx(cd0_vt, abs=1e-5)
    assert problem["data:aerodynamics:nacelles:cruise:CD0"] == pytest.approx(cd0_nac, abs=1e-5)
    assert problem["data:aerodynamics:landing_gear:cruise:CD0"] == pytest.approx(cd0_lg, abs=1e-5)
    assert problem["data:aerodynamics:other:cruise:CD0"] == pytest.approx(cd0_other, abs=1e-5)
    cd0_total_cal = 1.25 * (
        problem["data:aerodynamics:wing:cruise:CD0"]
        + problem["data:aerodynamics:fuselage:cruise:CD0"]
        + problem["data:aerodynamics:horizontal_tail:cruise:CD0"]
        + problem["data:aerodynamics:vertical_tail:cruise:CD0"]
        + problem["data:aerodynamics:nacelles:cruise:CD0"]
        + problem["data:aerodynamics:landing_gear:cruise:CD0"]
        + problem["data:aerodynamics:other:cruise:CD0"]
    )
    assert cd0_total_cal == pytest.approx(cd0_total, abs=1e-5)

    # Exclude check on wing, ht and nacelles Cd0 partials as it is computed by fd for now
    problem.check_partials(
        compact_print=True,
        excludes=[
            "data:aerodynamics:wing:*",
            "data:aerodynamics:horizontal_tail:*",
            "data:aerodynamics:nacelles:*",
        ],
    )


def cd0_low_speed(
    XML_FILE: str,
    ENGINE_WRAPPER: str,
    cd0_wing: float,
    cd0_fus: float,
    cd0_ht: float,
    cd0_vt: float,
    cd0_nac: float,
    cd0_lg: float,
    cd0_other: float,
    cd0_total: float,
):
    """Tests drag coefficient @ low speed!"""
    # Research independent input value in .xml file
    # noinspection PyTypeChecker
    ivc = get_indep_var_comp(
        list_inputs(Cd0(propulsion_id=ENGINE_WRAPPER, low_speed_aero=True)), __file__, XML_FILE
    )

    # noinspection PyTypeChecker
    problem = run_system(Cd0(propulsion_id=ENGINE_WRAPPER, low_speed_aero=True), ivc)
    assert problem["data:aerodynamics:wing:low_speed:CD0"] == pytest.approx(cd0_wing, abs=1e-5)
    assert problem["data:aerodynamics:fuselage:low_speed:CD0"] == pytest.approx(cd0_fus, abs=1e-5)
    assert problem["data:aerodynamics:horizontal_tail:low_speed:CD0"] == pytest.approx(
        cd0_ht, abs=1e-5
    )
    assert problem["data:aerodynamics:vertical_tail:low_speed:CD0"] == pytest.approx(
        cd0_vt, abs=1e-5
    )
    assert problem["data:aerodynamics:nacelles:low_speed:CD0"] == pytest.approx(cd0_nac, abs=1e-5)
    assert problem["data:aerodynamics:landing_gear:low_speed:CD0"] == pytest.approx(
        cd0_lg, abs=1e-5
    )
    assert problem["data:aerodynamics:other:low_speed:CD0"] == pytest.approx(cd0_other, abs=1e-5)
    cd0_total_cal = 1.25 * (
        problem["data:aerodynamics:wing:low_speed:CD0"]
        + problem["data:aerodynamics:fuselage:low_speed:CD0"]
        + problem["data:aerodynamics:horizontal_tail:low_speed:CD0"]
        + problem["data:aerodynamics:vertical_tail:low_speed:CD0"]
        + problem["data:aerodynamics:nacelles:low_speed:CD0"]
        + problem["data:aerodynamics:landing_gear:low_speed:CD0"]
        + problem["data:aerodynamics:other:low_speed:CD0"]
    )
    assert cd0_total_cal == pytest.approx(cd0_total, abs=1e-5)

    # Exclude check on wing, ht and nacelles Cd0 partials as it is computed by fd for now
    problem.check_partials(
        compact_print=True,
        excludes=[
            "data:aerodynamics:wing:*",
            "data:aerodynamics:horizontal_tail:*",
            "data:aerodynamics:nacelles:*",
        ],
    )


def polar(
    XML_FILE: str,
    mach_high_speed: float,
    reynolds_high_speed: float,
    mach_low_speed: float,
    reynolds_low_speed: float,
    cdp_1_high_speed: float,
    cl_max_2d: float,
    cdp_1_low_speed: float,
):
    """Tests polar execution (XFOIL) @ high and low speed!"""
    # Transfer saved polar results to temporary folder
    tmp_folder = polar_result_transfer()

    # Define high-speed parameters (with .xml file and additional inputs)
    ivc = get_indep_var_comp(list_inputs(XfoilPolar()), __file__, XML_FILE)
    ivc.add_output("xfoil:mach", mach_high_speed)
    ivc.add_output("xfoil:reynolds", reynolds_high_speed)

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
    assert np.interp(1.0, cl, cdp) == pytest.approx(cdp_1_high_speed, abs=1e-4)

    # Transfer saved polar results to temporary folder
    tmp_folder = polar_result_transfer()

    # Define low-speed parameters (with .xml file and additional inputs)
    ivc = get_indep_var_comp(list_inputs(XfoilPolar()), __file__, XML_FILE)
    ivc.add_output("xfoil:mach", mach_low_speed)
    ivc.add_output("xfoil:reynolds", reynolds_low_speed)

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
    assert problem["xfoil:CL_max_2D"] == pytest.approx(cl_max_2d, abs=1e-4)
    cl, cdp = reshape_polar(cl, cdp)
    assert np.interp(1.0, cl, cdp) == pytest.approx(cdp_1_low_speed, abs=1e-4)


def polar_ext_folder(
    XML_FILE: str,
    mach_high_speed: float,
    reynolds_high_speed: float,
    mach_low_speed: float,
    reynolds_low_speed: float,
    cdp_1_high_speed: float,
    cl_max_2d: float,
    cdp_1_low_speed: float,
):
    """Tests polar execution (XFOIL) @ high and low speed! with the option airfoil_folder_path"""
    # Transfer saved polar results to temporary folder
    tmp_folder = polar_result_transfer()
    shutil.copy(
        pth.join(DATA_FOLDER, "sample_airfoil.af"), pth.join(tmp_folder.name, "sample_airfoil.af")
    )

    # Define high-speed parameters (with .xml file and additional inputs)
    ivc = get_indep_var_comp(list_inputs(XfoilPolar()), __file__, XML_FILE)
    ivc.add_output("xfoil:mach", mach_high_speed)
    ivc.add_output("xfoil:reynolds", reynolds_high_speed)

    # Run problem
    xfoil_comp = XfoilPolar(
        alpha_start=0.0,
        alpha_end=25.0,
        iter_limit=20,
        xfoil_exe_path=xfoil_path,
        airfoil_folder_path=tmp_folder.name,
        airfoil_file="sample_airfoil.af",
    )
    problem = run_system(xfoil_comp, ivc)

    # Retrieve polar results from temporary folder
    polar_result_retrieve(tmp_folder)

    # Check obtained value(s) is/(are) correct
    cl = problem["xfoil:CL"]
    cdp = problem["xfoil:CDp"]
    cl, cdp = reshape_polar(cl, cdp)
    assert np.interp(1.0, cl, cdp) == pytest.approx(cdp_1_high_speed, abs=1e-4)

    # Transfer saved polar results to temporary folder
    tmp_folder = polar_result_transfer()
    shutil.copy(
        pth.join(DATA_FOLDER, "sample_airfoil.af"), pth.join(tmp_folder.name, "sample_airfoil.af")
    )

    # Define low-speed parameters (with .xml file and additional inputs)
    ivc = get_indep_var_comp(list_inputs(XfoilPolar()), __file__, XML_FILE)
    ivc.add_output("xfoil:mach", mach_low_speed)
    ivc.add_output("xfoil:reynolds", reynolds_low_speed)

    # Run problem
    xfoil_comp = XfoilPolar(
        alpha_start=0.0,
        alpha_end=25.0,
        iter_limit=20,
        xfoil_exe_path=xfoil_path,
        airfoil_folder_path=tmp_folder.name,
        airfoil_file="sample_airfoil.af",
    )
    problem = run_system(xfoil_comp, ivc)

    # Retrieve polar results from temporary folder
    polar_result_retrieve(tmp_folder)

    # Check obtained value(s) is/(are) correct
    cl = problem["xfoil:CL"]
    cdp = problem["xfoil:CDp"]
    assert problem["xfoil:CL_max_2D"] == pytest.approx(cl_max_2d, abs=1e-4)
    cl, cdp = reshape_polar(cl, cdp)
    assert np.interp(1.0, cl, cdp) == pytest.approx(cdp_1_low_speed, abs=1e-4)


def airfoil_slope_wt_xfoil(
    XML_FILE: str,
    wing_airfoil_file: str,
    htp_airfoil_file: str,
    vtp_airfoil_file: str,
):
    """Tests polar execution (XFOIL) @ high speed!"""
    # Define high-speed parameters (with .xml file and additional inputs)
    ivc = get_indep_var_comp(
        list_inputs(
            ComputeAirfoilLiftCurveSlope(
                wing_airfoil_file=wing_airfoil_file,
                htp_airfoil_file=htp_airfoil_file,
                vtp_airfoil_file=vtp_airfoil_file,
            )
        ),
        __file__,
        XML_FILE,
    )

    # Run problem
    problem = run_system(
        ComputeAirfoilLiftCurveSlope(
            wing_airfoil_file=wing_airfoil_file,
            htp_airfoil_file=htp_airfoil_file,
            vtp_airfoil_file=vtp_airfoil_file,
        ),
        ivc,
    )

    # Return problem for complementary values check
    return problem


def airfoil_slope_xfoil(
    XML_FILE: str,
    wing_airfoil_file: str,
    htp_airfoil_file: str,
    vtp_airfoil_file: str,
    cl_alpha_wing: float,
    cl_alpha_htp: float,
    cl_alpha_vtp: float,
):
    """Tests polar reading @ high speed!"""
    # Transfer saved polar results to temporary folder
    tmp_folder = polar_result_transfer()

    problem = airfoil_slope_wt_xfoil(
        XML_FILE,
        wing_airfoil_file,
        htp_airfoil_file,
        vtp_airfoil_file,
    )

    # Retrieve polar results from temporary folder
    polar_result_retrieve(tmp_folder)

    # Check obtained value(s) is/(are) correct
    assert problem.get_val(
        "data:aerodynamics:wing:airfoil:CL_alpha", units="rad**-1"
    ) == pytest.approx(cl_alpha_wing, abs=1e-4)
    assert problem.get_val(
        "data:aerodynamics:horizontal_tail:airfoil:CL_alpha", units="rad**-1"
    ) == pytest.approx(cl_alpha_htp, abs=1e-4)
    assert problem.get_val(
        "data:aerodynamics:vertical_tail:airfoil:CL_alpha", units="rad**-1"
    ) == pytest.approx(cl_alpha_vtp, abs=1e-4)


def compute_aero(
    XML_FILE: str,
    use_openvsp: bool,
    mach_interpolation: bool,
    low_speed_aero: bool,
):
    """Compute aero components!"""
    # Create result temporary directory
    results_folder = _create_tmp_directory()

    # Transfer saved polar results to temporary folder
    tmp_folder = polar_result_transfer()

    # Research independent input value in .xml file
    if use_openvsp:
        # noinspection PyTypeChecker
        ivc = get_indep_var_comp(
            list_inputs(ComputeAEROopenvsp(low_speed_aero=low_speed_aero)), __file__, XML_FILE
        )

        # Run problem twice
        start = time.time()
        # noinspection PyTypeChecker
        problem = run_system(
            ComputeAEROopenvsp(
                low_speed_aero=low_speed_aero,
                result_folder_path=results_folder.name,
                compute_mach_interpolation=mach_interpolation,
            ),
            ivc,
        )
        stop = time.time()
        duration_1st_run = stop - start
        start = time.time()
        # noinspection PyTypeChecker
        run_system(
            ComputeAEROopenvsp(
                low_speed_aero=low_speed_aero,
                result_folder_path=results_folder.name,
                compute_mach_interpolation=mach_interpolation,
            ),
            ivc,
        )
        stop = time.time()
    else:
        # noinspection PyTypeChecker
        ivc = get_indep_var_comp(
            list_inputs(ComputeAEROvlm(low_speed_aero=low_speed_aero)), __file__, XML_FILE
        )

        # Run problem twice
        start = time.time()
        # noinspection PyTypeChecker
        problem = run_system(
            ComputeAEROvlm(
                low_speed_aero=low_speed_aero,
                result_folder_path=results_folder.name,
                compute_mach_interpolation=mach_interpolation,
            ),
            ivc,
        )
        stop = time.time()
        duration_1st_run = stop - start
        start = time.time()
        # noinspection PyTypeChecker
        run_system(
            ComputeAEROvlm(
                low_speed_aero=low_speed_aero,
                result_folder_path=results_folder.name,
                compute_mach_interpolation=mach_interpolation,
            ),
            ivc,
        )
        stop = time.time()
    duration_2nd_run = stop - start

    # Retrieve polar results from temporary folder
    polar_result_retrieve(tmp_folder)

    # Remove existing result files
    results_folder.cleanup()

    # Check obtained value(s) is/(are) correct
    assert (duration_2nd_run / duration_1st_run) <= 0.1

    # Return problem for complementary values check
    return problem


def comp_high_speed(
    XML_FILE: str,
    use_openvsp: bool,
    cl0_wing: float,
    cl_ref_wing: float,
    cl_alpha_wing: float,
    cm0: float,
    coeff_k_wing: float,
    cl0_htp: float,
    cl_alpha_htp: float,
    cl_alpha_htp_isolated: float,
    coeff_k_htp: float,
    cl_alpha_vector: np.ndarray,
    mach_vector: np.ndarray,
):
    """Tests components @ high speed!"""
    for mach_interpolation in [True, False]:
        problem = compute_aero(XML_FILE, use_openvsp, mach_interpolation, False)

        # Check obtained value(s) is/(are) correct
        if mach_interpolation:
            assert problem[
                "data:aerodynamics:aircraft:mach_interpolation:CL_alpha_vector"
            ] == pytest.approx(cl_alpha_vector, abs=1e-2)
            assert problem[
                "data:aerodynamics:aircraft:mach_interpolation:mach_vector"
            ] == pytest.approx(mach_vector, abs=1e-2)
        else:
            assert problem["data:aerodynamics:wing:cruise:CL0_clean"] == pytest.approx(
                cl0_wing, abs=1e-4
            )
            assert problem["data:aerodynamics:wing:cruise:CL_ref"] == pytest.approx(
                cl_ref_wing, abs=1e-4
            )
            assert problem.get_val(
                "data:aerodynamics:wing:cruise:CL_alpha", units="rad**-1"
            ) == pytest.approx(cl_alpha_wing, abs=1e-3)
            assert problem["data:aerodynamics:wing:cruise:CM0_clean"] == pytest.approx(
                cm0, abs=1e-4
            )
            assert problem[
                "data:aerodynamics:wing:cruise:induced_drag_coefficient"
            ] == pytest.approx(coeff_k_wing, abs=1e-4)
            assert problem["data:aerodynamics:horizontal_tail:cruise:CL0"] == pytest.approx(
                cl0_htp, abs=1e-4
            )
            assert problem.get_val(
                "data:aerodynamics:horizontal_tail:cruise:CL_alpha", units="rad**-1"
            ) == pytest.approx(cl_alpha_htp, abs=1e-4)
            assert problem.get_val(
                "data:aerodynamics:horizontal_tail:cruise:CL_alpha_isolated", units="rad**-1"
            ) == pytest.approx(cl_alpha_htp_isolated, abs=1e-4)
            assert problem[
                "data:aerodynamics:horizontal_tail:cruise:induced_drag_coefficient"
            ] == pytest.approx(coeff_k_htp, abs=1e-4)


def comp_low_speed(
    XML_FILE: str,
    use_openvsp: bool,
    cl0_wing: float,
    cl_ref_wing: float,
    cl_alpha_wing: float,
    cm0: float,
    coeff_k_wing: float,
    cl0_htp: float,
    cl_alpha_htp: float,
    cl_alpha_htp_isolated: float,
    coeff_k_htp: float,
    y_vector_wing: np.ndarray,
    cl_vector_wing: np.ndarray,
    chord_vector_wing: np.ndarray,
    cl_ref_htp: float,
    y_vector_htp: np.ndarray,
    cl_vector_htp: np.ndarray,
):
    """Tests components @ low speed!"""
    problem = compute_aero(XML_FILE, use_openvsp, False, True)

    # Check obtained value(s) is/(are) correct
    assert problem["data:aerodynamics:wing:low_speed:CL0_clean"] == pytest.approx(
        cl0_wing, abs=1e-4
    )
    assert problem["data:aerodynamics:wing:low_speed:CL_ref"] == pytest.approx(
        cl_ref_wing, abs=1e-4
    )
    assert problem.get_val(
        "data:aerodynamics:wing:low_speed:CL_alpha", units="rad**-1"
    ) == pytest.approx(cl_alpha_wing, abs=1e-3)
    assert problem["data:aerodynamics:wing:low_speed:CM0_clean"] == pytest.approx(cm0, abs=1e-4)
    assert problem["data:aerodynamics:wing:low_speed:induced_drag_coefficient"] == pytest.approx(
        coeff_k_wing, abs=1e-4
    )
    assert problem["data:aerodynamics:horizontal_tail:low_speed:CL0"] == pytest.approx(
        cl0_htp, abs=1e-4
    )
    assert problem.get_val(
        "data:aerodynamics:horizontal_tail:low_speed:CL_alpha", units="rad**-1"
    ) == pytest.approx(cl_alpha_htp, abs=1e-4)
    assert problem.get_val(
        "data:aerodynamics:horizontal_tail:low_speed:CL_alpha_isolated", units="rad**-1"
    ) == pytest.approx(cl_alpha_htp_isolated, abs=1e-4)
    assert problem[
        "data:aerodynamics:horizontal_tail:low_speed:induced_drag_coefficient"
    ] == pytest.approx(coeff_k_htp, abs=1e-4)
    y, cl = reshape_curve(
        problem.get_val("data:aerodynamics:wing:low_speed:Y_vector", "m"),
        problem["data:aerodynamics:wing:low_speed:CL_vector"],
    )
    _, chord = reshape_curve(
        problem.get_val("data:aerodynamics:wing:low_speed:Y_vector", "m"),
        problem.get_val("data:aerodynamics:wing:low_speed:chord_vector", "m"),
    )
    assert np.max(np.abs(y_vector_wing - y)) <= 1e-3
    assert np.max(np.abs(cl_vector_wing - cl)) <= 1e-3
    assert np.max(np.abs(chord_vector_wing - chord)) <= 1e-3
    assert problem["data:aerodynamics:horizontal_tail:low_speed:CL_ref"] == pytest.approx(
        cl_ref_htp, abs=1e-4
    )
    y, cl = reshape_curve(
        problem.get_val("data:aerodynamics:horizontal_tail:low_speed:Y_vector", "m"),
        problem["data:aerodynamics:horizontal_tail:low_speed:CL_vector"],
    )
    assert np.max(np.abs(y_vector_htp - y)) <= 1e-3
    assert np.max(np.abs(cl_vector_htp - cl)) <= 1e-3


def hinge_moment_2d(XML_FILE: str, ch_alpha_2d: float, ch_delta_2d: float):
    """Tests tail hinge-moments"""
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(Compute2DHingeMomentsTail()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(Compute2DHingeMomentsTail(), ivc)
    assert problem.get_val(
        "data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_alpha_2D", units="rad**-1"
    ) == pytest.approx(ch_alpha_2d, abs=1e-4)
    assert problem.get_val(
        "data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_delta_2D", units="rad**-1"
    ) == pytest.approx(ch_delta_2d, abs=1e-4)


def hinge_moment_3d(XML_FILE: str, ch_alpha: float, ch_delta: float):
    """Tests tail hinge-moments!"""
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(Compute3DHingeMomentsTail()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(Compute3DHingeMomentsTail(), ivc)
    assert problem.get_val(
        "data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_alpha", units="rad**-1"
    ) == pytest.approx(ch_alpha, abs=1e-4)
    assert problem.get_val(
        "data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_delta", units="rad**-1"
    ) == pytest.approx(ch_delta, abs=1e-4)


def hinge_moments(XML_FILE: str, ch_alpha: float, ch_delta: float):
    """Tests tail hinge-moments complete computation!"""

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeHingeMomentsTail()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeHingeMomentsTail(), ivc)
    assert problem.get_val(
        "data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_alpha", units="rad**-1"
    ) == pytest.approx(ch_alpha, abs=1e-4)
    assert problem.get_val(
        "data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_delta", units="rad**-1"
    ) == pytest.approx(ch_delta, abs=1e-4)


def elevator(
    XML_FILE: str,
    cl_delta_elev: float,
    cd_delta_elev: float,
):

    ivc = get_indep_var_comp(list_inputs(ComputeDeltaElevator()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeDeltaElevator(), ivc)

    assert problem.get_val(
        "data:aerodynamics:elevator:low_speed:CL_delta", units="rad**-1"
    ) == pytest.approx(cl_delta_elev, abs=1e-4)
    assert problem.get_val(
        "data:aerodynamics:elevator:low_speed:CD_delta", units="rad**-2"
    ) == pytest.approx(cd_delta_elev, abs=1e-4)


def high_lift(
    XML_FILE: str,
    delta_cl0_landing: float,
    delta_cl0_landing_2d: float,
    delta_clmax_landing: float,
    delta_cm_landing: float,
    delta_cm_landing_2d: float,
    delta_cd_landing: float,
    delta_cd_landing_2d: float,
    delta_cl0_takeoff: float,
    delta_cl0_takeoff_2d: float,
    delta_clmax_takeoff: float,
    delta_cm_takeoff: float,
    delta_cm_takeoff_2d: float,
    delta_cd_takeoff: float,
    delta_cd_takeoff_2d: float,
):
    """Tests high-lift contribution!"""
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeDeltaHighLift()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeDeltaHighLift(), ivc)
    assert problem["data:aerodynamics:flaps:landing:CL"] == pytest.approx(
        delta_cl0_landing, abs=1e-4
    )
    assert problem["data:aerodynamics:flaps:landing:CL_2D"] == pytest.approx(
        delta_cl0_landing_2d, abs=1e-4
    )
    assert problem["data:aerodynamics:flaps:landing:CL_max"] == pytest.approx(
        delta_clmax_landing, abs=1e-4
    )
    assert problem["data:aerodynamics:flaps:landing:CM"] == pytest.approx(
        delta_cm_landing, abs=1e-4
    )
    assert problem["data:aerodynamics:flaps:landing:CM_2D"] == pytest.approx(
        delta_cm_landing_2d, abs=1e-4
    )
    assert problem["data:aerodynamics:flaps:landing:CD"] == pytest.approx(
        delta_cd_landing, abs=1e-4
    )
    assert problem["data:aerodynamics:flaps:landing:CD_2D"] == pytest.approx(
        delta_cd_landing_2d, abs=1e-4
    )
    assert problem["data:aerodynamics:flaps:takeoff:CL"] == pytest.approx(
        delta_cl0_takeoff, abs=1e-4
    )
    assert problem["data:aerodynamics:flaps:takeoff:CL_2D"] == pytest.approx(
        delta_cl0_takeoff_2d, abs=1e-4
    )
    assert problem["data:aerodynamics:flaps:takeoff:CL_max"] == pytest.approx(
        delta_clmax_takeoff, abs=1e-4
    )
    assert problem["data:aerodynamics:flaps:takeoff:CM"] == pytest.approx(
        delta_cm_takeoff, abs=1e-4
    )
    assert problem["data:aerodynamics:flaps:takeoff:CM_2D"] == pytest.approx(
        delta_cm_takeoff_2d, abs=1e-4
    )
    assert problem["data:aerodynamics:flaps:takeoff:CD"] == pytest.approx(
        delta_cd_takeoff, abs=1e-4
    )
    assert problem["data:aerodynamics:flaps:takeoff:CD_2D"] == pytest.approx(
        delta_cd_takeoff_2d, abs=1e-4
    )


def wing_extreme_cl_clean(XML_FILE: str, cl_max_clean_wing: float, cl_min_clean_wing: float):
    """Tests maximum minimum lift coefficient for clean wing."""

    # Transfer saved polar results to temporary folder
    tmp_folder = polar_result_transfer()

    # Research independent input value in .xml file for Openvsp test
    ivc = get_indep_var_comp(list_inputs(ComputeExtremeCLWing()), __file__, XML_FILE)

    # Run problem
    problem = run_system(ComputeExtremeCLWing(), ivc)

    # Retrieve polar results from temporary folder
    polar_result_retrieve(tmp_folder)

    assert problem["data:aerodynamics:wing:low_speed:CL_max_clean"] == pytest.approx(
        cl_max_clean_wing, abs=1e-2
    )
    assert problem["data:aerodynamics:wing:low_speed:CL_min_clean"] == pytest.approx(
        cl_min_clean_wing, abs=1e-2
    )


def htp_extreme_cl_clean(
    XML_FILE: str,
    cl_max_clean_htp: float,
    cl_min_clean_htp: float,
    alpha_max_clean_htp: float,
    alpha_min_clean_htp: float,
):
    """Tests maximum minimum lift coefficient for clean htp."""

    # Transfer saved polar results to temporary folder
    tmp_folder = polar_result_transfer()

    # Research independent input value in .xml file for Openvsp test
    ivc = get_indep_var_comp(list_inputs(ComputeExtremeCLHtp()), __file__, XML_FILE)

    # Run problem
    problem = run_system(ComputeExtremeCLHtp(), ivc)

    # Retrieve polar results from temporary folder
    polar_result_retrieve(tmp_folder)

    assert problem["data:aerodynamics:horizontal_tail:low_speed:CL_max_clean"] == pytest.approx(
        cl_max_clean_htp, abs=1e-2
    )
    assert problem["data:aerodynamics:horizontal_tail:low_speed:CL_min_clean"] == pytest.approx(
        cl_min_clean_htp, abs=1e-2
    )
    assert problem.get_val(
        "data:aerodynamics:horizontal_tail:low_speed:clean:alpha_aircraft_max", units="deg"
    ) == pytest.approx(alpha_max_clean_htp, abs=1e-2)
    assert problem.get_val(
        "data:aerodynamics:horizontal_tail:low_speed:clean:alpha_aircraft_min", units="deg"
    ) == pytest.approx(alpha_min_clean_htp, abs=1e-2)


def extreme_cl(
    XML_FILE: str,
    cl_max_takeoff_wing: float,
    cl_max_landing_wing: float,
):
    """Tests maximum/minimum cl component with default result cl=f(y) curve!"""
    # Transfer saved polar results to temporary folder
    tmp_folder = polar_result_transfer()

    # Research independent input value in .xml file for Openvsp test
    ivc = get_indep_var_comp(list_inputs(ComputeAircraftMaxCl()), __file__, XML_FILE)

    # Run problem
    problem = run_system(ComputeAircraftMaxCl(), ivc)

    # Retrieve polar results from temporary folder
    polar_result_retrieve(tmp_folder)

    # Check obtained value(s) is/(are) correct
    assert problem["data:aerodynamics:aircraft:takeoff:CL_max"] == pytest.approx(
        cl_max_takeoff_wing, abs=1e-2
    )
    assert problem["data:aerodynamics:aircraft:landing:CL_max"] == pytest.approx(
        cl_max_landing_wing, abs=1e-2
    )

    problem.check_partials(compact_print=True)


def l_d_max(
    XML_FILE: str, l_d_max_: float, optimal_cl: float, optimal_cd: float, optimal_alpha: float
):
    """Tests best lift/drag component!"""
    # Define independent input value (openVSP)
    ivc = get_indep_var_comp(list_inputs(ComputeLDMax()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeLDMax(), ivc)
    assert problem["data:aerodynamics:aircraft:cruise:L_D_max"] == pytest.approx(l_d_max_, abs=1e-1)
    assert problem["data:aerodynamics:aircraft:cruise:optimal_CL"] == pytest.approx(
        optimal_cl, abs=1e-4
    )
    assert problem["data:aerodynamics:aircraft:cruise:optimal_CD"] == pytest.approx(
        optimal_cd, abs=1e-4
    )
    assert problem.get_val(
        "data:aerodynamics:aircraft:cruise:optimal_alpha", units="deg"
    ) == pytest.approx(optimal_alpha, abs=1e-2)

    problem.check_partials(compact_print=True)


def cnbeta(XML_FILE: str, cn_beta_fus: float):
    """Tests cn beta fuselage"""
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeCnBetaFuselage()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCnBetaFuselage(), ivc)
    assert problem["data:aerodynamics:fuselage:Cn_beta"] == pytest.approx(cn_beta_fus, rel=1e-3)

    problem.check_partials(compact_print=True)


def slipstream_openvsp(
    XML_FILE: str,
    ENGINE_WRAPPER: str,
    low_speed_aero: bool,
):
    # Create result temporary directory
    results_folder = _create_tmp_directory()

    # Research independent input value in .xml file
    # noinspection PyTypeChecker
    ivc = get_indep_var_comp(
        list_inputs(
            ComputeSlipstreamOpenvsp(
                low_speed_aero=low_speed_aero,
                propulsion_id=ENGINE_WRAPPER,
                result_folder_path=results_folder.name,
            )
        ),
        __file__,
        XML_FILE,
    )
    # Run problem and check obtained value(s) is/(are) correct
    # noinspection PyTypeChecker
    problem = run_system(
        ComputeSlipstreamOpenvsp(
            low_speed_aero=low_speed_aero,
            propulsion_id=ENGINE_WRAPPER,
            result_folder_path=results_folder.name,
        ),
        ivc,
    )

    # Return problem for complementary values check
    return problem


def slipstream_openvsp_cruise(
    XML_FILE: str,
    ENGINE_WRAPPER: str,
    y_vector_prop_on: np.ndarray,
    cl_vector_prop_on: np.ndarray,
    ct: float,
    delta_cl: float,
):
    # Compute slipstream @ high speed
    problem = slipstream_openvsp(XML_FILE, ENGINE_WRAPPER, False)

    # Check obtained value(s) is/(are) correct
    y_result_prop_on = problem.get_val(
        "data:aerodynamics:slipstream:wing:cruise:prop_on:Y_vector", units="m"
    )
    assert np.max(np.abs(y_vector_prop_on - y_result_prop_on)) <= 1e-2
    cl_result_prop_on = problem.get_val(
        "data:aerodynamics:slipstream:wing:cruise:prop_on:CL_vector"
    )
    assert np.max(np.abs(cl_vector_prop_on - cl_result_prop_on)) <= 1e-2
    assert problem.get_val(
        "data:aerodynamics:slipstream:wing:cruise:prop_on:CT_ref"
    ) == pytest.approx(ct, abs=1e-4)
    delta_cl_result = problem.get_val(
        "data:aerodynamics:slipstream:wing:cruise:prop_on:CL"
    ) - problem.get_val("data:aerodynamics:slipstream:wing:cruise:prop_off:CL")
    assert delta_cl_result == pytest.approx(delta_cl, abs=1e-4)


def slipstream_openvsp_low_speed(
    XML_FILE: str,
    ENGINE_WRAPPER: str,
    y_vector_prop_on: np.ndarray,
    cl_vector_prop_on: np.ndarray,
    ct: float,
    delta_cl: float,
):
    # Compute slipstream @ high speed
    problem = slipstream_openvsp(XML_FILE, ENGINE_WRAPPER, True)

    # Check obtained value(s) is/(are) correct
    y_result_prop_on = problem.get_val(
        "data:aerodynamics:slipstream:wing:low_speed:prop_on:Y_vector", units="m"
    )
    assert np.max(np.abs(y_vector_prop_on - y_result_prop_on)) <= 1e-2
    cl_result_prop_on = problem.get_val(
        "data:aerodynamics:slipstream:wing:low_speed:prop_on:CL_vector"
    )
    assert np.max(np.abs(cl_vector_prop_on - cl_result_prop_on)) <= 1e-2
    assert problem.get_val(
        "data:aerodynamics:slipstream:wing:low_speed:prop_on:CT_ref"
    ) == pytest.approx(ct, abs=1e-4)
    delta_cl_result = problem.get_val(
        "data:aerodynamics:slipstream:wing:low_speed:prop_on:CL"
    ) - problem.get_val("data:aerodynamics:slipstream:wing:low_speed:prop_off:CL")
    assert delta_cl_result == pytest.approx(delta_cl, abs=1e-4)


def compute_mach_interpolation_roskam(
    XML_FILE: str, cl_alpha_vector: np.ndarray, mach_vector: np.ndarray
):
    """Tests computation of the mach interpolation vector using Roskam's approach!"""
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeMachInterpolation()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeMachInterpolation(), ivc)
    cl_alpha_result = problem["data:aerodynamics:aircraft:mach_interpolation:CL_alpha_vector"]
    assert np.max(np.abs(cl_alpha_vector - cl_alpha_result)) <= 1e-2
    mach_result = problem["data:aerodynamics:aircraft:mach_interpolation:mach_vector"]
    assert np.max(np.abs(mach_vector - mach_result)) <= 1e-2


def cl_alpha_vt(
    XML_FILE: str, cl_alpha_vt_ls: float, k_ar_effective: float, cl_alpha_vt_cruise: float
):
    """Tests Cl alpha vt!"""
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeClAlphaVerticalTail(low_speed_aero=True)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeClAlphaVerticalTail(low_speed_aero=True), ivc)
    assert problem.get_val(
        "data:aerodynamics:vertical_tail:low_speed:CL_alpha", units="rad**-1"
    ) == pytest.approx(cl_alpha_vt_ls, rel=1e-3)
    assert problem.get_val("data:aerodynamics:vertical_tail:k_ar_effective") == pytest.approx(
        k_ar_effective, rel=1e-3
    )

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeClAlphaVerticalTail()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeClAlphaVerticalTail(), ivc)
    assert problem.get_val(
        "data:aerodynamics:vertical_tail:cruise:CL_alpha", units="rad**-1"
    ) == pytest.approx(cl_alpha_vt_cruise, rel=1e-3)


def cy_delta_r(XML_FILE: str, cy_delta_r_: float, cy_delta_r_cruise):
    """Tests cy delta of the rudder!"""
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeCyDeltaRudder(low_speed_aero=True)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCyDeltaRudder(low_speed_aero=True), ivc)
    assert problem.get_val(
        "data:aerodynamics:rudder:low_speed:Cy_delta_r", units="rad**-1"
    ) == pytest.approx(cy_delta_r_, abs=1e-4)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeCyDeltaRudder()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCyDeltaRudder(), ivc)
    assert problem.get_val(
        "data:aerodynamics:rudder:cruise:Cy_delta_r", units="rad**-1"
    ) == pytest.approx(cy_delta_r_cruise, abs=1e-4)


def effective_efficiency(
    XML_FILE: str, effective_efficiency_low_speed: float, effective_efficiency_cruise: float
):
    """Tests effective efficiency of the propeller!"""
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeEffectiveEfficiencyPropeller(low_speed_aero=True)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeEffectiveEfficiencyPropeller(low_speed_aero=True), ivc)
    assert (
        problem.get_val(
            "data:aerodynamics:propeller:installation_effect:effective_efficiency:low_speed",
        )
        == pytest.approx(effective_efficiency_low_speed, abs=1e-4)
    )

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeEffectiveEfficiencyPropeller()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeEffectiveEfficiencyPropeller(), ivc)
    assert (
        problem.get_val(
            "data:aerodynamics:propeller:installation_effect:effective_efficiency:cruise",
        )
        == pytest.approx(effective_efficiency_cruise, abs=1e-4)
    )


def cm_alpha_fus(XML_FILE: str, cm_alpha_fus_: float):
    """Tests cm alpha of the fuselage"""
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeCmAlphaFuselage()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCmAlphaFuselage(), ivc)
    assert problem.get_val("data:aerodynamics:fuselage:cm_alpha", units="rad**-1") == pytest.approx(
        cm_alpha_fus_, abs=1e-4
    )


def high_speed_connection(XML_FILE: str, ENGINE_WRAPPER: str, use_openvsp: bool):
    """Tests high speed components connection!"""
    # Transfer saved polar results to temporary folder
    tmp_folder = polar_result_transfer()

    # load all inputs
    ivc = get_indep_var_comp(
        list_inputs(AerodynamicsHighSpeed(propulsion_id=ENGINE_WRAPPER, use_openvsp=use_openvsp)),
        __file__,
        XML_FILE,
    )

    # noinspection PyTypeChecker
    run_system(AerodynamicsHighSpeed(propulsion_id=ENGINE_WRAPPER, use_openvsp=use_openvsp), ivc)

    # Retrieve polar results from temporary folder
    polar_result_retrieve(tmp_folder)


def low_speed_connection(XML_FILE: str, ENGINE_WRAPPER: str, use_openvsp: bool):
    """Tests low speed components connection!"""
    # Transfer saved polar results to temporary folder
    tmp_folder = polar_result_transfer()

    # load all inputs
    ivc = get_indep_var_comp(
        list_inputs(AerodynamicsLowSpeed(propulsion_id=ENGINE_WRAPPER, use_openvsp=use_openvsp)),
        __file__,
        XML_FILE,
    )

    # noinspection PyTypeChecker
    run_system(AerodynamicsLowSpeed(propulsion_id=ENGINE_WRAPPER, use_openvsp=use_openvsp), ivc)

    # Retrieve polar results from temporary folder
    polar_result_retrieve(tmp_folder)


def v_n_diagram(
    XML_FILE: str, ENGINE_WRAPPER: str, velocity_vect: np.ndarray, load_factor_vect: np.ndarray
):
    # load all inputs
    ivc = get_indep_var_comp(
        list_inputs(ComputeVNAndVH(propulsion_id=ENGINE_WRAPPER)), __file__, XML_FILE
    )
    # Run problem with VLM and check obtained value(s) is/(are) correct
    # noinspection PyTypeChecker
    problem = run_system(ComputeVNAndVH(propulsion_id=ENGINE_WRAPPER), ivc)
    assert (
        np.max(
            np.abs(
                velocity_vect
                - problem.get_val(
                    "data:mission:sizing:cs23:flight_domain:mtow:velocity", units="m/s"
                )
            )
        )
        <= 1e-3
    )
    assert (
        np.max(
            np.abs(
                load_factor_vect
                - problem["data:mission:sizing:cs23:flight_domain:mtow:load_factor"]
            )
        )
        <= 1e-3
    )


def load_factor(
    XML_FILE: str,
    ENGINE_WRAPPER: str,
    load_factor_ultimate: float,
    load_factor_ultimate_mtow: float,
    load_factor_ultimate_mzfw: float,
    vh: float,
    va: float,
    vc: float,
    vd: float,
):
    # load all inputs
    ivc = get_indep_var_comp(
        list_inputs(LoadFactor(propulsion_id=ENGINE_WRAPPER)), __file__, XML_FILE
    )

    problem = run_system(LoadFactor(propulsion_id=ENGINE_WRAPPER), ivc)

    assert problem.get_val(
        "data:mission:sizing:cs23:sizing_factor:ultimate_aircraft"
    ) == pytest.approx(load_factor_ultimate, abs=1e-1)
    assert (
        max(
            problem.get_val("data:mission:sizing:cs23:sizing_factor:ultimate_mtow:positive"),
            problem.get_val("data:mission:sizing:cs23:sizing_factor:ultimate_mtow:negative"),
        )
        == pytest.approx(load_factor_ultimate_mtow, abs=1e-1)
    )
    assert (
        max(
            problem.get_val("data:mission:sizing:cs23:sizing_factor:ultimate_mzfw:positive"),
            problem.get_val("data:mission:sizing:cs23:sizing_factor:ultimate_mzfw:negative"),
        )
        == pytest.approx(load_factor_ultimate_mzfw, abs=1e-1)
    )
    assert problem.get_val("data:TLAR:v_max_sl", units="m/s") == pytest.approx(vh, abs=1e-2)
    assert problem.get_val(
        "data:mission:sizing:cs23:characteristic_speed:va", units="m/s"
    ) == pytest.approx(va, abs=1e-2)
    assert problem.get_val(
        "data:mission:sizing:cs23:characteristic_speed:vc", units="m/s"
    ) == pytest.approx(vc, abs=1e-2)
    assert problem.get_val(
        "data:mission:sizing:cs23:characteristic_speed:vd", units="m/s"
    ) == pytest.approx(vd, abs=1e-2)


def propeller(
    XML_FILE: str,
    thrust_SL: np.ndarray,
    thrust_SL_limit: np.ndarray,
    efficiency_SL: np.ndarray,
    thrust_CL: np.ndarray,
    thrust_CL_limit: np.ndarray,
    efficiency_CL: np.ndarray,
    speed: np.ndarray,
):
    # Transfer saved polar results to temporary folder
    tmp_folder = polar_result_transfer()

    # load all inputs and add missing ones
    ivc = get_indep_var_comp(
        list_inputs(
            ComputePropellerPerformance(
                sections_profile_name_list=["naca4430"],
                sections_profile_position_list=[0],
                elements_number=3,
            )
        ),
        __file__,
        XML_FILE,
    )

    # Run problem
    problem = run_system(
        ComputePropellerPerformance(
            sections_profile_name_list=["naca4430"],
            sections_profile_position_list=[0],
            elements_number=3,
        ),
        ivc,
    )

    # Retrieve polar results from temporary folder
    polar_result_retrieve(tmp_folder)

    # Check obtained value(s) is/(are) correct
    assert (
        np.sum(
            np.abs(
                thrust_SL
                - problem.get_val("data:aerodynamics:propeller:sea_level:thrust", units="N")
            )
        )
        < 1
    )
    assert (
        np.sum(
            np.abs(
                thrust_SL_limit
                - problem.get_val("data:aerodynamics:propeller:sea_level:thrust_limit", units="N")
            )
        )
        < 1
    )
    assert (
        np.sum(
            np.abs(
                speed - problem.get_val("data:aerodynamics:propeller:sea_level:speed", units="m/s")
            )
        )
        < 1e-2
    )
    assert (
        np.sum(np.abs(efficiency_SL - problem["data:aerodynamics:propeller:sea_level:efficiency"]))
        < 1e-5
    )
    assert (
        np.sum(
            np.abs(
                thrust_CL
                - problem.get_val("data:aerodynamics:propeller:cruise_level:thrust", units="N")
            )
        )
        < 1
    )
    assert (
        np.sum(
            np.abs(
                thrust_CL_limit
                - problem.get_val(
                    "data:aerodynamics:propeller:cruise_level:thrust_limit", units="N"
                )
            )
        )
        < 1
    )
    assert (
        np.sum(
            np.abs(
                speed
                - problem.get_val("data:aerodynamics:propeller:cruise_level:speed", units="m/s")
            )
        )
        < 1e-2
    )
    assert (
        np.sum(
            np.abs(efficiency_CL - problem["data:aerodynamics:propeller:cruise_level:efficiency"])
        )
        < 1e-5
    )


def non_equilibrated_cl_cd_polar(
    XML_FILE: str,
    cl_polar_ls_: np.ndarray,
    cd_polar_ls_: np.ndarray,
    cl_polar_cruise_: np.ndarray,
    cd_polar_cruise_: np.ndarray,
):
    """Tests non-equilibrated cl/cd polar of the aircraft"""
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeNonEquilibratedPolar(low_speed_aero=True)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeNonEquilibratedPolar(low_speed_aero=True), ivc)
    assert problem.get_val("data:aerodynamics:aircraft:low_speed:CD")[::10] == pytest.approx(
        cd_polar_ls_, abs=1e-4
    )
    assert problem.get_val("data:aerodynamics:aircraft:low_speed:CL")[::10] == pytest.approx(
        cl_polar_ls_, abs=1e-2
    )

    ivc = get_indep_var_comp(
        list_inputs(ComputeNonEquilibratedPolar(low_speed_aero=False)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeNonEquilibratedPolar(low_speed_aero=False), ivc)
    assert problem.get_val("data:aerodynamics:aircraft:cruise:CD")[::10] == pytest.approx(
        cd_polar_cruise_, abs=1e-4
    )
    assert problem.get_val("data:aerodynamics:aircraft:cruise:CL")[::10] == pytest.approx(
        cl_polar_cruise_, abs=1e-2
    )


def equilibrated_cl_cd_polar(
    XML_FILE: str,
    cl_polar_ls_: np.ndarray,
    cd_polar_ls_: np.ndarray,
    cl_polar_cruise_: np.ndarray,
    cd_polar_cruise_: np.ndarray,
):
    """Tests equilibrated cl/cd polar of the aircraft"""
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeEquilibratedPolar(low_speed_aero=True, cg_ratio=0.5)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeEquilibratedPolar(low_speed_aero=True, cg_ratio=0.5), ivc)
    polar_cd = np.array(problem.get_val("data:aerodynamics:aircraft:low_speed:equilibrated:CD"))
    valid_polar_cd = polar_cd[np.where(polar_cd < FIRST_INVALID_COEFF)[0]]
    assert list(valid_polar_cd)[::10] == pytest.approx(cd_polar_ls_, abs=1e-4)
    polar_cl = np.array(problem.get_val("data:aerodynamics:aircraft:low_speed:equilibrated:CL"))
    valid_polar_cl = polar_cl[np.where(polar_cl < FIRST_INVALID_COEFF)[0]]
    assert list(valid_polar_cl)[::10] == pytest.approx(cl_polar_ls_, abs=1e-2)

    ivc = get_indep_var_comp(
        list_inputs(ComputeEquilibratedPolar(low_speed_aero=False, cg_ratio=0.5)),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeEquilibratedPolar(low_speed_aero=False, cg_ratio=0.5), ivc)
    polar_cd = np.array(problem.get_val("data:aerodynamics:aircraft:cruise:equilibrated:CD"))
    valid_polar_cd = polar_cd[np.where(polar_cd < FIRST_INVALID_COEFF)[0]]
    assert list(valid_polar_cd)[::10] == pytest.approx(cd_polar_cruise_, abs=1e-4)
    polar_cl = np.array(problem.get_val("data:aerodynamics:aircraft:cruise:equilibrated:CL"))
    valid_polar_cl = polar_cl[np.where(polar_cl < FIRST_INVALID_COEFF)[0]]
    assert list(valid_polar_cl)[::10] == pytest.approx(cl_polar_cruise_, abs=1e-2)


def cy_beta_fus(
    XML_FILE: str,
    cy_beta_fus_: float,
):

    """Tests cy beta of the fuselage"""
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeCyBetaFuselage()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCyBetaFuselage(), ivc)
    assert problem.get_val("data:aerodynamics:fuselage:Cy_beta", units="rad**-1") == pytest.approx(
        cy_beta_fus_, rel=1e-3
    )

    problem.check_partials(compact_print=True)


def downwash_gradient(
    XML_FILE: str,
    downwash_gradient_ls_: float,
    downwash_gradient_cruise_: float,
):

    """Tests cy beta of the fuselage"""
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(DownWashGradientComputation(low_speed_aero=True)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(DownWashGradientComputation(low_speed_aero=True), ivc)
    assert problem.get_val(
        "data:aerodynamics:horizontal_tail:low_speed:downwash_gradient"
    ) == pytest.approx(downwash_gradient_ls_, rel=1e-3)

    problem.check_partials(compact_print=True)

    """Tests cy beta of the fuselage"""
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(DownWashGradientComputation(low_speed_aero=False)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(DownWashGradientComputation(low_speed_aero=False), ivc)
    assert problem.get_val(
        "data:aerodynamics:horizontal_tail:cruise:downwash_gradient"
    ) == pytest.approx(downwash_gradient_cruise_, rel=1e-3)

    problem.check_partials(compact_print=True)


def lift_aoa_rate_derivative(
    XML_FILE: str,
    cl_aoa_dot_low_speed_: float,
    cl_aoa_dot_cruise_: float,
):

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeCLAlphaDotAircraft(low_speed_aero=True)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCLAlphaDotAircraft(low_speed_aero=True), ivc)
    assert problem.get_val("data:aerodynamics:aircraft:low_speed:CL_alpha_dot") == pytest.approx(
        cl_aoa_dot_low_speed_, rel=1e-3
    )

    problem.check_partials(compact_print=True)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeCLAlphaDotAircraft(low_speed_aero=False)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCLAlphaDotAircraft(low_speed_aero=False), ivc)
    assert problem.get_val("data:aerodynamics:aircraft:cruise:CL_alpha_dot") == pytest.approx(
        cl_aoa_dot_cruise_, rel=1e-3
    )

    problem.check_partials(compact_print=True)


def lift_pitch_velocity_derivative_ht(
    XML_FILE: str,
    cl_q_ht_low_speed_: float,
    cl_q_ht_cruise_: float,
):

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeCLPitchVelocityHorizontalTail(low_speed_aero=True)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCLPitchVelocityHorizontalTail(low_speed_aero=True), ivc)
    assert problem.get_val("data:aerodynamics:horizontal_tail:low_speed:CL_q") == pytest.approx(
        cl_q_ht_low_speed_, rel=1e-3
    )

    problem.check_partials(compact_print=True)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeCLPitchVelocityHorizontalTail(low_speed_aero=False)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCLPitchVelocityHorizontalTail(low_speed_aero=False), ivc)
    assert problem.get_val("data:aerodynamics:horizontal_tail:cruise:CL_q") == pytest.approx(
        cl_q_ht_cruise_, rel=1e-3
    )

    problem.check_partials(compact_print=True)


def lift_pitch_velocity_derivative_wing(
    XML_FILE: str,
    cl_q_wing_low_speed_: float,
    cl_q_wing_cruise_: float,
):

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeCLPitchVelocityWing(low_speed_aero=True)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCLPitchVelocityWing(low_speed_aero=True), ivc)
    assert problem.get_val("data:aerodynamics:wing:low_speed:CL_q") == pytest.approx(
        cl_q_wing_low_speed_, rel=1e-3
    )

    problem.check_partials(compact_print=True)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeCLPitchVelocityWing(low_speed_aero=False)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCLPitchVelocityWing(low_speed_aero=False), ivc)
    assert problem.get_val("data:aerodynamics:wing:cruise:CL_q") == pytest.approx(
        cl_q_wing_cruise_, rel=1e-3
    )

    problem.check_partials(compact_print=True)


def lift_pitch_velocity_derivative_aircraft(
    XML_FILE: str,
    cl_q_low_speed_: float,
    cl_q_cruise_: float,
):

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeCLPitchVelocityAircraft(low_speed_aero=True)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCLPitchVelocityAircraft(low_speed_aero=True), ivc)
    assert problem.get_val("data:aerodynamics:aircraft:low_speed:CL_q") == pytest.approx(
        cl_q_low_speed_, rel=1e-3
    )

    problem.check_partials(compact_print=True)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeCLPitchVelocityAircraft(low_speed_aero=False)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCLPitchVelocityAircraft(low_speed_aero=False), ivc)
    assert problem.get_val("data:aerodynamics:aircraft:cruise:CL_q") == pytest.approx(
        cl_q_cruise_, rel=1e-3
    )

    problem.check_partials(compact_print=True)


def side_force_sideslip_derivative_wing(
    XML_FILE: str,
    cy_beta_wing_: float,
):

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeCyBetaWing()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCyBetaWing(), ivc)
    assert problem.get_val("data:aerodynamics:wing:Cy_beta", units="rad**-1") == pytest.approx(
        cy_beta_wing_, rel=1e-3
    )

    problem.check_partials(compact_print=True)


def side_force_sideslip_derivative_vt(
    XML_FILE: str,
    cy_beta_vt_low_speed_: float,
    cy_beta_vt_cruise_: float,
):

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeCyBetaVerticalTail(low_speed_aero=True)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCyBetaVerticalTail(low_speed_aero=True), ivc)
    assert problem.get_val(
        "data:aerodynamics:vertical_tail:low_speed:Cy_beta", units="rad**-1"
    ) == pytest.approx(cy_beta_vt_low_speed_, rel=1e-3)

    problem.check_partials(compact_print=True)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeCyBetaVerticalTail(low_speed_aero=False)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCyBetaVerticalTail(low_speed_aero=False), ivc)
    assert problem.get_val(
        "data:aerodynamics:vertical_tail:cruise:Cy_beta", units="rad**-1"
    ) == pytest.approx(cy_beta_vt_cruise_, rel=1e-3)

    problem.check_partials(compact_print=True)


def side_force_sideslip_aircraft(
    XML_FILE: str,
    cy_beta_low_speed_: float,
):
    # Only testing the low speed case since the high can't run on its own (fuselage and wing
    # contribution are independent of mach number and are thus only computed at low speed)
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeCYBetaAircraft(low_speed_aero=True)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCYBetaAircraft(low_speed_aero=True), ivc)
    assert problem.get_val(
        "data:aerodynamics:aircraft:low_speed:Cy_beta", units="rad**-1"
    ) == pytest.approx(cy_beta_low_speed_, rel=1e-3)

    problem.check_partials(compact_print=True)


def side_force_yaw_rate_aircraft(
    XML_FILE: str,
    cy_yaw_rate_low_speed_: float,
    cy_yaw_rate_cruise_: float,
):
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeCyYawRateAircraft(low_speed_aero=True)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCyYawRateAircraft(low_speed_aero=True), ivc)
    assert problem.get_val(
        "data:aerodynamics:aircraft:low_speed:Cy_r", units="rad**-1"
    ) == pytest.approx(cy_yaw_rate_low_speed_, rel=1e-3)

    problem.check_partials(compact_print=True)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeCyYawRateAircraft(low_speed_aero=False)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCyYawRateAircraft(low_speed_aero=False), ivc)
    assert problem.get_val(
        "data:aerodynamics:aircraft:cruise:Cy_r", units="rad**-1"
    ) == pytest.approx(cy_yaw_rate_cruise_, rel=1e-3)

    problem.check_partials(compact_print=True)


def side_force_roll_rate_aircraft(
    XML_FILE: str,
    cy_roll_rate_low_speed_: float,
    cy_roll_rate_cruise_: float,
):
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeCyRollRateAircraft(low_speed_aero=True)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCyRollRateAircraft(low_speed_aero=True), ivc)
    assert problem.get_val(
        "data:aerodynamics:aircraft:low_speed:Cy_p", units="rad**-1"
    ) == pytest.approx(cy_roll_rate_low_speed_, rel=1e-3)

    problem.check_partials(compact_print=True)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeCyRollRateAircraft(low_speed_aero=False)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCyRollRateAircraft(low_speed_aero=False), ivc)
    assert problem.get_val(
        "data:aerodynamics:aircraft:cruise:Cy_p", units="rad**-1"
    ) == pytest.approx(cy_roll_rate_cruise_, rel=1e-3)

    problem.check_partials(compact_print=True)


def roll_moment_side_slip_wing(
    XML_FILE: str,
    cl_beta_wing_low_speed_: float,
    cl_beta_wing_cruise_: float,
):
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeClBetaWing(low_speed_aero=True)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeClBetaWing(low_speed_aero=True), ivc)
    assert problem.get_val(
        "data:aerodynamics:wing:low_speed:Cl_beta", units="rad**-1"
    ) == pytest.approx(cl_beta_wing_low_speed_, rel=1e-3)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeClBetaWing(low_speed_aero=False)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeClBetaWing(low_speed_aero=False), ivc)
    assert problem.get_val(
        "data:aerodynamics:wing:cruise:Cl_beta", units="rad**-1"
    ) == pytest.approx(cl_beta_wing_cruise_, rel=1e-3)


def roll_moment_side_slip_ht(
    XML_FILE: str,
    cl_beta_ht_low_speed_: float,
    cl_beta_ht_cruise_: float,
):
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeClBetaHorizontalTail(low_speed_aero=True)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeClBetaHorizontalTail(low_speed_aero=True), ivc)
    assert problem.get_val(
        "data:aerodynamics:horizontal_tail:low_speed:Cl_beta", units="rad**-1"
    ) == pytest.approx(cl_beta_ht_low_speed_, rel=1e-3)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeClBetaHorizontalTail(low_speed_aero=False)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeClBetaHorizontalTail(low_speed_aero=False), ivc)
    assert problem.get_val(
        "data:aerodynamics:horizontal_tail:cruise:Cl_beta", units="rad**-1"
    ) == pytest.approx(cl_beta_ht_cruise_, rel=1e-3)


def roll_moment_side_slip_vt(
    XML_FILE: str,
    cl_beta_vt_low_speed_: float,
    cl_beta_vt_cruise_: float,
):
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeClBetaVerticalTail(low_speed_aero=True)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeClBetaVerticalTail(low_speed_aero=True), ivc)
    assert problem.get_val(
        "data:aerodynamics:vertical_tail:low_speed:Cl_beta", units="rad**-1"
    ) == pytest.approx(cl_beta_vt_low_speed_, rel=1e-3)

    problem.check_partials(compact_print=True)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeClBetaVerticalTail(low_speed_aero=False)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeClBetaVerticalTail(low_speed_aero=False), ivc)
    assert problem.get_val(
        "data:aerodynamics:vertical_tail:cruise:Cl_beta", units="rad**-1"
    ) == pytest.approx(cl_beta_vt_cruise_, rel=1e-3)

    problem.check_partials(compact_print=True)


def roll_moment_side_slip_aircraft(
    XML_FILE: str,
    cl_beta_low_speed_: float,
    cl_beta_cruise_: float,
):
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeClBetaAircraft(low_speed_aero=True)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeClBetaAircraft(low_speed_aero=True), ivc)
    assert problem.get_val(
        "data:aerodynamics:aircraft:low_speed:Cl_beta", units="rad**-1"
    ) == pytest.approx(cl_beta_low_speed_, rel=1e-3)

    # No need to check wing/HT contribution as it is computed with fd
    problem.check_partials(
        compact_print=True,
        excludes=["data:aerodynamics:wing:*", "data:aerodynamics:horizontal_tail:*"],
    )

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeClBetaAircraft(low_speed_aero=False)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeClBetaAircraft(low_speed_aero=False), ivc)
    assert problem.get_val(
        "data:aerodynamics:aircraft:cruise:Cl_beta", units="rad**-1"
    ) == pytest.approx(cl_beta_cruise_, rel=1e-3)

    # No need to check wing/HT contribution as it is computed with fd
    problem.check_partials(
        compact_print=True,
        excludes=["data:aerodynamics:wing:*", "data:aerodynamics:horizontal_tail:*"],
    )


def roll_moment_roll_rate_wing(
    XML_FILE: str,
    cl_p_wing_low_speed_: float,
    cl_p_wing_cruise_: float,
):
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeClRollRateWing(low_speed_aero=True)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeClRollRateWing(low_speed_aero=True), ivc)
    assert problem.get_val(
        "data:aerodynamics:wing:low_speed:Cl_p", units="rad**-1"
    ) == pytest.approx(cl_p_wing_low_speed_, rel=1e-3)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeClRollRateWing(low_speed_aero=False)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeClRollRateWing(low_speed_aero=False), ivc)
    assert problem.get_val("data:aerodynamics:wing:cruise:Cl_p", units="rad**-1") == pytest.approx(
        cl_p_wing_cruise_, rel=1e-3
    )


def roll_moment_roll_rate_ht(
    XML_FILE: str,
    cl_p_ht_low_speed_: float,
    cl_p_ht_cruise_: float,
):
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeClRollRateHorizontalTail(low_speed_aero=True)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeClRollRateHorizontalTail(low_speed_aero=True), ivc)
    assert problem.get_val(
        "data:aerodynamics:horizontal_tail:low_speed:Cl_p", units="rad**-1"
    ) == pytest.approx(cl_p_ht_low_speed_, rel=1e-3)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeClRollRateHorizontalTail(low_speed_aero=False)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeClRollRateHorizontalTail(low_speed_aero=False), ivc)
    assert problem.get_val(
        "data:aerodynamics:horizontal_tail:cruise:Cl_p", units="rad**-1"
    ) == pytest.approx(cl_p_ht_cruise_, rel=1e-3)


def roll_moment_roll_rate_vt(
    XML_FILE: str,
    cl_p_vt_low_speed_: float,
    cl_p_vt_cruise_: float,
):
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeClRollRateVerticalTail(low_speed_aero=True)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeClRollRateVerticalTail(low_speed_aero=True), ivc)
    assert problem.get_val(
        "data:aerodynamics:vertical_tail:low_speed:Cl_p", units="rad**-1"
    ) == pytest.approx(cl_p_vt_low_speed_, rel=1e-3)

    problem.check_partials(compact_print=True)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeClRollRateVerticalTail(low_speed_aero=False)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeClRollRateVerticalTail(low_speed_aero=False), ivc)
    assert problem.get_val(
        "data:aerodynamics:vertical_tail:cruise:Cl_p", units="rad**-1"
    ) == pytest.approx(cl_p_vt_cruise_, rel=1e-3)

    problem.check_partials(compact_print=True)


def roll_moment_roll_rate_aircraft(
    XML_FILE: str,
    cl_p_low_speed_: float,
    cl_p_cruise_: float,
):
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeClRollRateAircraft(low_speed_aero=True)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeClRollRateAircraft(low_speed_aero=True), ivc)
    assert problem.get_val(
        "data:aerodynamics:aircraft:low_speed:Cl_p", units="rad**-1"
    ) == pytest.approx(cl_p_low_speed_, rel=1e-3)

    # No need to check wing/HT contribution as it is computed with fd
    problem.check_partials(
        compact_print=True,
        excludes=["data:aerodynamics:wing:*", "data:aerodynamics:horizontal_tail:*"],
    )

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeClRollRateAircraft(low_speed_aero=False)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeClRollRateAircraft(low_speed_aero=False), ivc)
    assert problem.get_val(
        "data:aerodynamics:aircraft:cruise:Cl_p", units="rad**-1"
    ) == pytest.approx(cl_p_cruise_, rel=1e-3)

    # No need to check wing/HT contribution as it is computed with fd
    problem.check_partials(
        compact_print=True,
        excludes=["data:aerodynamics:wing:*", "data:aerodynamics:horizontal_tail:*"],
    )


def roll_moment_yaw_rate_wing(
    XML_FILE: str,
    cl_r_wing_low_speed_: float,
    cl_r_wing_cruise_: float,
):
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeClYawRateWing(low_speed_aero=True)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeClYawRateWing(low_speed_aero=True), ivc)
    assert problem.get_val(
        "data:aerodynamics:wing:low_speed:Cl_r", units="rad**-1"
    ) == pytest.approx(cl_r_wing_low_speed_, rel=1e-3)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeClYawRateWing(low_speed_aero=False)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeClYawRateWing(low_speed_aero=False), ivc)
    assert problem.get_val("data:aerodynamics:wing:cruise:Cl_r", units="rad**-1") == pytest.approx(
        cl_r_wing_cruise_, rel=1e-3
    )


def roll_moment_yaw_rate_vt(
    XML_FILE: str,
    cl_r_vt_low_speed_: float,
    cl_r_vt_cruise_: float,
):
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeClYawRateVerticalTail(low_speed_aero=True)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeClYawRateVerticalTail(low_speed_aero=True), ivc)
    assert problem.get_val(
        "data:aerodynamics:vertical_tail:low_speed:Cl_r", units="rad**-1"
    ) == pytest.approx(cl_r_vt_low_speed_, rel=1e-3)

    problem.check_partials(compact_print=True)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeClYawRateVerticalTail(low_speed_aero=False)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeClYawRateVerticalTail(low_speed_aero=False), ivc)
    assert problem.get_val(
        "data:aerodynamics:vertical_tail:cruise:Cl_r", units="rad**-1"
    ) == pytest.approx(cl_r_vt_cruise_, rel=1e-3)
    problem.check_partials(compact_print=True)


def roll_moment_yaw_rate_aircraft(
    XML_FILE: str,
    cl_r_low_speed_: float,
    cl_r_cruise_: float,
):
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeClYawRateAircraft(low_speed_aero=True)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeClYawRateAircraft(low_speed_aero=True), ivc)
    assert problem.get_val(
        "data:aerodynamics:aircraft:low_speed:Cl_r", units="rad**-1"
    ) == pytest.approx(cl_r_low_speed_, rel=1e-3)

    # No need to check wing contribution as it is computed with fd
    problem.check_partials(compact_print=True, excludes=["data:aerodynamics:wing:*"])

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeClYawRateAircraft(low_speed_aero=False)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeClYawRateAircraft(low_speed_aero=False), ivc)
    assert problem.get_val(
        "data:aerodynamics:aircraft:cruise:Cl_r", units="rad**-1"
    ) == pytest.approx(cl_r_cruise_, rel=1e-3)

    # No need to check wing contribution as it is computed with fd
    problem.check_partials(compact_print=True, excludes=["data:aerodynamics:wing:*"])


def roll_authority_aileron(
    XML_FILE: str,
    cl_delta_a_low_speed_: float,
    cl_delta_a_cruise_: float,
):
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeClDeltaAileron(low_speed_aero=True)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeClDeltaAileron(low_speed_aero=True), ivc)
    assert problem.get_val(
        "data:aerodynamics:aileron:low_speed:Cl_delta_a", units="rad**-1"
    ) == pytest.approx(cl_delta_a_low_speed_, rel=1e-3)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeClDeltaAileron(low_speed_aero=False)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeClDeltaAileron(low_speed_aero=False), ivc)
    assert problem.get_val(
        "data:aerodynamics:aileron:cruise:Cl_delta_a", units="rad**-1"
    ) == pytest.approx(cl_delta_a_cruise_, rel=1e-3)


def roll_moment_rudder(
    XML_FILE: str,
    cl_delta_r_low_speed_: float,
    cl_delta_r_cruise_: float,
):
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeClDeltaRudder(low_speed_aero=True)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeClDeltaRudder(low_speed_aero=True), ivc)
    assert problem.get_val(
        "data:aerodynamics:rudder:low_speed:Cl_delta_r", units="rad**-1"
    ) == pytest.approx(cl_delta_r_low_speed_, rel=1e-3)

    problem.check_partials(compact_print=True)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeClDeltaRudder(low_speed_aero=False)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeClDeltaRudder(low_speed_aero=False), ivc)
    assert problem.get_val(
        "data:aerodynamics:rudder:cruise:Cl_delta_r", units="rad**-1"
    ) == pytest.approx(cl_delta_r_cruise_, rel=1e-3)

    problem.check_partials(compact_print=True)


def pitch_moment_pitch_rate_wing(
    XML_FILE: str,
    cm_q_wing_low_speed_: float,
    cm_q_wing_cruise_: float,
):
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeCMPitchVelocityWing(low_speed_aero=True)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCMPitchVelocityWing(low_speed_aero=True), ivc)
    assert problem.get_val(
        "data:aerodynamics:wing:low_speed:Cm_q", units="rad**-1"
    ) == pytest.approx(cm_q_wing_low_speed_, rel=1e-3)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeCMPitchVelocityWing(low_speed_aero=False)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCMPitchVelocityWing(low_speed_aero=False), ivc)
    assert problem.get_val("data:aerodynamics:wing:cruise:Cm_q", units="rad**-1") == pytest.approx(
        cm_q_wing_cruise_, rel=1e-3
    )


def pitch_moment_pitch_rate_ht(
    XML_FILE: str,
    cm_q_ht_low_speed_: float,
    cm_q_ht_cruise_: float,
):
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeCMPitchVelocityHorizontalTail(low_speed_aero=True)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCMPitchVelocityHorizontalTail(low_speed_aero=True), ivc)
    assert problem.get_val(
        "data:aerodynamics:horizontal_tail:low_speed:Cm_q", units="rad**-1"
    ) == pytest.approx(cm_q_ht_low_speed_, rel=1e-3)

    problem.check_partials(compact_print=True)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeCMPitchVelocityHorizontalTail(low_speed_aero=False)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCMPitchVelocityHorizontalTail(low_speed_aero=False), ivc)
    assert problem.get_val(
        "data:aerodynamics:horizontal_tail:cruise:Cm_q", units="rad**-1"
    ) == pytest.approx(cm_q_ht_cruise_, rel=1e-3)

    problem.check_partials(compact_print=True)


def pitch_moment_pitch_rate_aircraft(
    XML_FILE: str,
    cm_q_low_speed_: float,
    cm_q_cruise_: float,
):
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeCMPitchVelocityAircraft(low_speed_aero=True)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCMPitchVelocityAircraft(low_speed_aero=True), ivc)
    assert problem.get_val(
        "data:aerodynamics:aircraft:low_speed:Cm_q", units="rad**-1"
    ) == pytest.approx(cm_q_low_speed_, rel=1e-3)

    # No need to check wing contribution as it is computed with fd
    problem.check_partials(compact_print=True, excludes=["data:aerodynamics:wing:*"])

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeCMPitchVelocityAircraft(low_speed_aero=False)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCMPitchVelocityAircraft(low_speed_aero=False), ivc)
    assert problem.get_val(
        "data:aerodynamics:aircraft:cruise:Cm_q", units="rad**-1"
    ) == pytest.approx(cm_q_cruise_, rel=1e-3)

    # No need to check wing contribution as it is computed with fd
    problem.check_partials(compact_print=True, excludes=["data:aerodynamics:wing:*"])


def pitch_moment_aoa_rate_derivative(
    XML_FILE: str,
    cm_aoa_dot_low_speed_: float,
    cm_aoa_dot_cruise_: float,
):

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeCMAlphaDotAircraft(low_speed_aero=True)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCMAlphaDotAircraft(low_speed_aero=True), ivc)
    assert problem.get_val("data:aerodynamics:aircraft:low_speed:Cm_alpha_dot") == pytest.approx(
        cm_aoa_dot_low_speed_, rel=1e-3
    )

    problem.check_partials(compact_print=True)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeCMAlphaDotAircraft(low_speed_aero=False)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCMAlphaDotAircraft(low_speed_aero=False), ivc)
    assert problem.get_val("data:aerodynamics:aircraft:cruise:Cm_alpha_dot") == pytest.approx(
        cm_aoa_dot_cruise_, rel=1e-3
    )

    problem.check_partials(compact_print=True)


def yaw_moment_sideslip_derivative_vt(
    XML_FILE: str,
    cn_beta_vt_low_speed_: float,
    cn_beta_vt_cruise_: float,
):

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeCnBetaVerticalTail(low_speed_aero=True)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCnBetaVerticalTail(low_speed_aero=True), ivc)
    assert problem.get_val(
        "data:aerodynamics:vertical_tail:low_speed:Cn_beta", units="rad**-1"
    ) == pytest.approx(cn_beta_vt_low_speed_, rel=1e-3)

    problem.check_partials(compact_print=True)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeCnBetaVerticalTail(low_speed_aero=False)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCnBetaVerticalTail(low_speed_aero=False), ivc)
    assert problem.get_val(
        "data:aerodynamics:vertical_tail:cruise:Cn_beta", units="rad**-1"
    ) == pytest.approx(cn_beta_vt_cruise_, rel=1e-3)

    problem.check_partials(compact_print=True)


def yaw_moment_sideslip_aircraft(
    XML_FILE: str,
    cn_beta_low_speed_: float,
):
    # Only testing the low speed case since the high can't run on its own (fuselage is
    # independent of mach number and are thus only computed at low speed)
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeCnBetaAircraft(low_speed_aero=True)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCnBetaAircraft(low_speed_aero=True), ivc)
    assert problem.get_val(
        "data:aerodynamics:aircraft:low_speed:Cn_beta", units="rad**-1"
    ) == pytest.approx(cn_beta_low_speed_, rel=1e-3)

    problem.check_partials(compact_print=True)


def yaw_moment_aileron(
    XML_FILE: str,
    cn_delta_a_low_speed_: float,
    cn_delta_a_cruise_: float,
):
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeCnDeltaAileron(low_speed_aero=True)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCnDeltaAileron(low_speed_aero=True), ivc)
    assert problem.get_val(
        "data:aerodynamics:aileron:low_speed:Cn_delta_a", units="rad**-1"
    ) == pytest.approx(cn_delta_a_low_speed_, rel=1e-3)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeCnDeltaAileron(low_speed_aero=False)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCnDeltaAileron(low_speed_aero=False), ivc)
    assert problem.get_val(
        "data:aerodynamics:aileron:cruise:Cn_delta_a", units="rad**-1"
    ) == pytest.approx(cn_delta_a_cruise_, rel=1e-3)


def yaw_moment_rudder(
    XML_FILE: str,
    cn_delta_r_low_speed_: float,
    cn_delta_r_cruise_: float,
):

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeCnDeltaRudder(low_speed_aero=True)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCnDeltaRudder(low_speed_aero=True), ivc)
    assert problem.get_val(
        "data:aerodynamics:rudder:low_speed:Cn_delta_r", units="rad**-1"
    ) == pytest.approx(cn_delta_r_low_speed_, rel=1e-3)

    problem.check_partials(compact_print=True)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeCnDeltaRudder(low_speed_aero=False)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCnDeltaRudder(low_speed_aero=False), ivc)
    assert problem.get_val(
        "data:aerodynamics:rudder:cruise:Cn_delta_r", units="rad**-1"
    ) == pytest.approx(cn_delta_r_cruise_, rel=1e-3)

    problem.check_partials(compact_print=True)


def yaw_moment_roll_rate_wing(
    XML_FILE: str,
    cn_p_wing_low_speed_: float,
    cn_p_wing_cruise_: float,
):
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeCnRollRateWing(low_speed_aero=True)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCnRollRateWing(low_speed_aero=True), ivc)
    assert problem.get_val(
        "data:aerodynamics:wing:low_speed:Cn_p", units="rad**-1"
    ) == pytest.approx(cn_p_wing_low_speed_, rel=1e-3)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeCnRollRateWing(low_speed_aero=False)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCnRollRateWing(low_speed_aero=False), ivc)
    assert problem.get_val("data:aerodynamics:wing:cruise:Cn_p", units="rad**-1") == pytest.approx(
        cn_p_wing_cruise_, rel=1e-3
    )


def yaw_moment_roll_rate_vt(
    XML_FILE: str,
    cn_p_vt_low_speed_: float,
    cn_p_vt_cruise_: float,
):

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeCnRollRateVerticalTail(low_speed_aero=True)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCnRollRateVerticalTail(low_speed_aero=True), ivc)
    assert problem.get_val(
        "data:aerodynamics:vertical_tail:low_speed:Cn_p", units="rad**-1"
    ) == pytest.approx(cn_p_vt_low_speed_, rel=1e-3)

    problem.check_partials(compact_print=True)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeCnRollRateVerticalTail(low_speed_aero=False)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCnRollRateVerticalTail(low_speed_aero=False), ivc)
    assert problem.get_val(
        "data:aerodynamics:vertical_tail:cruise:Cn_p", units="rad**-1"
    ) == pytest.approx(cn_p_vt_cruise_, rel=1e-3)

    problem.check_partials(compact_print=True)


def yaw_moment_roll_rate_aircraft(
    XML_FILE: str,
    cn_p_low_speed_: float,
    cn_p_cruise_: float,
):

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeCnRollRateAircraft(low_speed_aero=True)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCnRollRateAircraft(low_speed_aero=True), ivc)
    assert problem.get_val(
        "data:aerodynamics:aircraft:low_speed:Cn_p", units="rad**-1"
    ) == pytest.approx(cn_p_low_speed_, rel=1e-3)

    # Do not check partials on wing contribution as it is already calculated by fd
    problem.check_partials(compact_print=True, excludes=["data:aerodynamics:wing:*"])

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeCnRollRateAircraft(low_speed_aero=False)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCnRollRateAircraft(low_speed_aero=False), ivc)
    assert problem.get_val(
        "data:aerodynamics:aircraft:cruise:Cn_p", units="rad**-1"
    ) == pytest.approx(cn_p_cruise_, rel=1e-3)

    # Do not check partials on wing contribution as it is already calculated by fd
    problem.check_partials(compact_print=True, excludes=["data:aerodynamics:wing:*"])


def yaw_moment_yaw_rate_wing(
    XML_FILE: str,
    cn_r_wing_low_speed_: float,
    cn_r_wing_cruise_: float,
):
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeCnYawRateWing(low_speed_aero=True)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCnYawRateWing(low_speed_aero=True), ivc)
    assert problem.get_val(
        "data:aerodynamics:wing:low_speed:Cn_r", units="rad**-1"
    ) == pytest.approx(cn_r_wing_low_speed_, rel=1e-3)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeCnYawRateWing(low_speed_aero=False)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCnYawRateWing(low_speed_aero=False), ivc)
    assert problem.get_val("data:aerodynamics:wing:cruise:Cn_r", units="rad**-1") == pytest.approx(
        cn_r_wing_cruise_, rel=1e-3
    )


def yaw_moment_yaw_rate_vt(
    XML_FILE: str,
    cn_r_vt_low_speed_: float,
    cn_r_vt_cruise_: float,
):

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeCnYawRateVerticalTail(low_speed_aero=True)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCnYawRateVerticalTail(low_speed_aero=True), ivc)
    assert problem.get_val(
        "data:aerodynamics:vertical_tail:low_speed:Cn_r", units="rad**-1"
    ) == pytest.approx(cn_r_vt_low_speed_, rel=1e-3)

    problem.check_partials(compact_print=True)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeCnYawRateVerticalTail(low_speed_aero=False)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCnYawRateVerticalTail(low_speed_aero=False), ivc)
    assert problem.get_val(
        "data:aerodynamics:vertical_tail:cruise:Cn_r", units="rad**-1"
    ) == pytest.approx(cn_r_vt_cruise_, rel=1e-3)

    problem.check_partials(compact_print=True)


def yaw_moment_yaw_rate_aircraft(
    XML_FILE: str,
    cn_r_low_speed_: float,
    cn_r_cruise_: float,
):

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeCnYawRateAircraft(low_speed_aero=True)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCnYawRateAircraft(low_speed_aero=True), ivc)
    assert problem.get_val(
        "data:aerodynamics:aircraft:low_speed:Cn_r", units="rad**-1"
    ) == pytest.approx(cn_r_low_speed_, rel=1e-3)

    # Do not check partials on wing contribution as it is already calculated by fd
    problem.check_partials(compact_print=True, excludes=["data:aerodynamics:wing:*"])

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(ComputeCnYawRateAircraft(low_speed_aero=False)), __file__, XML_FILE
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCnYawRateAircraft(low_speed_aero=False), ivc)
    assert problem.get_val(
        "data:aerodynamics:aircraft:cruise:Cn_r", units="rad**-1"
    ) == pytest.approx(cn_r_cruise_, rel=1e-3)

    # Do not check partials on wing contribution as it is already calculated by fd
    problem.check_partials(compact_print=True, excludes=["data:aerodynamics:wing:*"])
