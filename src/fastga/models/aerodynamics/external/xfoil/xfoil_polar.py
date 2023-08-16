"""Computation of the airfoil aerodynamic properties using Xfoil."""
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

import logging
import os
import os.path as pth
import shutil
import sys
import warnings
from importlib.resources import path
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Tuple

import numpy as np
import pandas as pd

# noinspection PyProtectedMember
from fastoad._utils.resource_management.copy import copy_resource
from openmdao.components.external_code_comp import ExternalCodeComp
from openmdao.utils.file_wrap import InputFileGenerator

from fastga.models.aerodynamics.external.xfoil import xfoil699
from fastga.models.geometry.profiles.get_profile import get_profile
from . import resources as local_resources
from ...constants import POLAR_POINT_COUNT

OPTION_RESULT_POLAR_FILENAME = "result_polar_filename"
OPTION_RESULT_FOLDER_PATH = "result_folder_path"
OPTION_XFOIL_EXE_PATH = "xfoil_exe_path"
OPTION_ALPHA_START = "alpha_start"
OPTION_ALPHA_END = "alpha_end"
OPTION_ITER_LIMIT = "iter_limit"
OPTION_COMP_NEG_AIR_SYM = "activate_negative_angle"
DEFAULT_2D_CL_MAX = 1.9
DEFAULT_2D_CL_MIN = -1.7
DEFAULT_2D_CD_MIN = 0
ALPHA_STEP = 0.5

_INPUT_FILE_NAME = "polar_session.txt"
_STDOUT_FILE_NAME = "polar_calc.log"
_STDERR_FILE_NAME = "polar_calc.err"
_DEFAULT_AIRFOIL_FILE = "naca23012.af"
_TMP_PROFILE_FILE_NAME = "in"  # as short as possible to avoid problems of path length
_TMP_RESULT_FILE_NAME = "out"  # as short as possible to avoid problems of path length
XFOIL_EXE_NAME = "xfoil.exe"  # name of embedded XFoil executable

_LOGGER = logging.getLogger(__name__)

_XFOIL_PATH_LIMIT = 64


class XfoilPolar(ExternalCodeComp):
    """Runs a polar computation with XFOIL and returns the 2D max lift coefficient."""

    _xfoil_output_names = ["alpha", "CL", "CD", "CDp", "CM", "Top_Xtr", "Bot_Xtr"]
    """Column names in XFOIL polar result"""

    def initialize(self):

        self.options.declare(OPTION_XFOIL_EXE_PATH, default="", types=str, allow_none=True)
        self.options.declare("airfoil_folder_path", default=None, types=str, allow_none=True)
        self.options.declare("airfoil_file", default=_DEFAULT_AIRFOIL_FILE, types=str)
        self.options.declare(OPTION_RESULT_FOLDER_PATH, default="", types=str)
        self.options.declare(OPTION_RESULT_POLAR_FILENAME, default="polar_result.txt", types=str)
        self.options.declare(OPTION_ALPHA_START, default=0.0, types=float)
        self.options.declare(OPTION_ALPHA_END, default=90.0, types=float)
        self.options.declare(OPTION_ITER_LIMIT, default=100, types=int)
        self.options.declare(OPTION_COMP_NEG_AIR_SYM, default=False, types=bool)
        self.options.declare("Invicid_calculation", default=False, types=bool)
        self.options.declare(
            "single_AoA",
            default=False,
            types=bool,
            desc="If only one angle of attack is required, Cl_max_2D and Cl_min_2D won't be "
            "returned and results won't be written in the resources nor will they be read from "
            "them. In addition to that, options "
            + OPTION_ALPHA_END
            + " and "
            + OPTION_ALPHA_END
            + " must match",
        )

    def setup(self):
        """_summary_
        Set up inputs and outputs required for this operation
        """
        self.add_input("xfoil:mach", val=np.nan)
        self.add_input("xfoil:reynolds", val=np.nan)
        multiple_AoA = not self.options["single_AoA"]
        if multiple_AoA:
            self.add_output("xfoil:alpha", shape=POLAR_POINT_COUNT, units="deg")
            self.add_output("xfoil:CL", shape=POLAR_POINT_COUNT)
            self.add_output("xfoil:CD", shape=POLAR_POINT_COUNT)
            self.add_output("xfoil:CDp", shape=POLAR_POINT_COUNT)
            self.add_output("xfoil:CM", shape=POLAR_POINT_COUNT)
            self.add_output("xfoil:CL_max_2D")
            self.add_output("xfoil:CL_min_2D")
            self.add_output("xfoil:CD_min_2D")
        else:
            self.add_output("xfoil:alpha", units="deg")
            self.add_output("xfoil:CL")
            self.add_output("xfoil:CD")
            self.add_output("xfoil:CDp")
            self.add_output("xfoil:CM")

        self.declare_partials("*", "*", method="fd")

    def check_config(self, logger):
        # let void to avoid logger error on "The command cannot be empty"
        pass

    def compute(self, inputs, outputs):
        """
        Function that computes airfoil in XFoil environment and returns the different
        aerodynamic parameters.

        @param inputs: inputs parameters defined within FAST-OAD-GA
        @param outputs: outputs parameters defined within FAST-OAD-GA
        @param altitude: altitude for aerodynamic calculation in meters
        @param mach: air speed expressed in mach
        @param aoa_angle: air speed angle of attack (degree)

        """

        # Define timeout for the function
        self.options["timeout"] = 15.0

        # Get inputs and initialise outputs
        mach = round(float(inputs["xfoil:mach"]) * 1e4) / 1e4
        reynolds = round(float(inputs["xfoil:reynolds"]))
        
        # Search if data already stored for this profile and mach with reynolds values bounding
        # current value. If so, use linear interpolation with the nearest upper/lower reynolds
        no_file = True
        data_saved = None
        interpolated_result = None
        # Modify file type respect to negaive AoA/ inviscod/single AoA options
        result_file = self.define_result_file_path()
        multiple_AoA = not self.options["single_AoA"]
        single_AoA = self.options["single_AoA"]
        if pth.exists(result_file):
            no_file = False
            interpolated_result = self.interpolation_for_exist_data(result_file, mach, reynolds)
        # reslut_array_p(+AoA), reslut_array_n(-AoA)
        if interpolated_result is None:
            (
                result_array_p,
                result_array_n,
                result_folder_path,
                tmp_directory,
                tmp_result_file_path,
            ) = self.run_XFoil(inputs, outputs, reynolds, mach)
            if single_AoA:
                (
                    alpha,
                    cl,
                    cd,
                    cdp,
                    cm,
                    cl_max_2d,
                    cl_min_2d,
                    cd_min_2d,
                    error,
                ) = self.xfoil_single_aoa_convergence_check(
                    inputs, outputs, reynolds, mach, result_array_p, result_array_n
                )

            if multiple_AoA:
                (
                    alpha,
                    cl,
                    cd,
                    cdp,
                    cm,
                    cl_max_2d,
                    cl_min_2d,
                    cd_min_2d,
                    error,
                ) = self.post_processing_fill_value(result_array_p, result_array_n)
                # Fix output length if needed
                alpha, cl, cd, cdp, cm = self.fix_mutiple_aoa_output_length(alpha, cl, cd, cdp, cm)

                # Save results to defined path
                if not error:
                    results, labels = self.give_data_labels(
                        alpha, cl, cd, cdp, cm, cl_max_2d, cl_min_2d, cd_min_2d, mach, reynolds
                    )
                    if no_file or (data_saved is None):
                        data = pd.DataFrame(results, index=labels)
                    else:
                        data = pd.DataFrame(np.c_[data_saved, results], index=labels)
                    # noinspection PyBroadException
                    try:
                        data.to_csv(result_file)
                    except:
                        warnings.warn(
                            "Unable to save XFoil results to *.csv file: writing permission denied for "
                            "%s folder!" % local_resources.__path__[0]
                        )

            # Getting output files if needed
            if self.options[OPTION_RESULT_FOLDER_PATH] != "":
                self.get_output_files(result_folder_path, tmp_result_file_path)
            # Try to delete the temp directory, if process not finished correctly try to
            # close files before removing directory for second attempt
            # noinspection PyBroadException
            try:
                tmp_directory.cleanup()
            except:
                for file_path in os.listdir(tmp_directory.name):
                    if os.path.isfile(file_path):
                        # noinspection PyBroadException
                        try:
                            file = os.open(file_path, os.O_WRONLY)
                            os.close(file)
                        except:
                            _LOGGER.info("Error while trying to close %s file!", file_path)
                # noinspection PyBroadException
                try:
                    tmp_directory.cleanup()
                except:
                    _LOGGER.info(
                        "Error while trying to erase %s temporary directory!", tmp_directory.name
                    )

        else:
            # adjust the interpolated results size for outher model use
            (
                alpha,
                cl,
                cd,
                cdp,
                cm,
                cl_max_2d,
                cl_min_2d,
                cd_min_2d,
            ) = self.extract_fix_interpolated_result_length(interpolated_result)

        outputs = self.define_outputs(
            outputs, alpha, cl, cd, cdp, cm, cl_max_2d, cl_min_2d, cd_min_2d, multiple_AoA
        )

    def _write_script_file(
        self,
        reynolds,
        mach,
        tmp_profile_file_path,
        tmp_result_file_path,
        alpha_start,
        alpha_end,
        step,
    ):
        """_summary_
        Create command script to run XFoil for obtaining results

        Args:
            reynolds (_list_): Renold's number
            mach (_list_): mach number
            tmp_profile_file_path (_path_): temporary profile path
            tmp_result_file_path (_path_): temporary result path
            alpha_start (_float_): starting angle of attack in XFoil caculation
            alpha_end (_float_): ending angle of attack in XFoil caculation
            step (_float_): steps between each angle of attack in XFoil calculation
        """
        parser = InputFileGenerator()
        invicid = self.options["Invicid_calculation"]
        viscid = not self.options["Invicid_calculation"]
        single_AoA = self.options["single_AoA"]
        # Check the computation options and select different script templates
        if invicid:
            _INPUT_FILE_NAME = "polar_session_inv.txt"
        elif viscid:
            _INPUT_FILE_NAME = "polar_session.txt"
        if single_AoA and invicid:
            _INPUT_FILE_NAME = "polar_session_single_AoA_inv.txt"
        elif single_AoA:
            _INPUT_FILE_NAME = "polar_session_single_AoA.txt"

        # input command to run XFoil
        with path(local_resources, _INPUT_FILE_NAME) as input_template_path:
            parser.set_template_file(str(input_template_path))
            parser.set_generated_file(self.stdin)
            if viscid:
                parser.mark_anchor("RE")
                parser.transfer_var(float(reynolds), 1, 1)
            parser.mark_anchor("M")
            parser.transfer_var(float(mach), 1, 1)
            parser.mark_anchor("ITER")
            parser.transfer_var(self.options[OPTION_ITER_LIMIT], 1, 1)

            if single_AoA:
                parser.mark_anchor("ALFA")
                parser.transfer_var(alpha_start, 1, 1)
            else:
                parser.mark_anchor("ASEQ")
                parser.transfer_var(alpha_start, 1, 1)
                parser.transfer_var(alpha_end, 2, 1)

            parser.transfer_var(step, 3, 1)
            parser.reset_anchor()
            parser.mark_anchor("/profile")
            parser.transfer_var(tmp_profile_file_path, 0, 1)
            parser.mark_anchor("/polar_result")
            parser.transfer_var(tmp_result_file_path, 0, 1)
            parser.generate()

    @staticmethod
    def _read_polar(xfoil_result_file_path: str) -> np.ndarray:
        """
        :param xfoil_result_file_path:
        :return: numpy array with XFoil polar results
        """
        if os.path.isfile(xfoil_result_file_path):
            dtypes = [(name, "f8") for name in XfoilPolar._xfoil_output_names]
            result_array = np.genfromtxt(xfoil_result_file_path, skip_header=12, dtype=dtypes)
            return result_array

        _LOGGER.error("XFOIL results file not found")
        return np.array([])

    def _get_max_cl(self, alpha: np.ndarray, lift_coeff: np.ndarray) -> Tuple[float, bool]:
        """

        :param alpha:
        :param lift_coeff: CL
        :return: max CL within +/- 0.3 around linear zone if enough alpha computed, or default value
        otherwise
        """
        alpha_range = self.options[OPTION_ALPHA_END] - self.options[OPTION_ALPHA_START]
        if len(alpha) > 2:
            covered_range = max(alpha) - min(alpha)
            if np.abs(covered_range / alpha_range) >= 0.4:
                lift_fct = (
                    lambda x: (lift_coeff[1] - lift_coeff[0])
                    / (alpha[1] - alpha[0])
                    * (x - alpha[0])
                    + lift_coeff[0]
                )
                delta = np.abs(lift_coeff - lift_fct(alpha))
                return max(lift_coeff[delta <= 0.3]), False

        _LOGGER.warning(
            "2D CL max not found, less than 40%% of angle range computed: using default value %f",
            DEFAULT_2D_CL_MAX,
        )
        return DEFAULT_2D_CL_MAX, True

    def _get_min_cl(self, alpha: np.ndarray, lift_coeff: np.ndarray) -> Tuple[float, bool]:
        """
        :param alpha:
        :param lift_coeff: CL

        :return: min CL +/- 0.3 around linear zone if enough alpha computed, or default value
        otherwise
        """
        alpha_range = self.options[OPTION_ALPHA_END] - self.options[OPTION_ALPHA_START]
        if len(alpha) > 2:
            covered_range = max(alpha) - min(alpha)
            if covered_range / alpha_range >= 0.4:
                lift_fct = (
                    lambda x: (lift_coeff[1] - lift_coeff[0])
                    / (alpha[1] - alpha[0])
                    * (x - alpha[0])
                    + lift_coeff[0]
                )
                delta = np.abs(lift_coeff - lift_fct(alpha))
                return min(lift_coeff[delta <= 0.3]), False

        _LOGGER.warning(
            "2D CL min not found, less than 40%% of angle range computed: using default value %f",
            DEFAULT_2D_CL_MIN,
        )
        return DEFAULT_2D_CL_MIN, True

    @staticmethod
    def _reshape(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        # Delete ending 0.0 values
        for idx in range(len(x)):
            if np.sum(x[idx : len(x)] == 0.0) == (len(x) - idx):
                y = y[0:idx]
                break
        return y

    @staticmethod
    def _create_tmp_directory() -> TemporaryDirectory:
        """_summary_
        Dev Note: XFOIL fails if length of provided file path exceeds 64 characters.
        Changing working directory to the tmp dir would allow to just provide file name,
        but it is not really safe (at least, it does mess with the coverage report).
        Then the point is to get a tmp directory with a short path.
        On Windows, the default (user-dependent) tmp dir can exceed the limit.
        Therefore, as a second choice, tmp dir is created as close of user home
        directory as possible.
        """
        tmp_candidates = []
        for tmp_base_path in [None, pth.join(str(Path.home()), ".fast")]:
            if tmp_base_path is not None:
                os.makedirs(tmp_base_path, exist_ok=True)
            tmp_directory = TemporaryDirectory(prefix="x", dir=tmp_base_path)
            tmp_candidates.append(tmp_directory.name)
            tmp_profile_file_path = pth.join(tmp_directory.name, _TMP_PROFILE_FILE_NAME)
            tmp_result_file_path = pth.join(tmp_directory.name, _TMP_RESULT_FILE_NAME)

            if max(len(tmp_profile_file_path), len(tmp_result_file_path)) <= _XFOIL_PATH_LIMIT:
                # tmp_directory is OK. Stop there
                break
            # tmp_directory has a too long path. Erase and continue...
            tmp_directory.cleanup()

        if max(len(tmp_profile_file_path), len(tmp_result_file_path)) > _XFOIL_PATH_LIMIT:
            raise IOError(
                "Could not create a tmp directory where file path will respects XFOIL "
                "limitation (%i): tried %s" % (_XFOIL_PATH_LIMIT, tmp_candidates)
            )

        return tmp_directory

    def run_XFoil(self, inputs, outputs, reynolds, mach):
        """_summary_
        Imoprt required list data and XFoil.exe to script and save
        for exporting the results
        Args:
            inputs (_list_): input data list
            outputs (_list_): ooutput data list
            reynolds (_list_): reynold's number list
            mach (_list_): mach number list

        Raises:
            TimeoutError: _description_
            TimeoutError: _description_

        Returns:
            _list_: export data list for postprocessing
        """

        # Create result folder first (if it must fail, let it fail as soon as possible)
        result_folder_path = self.options[OPTION_RESULT_FOLDER_PATH]
        if result_folder_path != "":
            os.makedirs(result_folder_path, exist_ok=True)

        # Pre-processing (populating temp directory)
        # XFoil exe
        tmp_directory = self._create_tmp_directory()
        if self.options[OPTION_XFOIL_EXE_PATH]:
            # if a path for Xfoil has been provided, simply use it
            self.options["command"] = [self.options[OPTION_XFOIL_EXE_PATH]]
        else:
            # otherwise, copy the embedded resource in tmp dir
            # noinspection PyTypeChecker
            copy_resource(xfoil699, XFOIL_EXE_NAME, tmp_directory.name)
            self.options["command"] = [pth.join(tmp_directory.name, XFOIL_EXE_NAME)]

        # I/O files
        self.stdin = pth.join(tmp_directory.name, _INPUT_FILE_NAME)
        self.stdout = pth.join(tmp_directory.name, _STDOUT_FILE_NAME)
        self.stderr = pth.join(tmp_directory.name, _STDERR_FILE_NAME)

        # profile file
        tmp_profile_file_path = pth.join(tmp_directory.name, _TMP_PROFILE_FILE_NAME)
        profile = get_profile(
            airfoil_folder_path=self.options["airfoil_folder_path"],
            file_name=self.options["airfoil_file"],
        ).get_sides()
        # noinspection PyTypeChecker
        np.savetxt(
            tmp_profile_file_path,
            profile.to_numpy(),
            fmt="%.15f",
            delimiter=" ",
            header="Wing",
            comments="",
        )

        # standard input file
        tmp_result_file_path = pth.join(tmp_directory.name, _TMP_RESULT_FILE_NAME)
        self._write_script_file(
            reynolds,
            mach,
            tmp_profile_file_path,
            tmp_result_file_path,
            self.options[OPTION_ALPHA_START],
            self.options[OPTION_ALPHA_END],
            ALPHA_STEP,
        )

        # Run XFOIL
        self.options["external_input_files"] = [self.stdin, tmp_profile_file_path]
        self.options["external_output_files"] = [tmp_result_file_path]
        # noinspection PyBroadException
        try:
            super().compute(inputs, outputs)
            result_array_p = self._read_polar(tmp_result_file_path)
        except:
            # catch the error and try to read result file for non-convergence on higher angles
            error = sys.exc_info()[1]
            try:
                result_array_p = self._read_polar(tmp_result_file_path)
            except:
                raise TimeoutError("<p>Error: %s</p>" % error)
        result_array_n = np.array([])

        if self.options[OPTION_COMP_NEG_AIR_SYM]:
            os.remove(self.stdin)
            os.remove(self.stdout)
            os.remove(self.stderr)
            os.remove(tmp_result_file_path)
            alpha_start = min(-1 * self.options[OPTION_ALPHA_START], -ALPHA_STEP)
            self._write_script_file(
                reynolds,
                mach,
                tmp_profile_file_path,
                tmp_result_file_path,
                alpha_start,
                -1 * self.options[OPTION_ALPHA_END],
                -ALPHA_STEP,
            )
            # noinspection PyBroadException
            try:
                super().compute(inputs, outputs)
                result_array_n = self._read_polar(tmp_result_file_path)
            except:
                # catch the error and try to read result file for non-convergence on higher
                # angles
                e = sys.exc_info()[1]
                try:
                    result_array_n = self._read_polar(tmp_result_file_path)
                except:
                    raise TimeoutError("<p>Error: %s</p>" % e)

        return (
            result_array_p,
            result_array_n,
            result_folder_path,
            tmp_directory,
            tmp_result_file_path,
        )

    def interpolation_for_exist_data(self, result_file, mach, reynolds):
        """_summary_
        First, check if the existed data list has the exact same reult.
        Then, use existed Renolds number and mach number to find
        corresponded existed results for interpolation.
        With interpolation, the the result can be obtained.
        Args:
            result_file (_path_): result data path
            mach (_list_): existed mach number list
            reynolds (_list_): existed reynolds umber list

        Returns:
            _list_: interpolated result list
        """
        data_saved = pd.read_csv(result_file)
        values = data_saved.to_numpy()[:, 1 : len(data_saved.to_numpy()[0])]
        labels = data_saved.to_numpy()[:, 0].tolist()
        data_saved = pd.DataFrame(values, index=labels)
        saved_mach_list = data_saved.loc["mach", :].to_numpy().astype(float)
        index_near_mach = np.where(abs(saved_mach_list - mach) < 0.03)[0]
        near_mach = []
        distance_to_mach = []
        # Check if there is a velocity (Mach) value that is close to this one
        for index in index_near_mach:
            if not (saved_mach_list[index] in near_mach):
                near_mach.append(saved_mach_list[index])
                distance_to_mach.append(abs(saved_mach_list[index] - mach))
        if not near_mach:
            index_mach = np.where(data_saved.loc["mach", :].to_numpy() == str(mach))[0]
        else:
            selected_mach_index = distance_to_mach.index(min(distance_to_mach))
            index_mach = np.where(saved_mach_list == near_mach[selected_mach_index])[0]
        data_reduced = data_saved.loc[labels, index_mach]
        # Search if this exact reynolds has been computed and save results
        reynolds_vect = np.array(
            [float(x) for x in list(data_reduced.loc["reynolds", :].to_numpy())]
        )
        index_reynolds = index_mach[np.where(reynolds_vect == reynolds)[0]]
        if len(index_reynolds) == 1:
            interpolated_result = data_reduced.loc[labels, index_reynolds]
        # Else search for lower/upper Reynolds
        else:
            lower_reynolds = reynolds_vect[np.where(reynolds_vect < reynolds)[0]]
            upper_reynolds = reynolds_vect[np.where(reynolds_vect > reynolds)[0]]
            if not (len(lower_reynolds) == 0 or len(upper_reynolds) == 0):
                index_lower_reynolds = index_mach[np.where(reynolds_vect == max(lower_reynolds))[0]]
                index_upper_reynolds = index_mach[np.where(reynolds_vect == min(upper_reynolds))[0]]
                lower_values = data_reduced.loc[labels, index_lower_reynolds]
                upper_values = data_reduced.loc[labels, index_upper_reynolds]
                # Initialise values with lower reynolds
                interpolated_result = lower_values
                # Calculate reynolds ratio split for linear interpolation
                x_ratio = (min(upper_reynolds) - reynolds) / (
                    min(upper_reynolds) - max(lower_reynolds)
                )
                # Search for common alpha range for linear interpolation
                
                aoa_lower = string_to_array(lower_values.loc["alpha", index_lower_reynolds].to_numpy()[0])
                alpha_lower = aoa_lower.tolist()
                
                aoa_upper = string_to_array(upper_values.loc["alpha", index_upper_reynolds].to_numpy()[0])
                alpha_upper = aoa_upper.tolist()
                
                alpha_shared = np.array(list(set(alpha_upper).intersection(alpha_lower)))
                interpolated_result.loc["alpha", index_lower_reynolds] = str(alpha_shared.tolist())
                labels.remove("alpha")
                # Calculate average values (cd, cl...) with linear interpolation
                for label in labels:
                    lower_value = np.array(
                        lower_values.loc[label, index_lower_reynolds].to_numpy()[0],
                        dtype=np.float64,
                    ).ravel()
                    upper_value = np.array(
                        upper_values.loc[label, index_upper_reynolds].to_numpy()[0],
                        dtype=np.float64,
                    ).ravel()
                    # If values relative to alpha vector, performs interpolation with shared
                    # vector
                    # aoa_upper = string_2_float_array(alpha_upper)
                    # aoa_lower = string_2_float_array(alpha_lower)
                    if np.size(lower_value) == len(alpha_lower):
                        lower_value = np.interp(alpha_shared, aoa_lower, lower_value)
                        upper_value = np.interp(alpha_shared, aoa_upper, upper_value)
                        return lower_value, upper_value
                    value = (lower_value * x_ratio + upper_value * (1 - x_ratio)).tolist()
                    interpolated_result.loc[label, index_lower_reynolds] = str(value)
                    return interpolated_result

                return interpolated_result

    def define_result_file_path(self):
        """_summary_
        rename the file for better underdtanding
        of the computation options
        Returns:
            _path_: result data path
        """
        negative_angle_badge = ""
        single_aoa_badge = ""
        invicid_badge = ""
        if self.options[OPTION_COMP_NEG_AIR_SYM]:
            negative_angle_badge = "_" + str(int(np.ceil(self.options[OPTION_ALPHA_END]))) + "S"
        if self.options["single_AoA"]:
            single_aoa_badge = "_S_AOA"
        if self.options["Invicid_calculation"]:
            invicid_badge = "_inv"
        naming = negative_angle_badge + single_aoa_badge + invicid_badge
        result_file = pth.join(
            pth.split(os.path.realpath(__file__))[0],
            "resources",
            self.options["airfoil_file"].replace(".af", naming) + ".csv",
        )
        return result_file

    def post_processing_fill_value(self, result_array_p, result_array_n):
        """_summary_
        Filling value after XFoil calculation
        Args:
            result_array_p (_array_): results with positive angle of attacks
            result_array_n (_array_): results with negative angle of attacks

        Returns:
            _list_: aerodynamic characteristics in different angle of attack
        """
        if self.options["single_AoA"]:
            alpha = result_array_p["alpha"].tolist()
            cl = result_array_p["CL"].tolist()
            cd = result_array_p["CD"].tolist()
            cdp = result_array_p["CDp"].tolist()
            cm = result_array_p["CM"].tolist()
            cl_max_2d = []
            cl_min_2d = []
            cd_min_2d = []
            error = []
        elif self.options[OPTION_COMP_NEG_AIR_SYM]:
            cl_max_2d, error = self._get_max_cl(result_array_p["alpha"], result_array_p["CL"])
            # noinspection PyUnboundLocalVariable
            cl_min_2d, _ = self._get_min_cl(result_array_n["alpha"], result_array_n["CL"])
            cd_min_2d = np.min(result_array_p["CD"])
            alpha = result_array_n["alpha"].tolist()
            alpha.reverse()
            alpha.extend(result_array_p["alpha"].tolist())
            cl = result_array_n["CL"].tolist()
            cl.reverse()
            cl.extend(result_array_p["CL"].tolist())
            cd = result_array_n["CD"].tolist()
            cd.reverse()
            cd.extend(result_array_p["CD"].tolist())
            cdp = result_array_n["CDp"].tolist()
            cdp.reverse()
            cdp.extend(result_array_p["CDp"].tolist())
            cm = result_array_n["CM"].tolist()
            cm.reverse()
            cm.extend(result_array_p["CM"].tolist())
        else:
            cl_max_2d, error = self._get_max_cl(result_array_p["alpha"], result_array_p["CL"])
            cl_min_2d, _ = self._get_min_cl(result_array_p["alpha"], result_array_p["CL"])
            cd_min_2d = np.min(result_array_p["CD"])
            alpha = result_array_p["alpha"].tolist()
            cl = result_array_p["CL"].tolist()
            cd = result_array_p["CD"].tolist()
            cdp = result_array_p["CDp"].tolist()
            cm = result_array_p["CM"].tolist()

        return alpha, cl, cd, cdp, cm, cl_max_2d, cl_min_2d, cd_min_2d, error

    def xfoil_single_aoa_convergence_check(
        self, inputs, outputs, reynolds, mach, result_array_p, result_array_n
    ):

        try:
            # Post-processing
            (
                alpha,
                cl,
                cd,
                cdp,
                cm,
                cl_max_2d,
                cl_min_2d,
                cd_min_2d,
                error,
            ) = self.post_processing_fill_value(result_array_p, result_array_n)

        except:
            self.options[OPTION_ITER_LIMIT] = 10 * self.options[OPTION_ITER_LIMIT]

            # Rerun XFoil
            (
                result_array_p,
                result_array_n,
                result_folder_path,
                tmp_directory,
                tmp_result_file_path,
            ) = self.run_XFoil(inputs, outputs, reynolds, mach)

            # Try post processing again
            try:
                (
                    alpha,
                    cl,
                    cd,
                    cdp,
                    cm,
                    cl_max_2d,
                    cl_min_2d,
                    cd_min_2d,
                    error,
                ) = self.post_processing_fill_value(result_array_p, result_array_n)
            except:
                # Custom error message
                print("Error: Xfoil failed to converge, please increase the iteration limit")
                raise
        return (
            alpha,
            cl,
            cd,
            cdp,
            cm,
            cl_max_2d,
            cl_min_2d,
            cd_min_2d,
            error,
        )

    def define_outputs(
        self, outputs, alpha, cl, cd, cdp, cm, cl_max_2d, cl_min_2d, cd_min_2d, multiple_AoA
    ):
        """_summary_
        assign output to outputs list
        Args:
            outputs (_list_): the output data list needs to be filled
            alpha (_array_):length-adjusted angle of attack array
            cl (_array_): length-adjusted lift coefficient array
            cd (_array_): length-adjusted drag coefficient array
            cdp (_array_): length-adjusted pressure drag coefficient array
            cm (_array_): length-adjusted moment coefficient array
            cl_max_2d (_float_): maximum lift coefficient
            cl_min_2d (_float_): miniimum lift coefficient
            cd_min_2d (_float_): miniimum drag coefficient
            multiple_AoA (_boolean_): multiple angle of attack option

        Returns:
            _type_: _description_
        """
        outputs["xfoil:alpha"] = alpha
        outputs["xfoil:CL"] = cl
        outputs["xfoil:CD"] = cd
        outputs["xfoil:CDp"] = cdp
        outputs["xfoil:CM"] = cm
        if multiple_AoA:
            outputs["xfoil:CL_max_2D"] = cl_max_2d
            outputs["xfoil:CL_min_2D"] = cl_min_2d
            outputs["xfoil:CD_min_2D"] = cd_min_2d

        return outputs

    def fix_mutiple_aoa_output_length(self, alpha, cl, cd, cdp, cm):
        """_summary_
        Fix the length of all results respect to the length of array alpha
        so that the results will have no error in fast-oad.
        If the length of result arrays are longer, this function will interpolate
        value for ALPHA. On the contrary, the extra zero elements is filled in
        the results array.
        Args:
            alpha (_list_): angle of attach list
            cl (_list_): lift coefficient list
            cd (_list_): drag coefficient list
            cdp (_list_): pressure drag coefficient list
            cm (_list_): moment coefficient list

        Returns:
            _array_: length-modified aerodynamic characteristic array
        """
        shorter = POLAR_POINT_COUNT < len(alpha)
        # use inerpolation to fill missing values and add zero for values that are out of range
        if shorter:
            alpha_interp = np.linspace(alpha[0], alpha[-1], POLAR_POINT_COUNT)
            cl = np.interp(alpha_interp, alpha, cl)
            cd = np.interp(alpha_interp, alpha, cd)
            cdp = np.interp(alpha_interp, alpha, cdp)
            cm = np.interp(alpha_interp, alpha, cm)
            alpha = alpha_interp
            warnings.warn("Defined polar point in fast aerodynamics\\constants.py exceeded!")
        else:
            alpha = add_zeros(alpha)
            cl = add_zeros(cl)
            cd = add_zeros(cd)
            cdp = add_zeros(cdp)
            cm = add_zeros(cm)

        return alpha, cl, cd, cdp, cm

    def extract_fix_interpolated_result_length(self, interpolated_result):
        """_summary_
        Fix the length of all results respect to the length of array ALPHA
        so that the results will have no error in fast-oad.
        If the length of result arrays are longer, this function will interpolate
        value for ALPHA. On the contrary, the extra zero elements is filled to make
        the results array as long as ALPHA.

        Args:
            interpolated_result (_list_): result dataset with labels and values

        Returns:
            _array_: length-modified aerodynamic characteristic array
        """
        # Extract results
        cl_max_2d = string_to_array(interpolated_result.loc["cl_max_2d", :].to_numpy()[0]).ravel()
        cl_min_2d = string_to_array(interpolated_result.loc["cl_min_2d", :].to_numpy()[0]).ravel()
        ALPHA = string_to_array(interpolated_result.loc["alpha", :].to_numpy()[0]).ravel()
        CL = string_to_array(interpolated_result.loc["cl", :].to_numpy()[0]).ravel()
        CD = string_to_array(interpolated_result.loc["cd", :].to_numpy()[0]).ravel()
        CDP = string_to_array(interpolated_result.loc["cdp", :].to_numpy()[0]).ravel()
        CM = string_to_array(interpolated_result.loc["cm", :].to_numpy()[0]).ravel()
        cd_min_2d = np.min(CD)
        # Modify vector length if necessary
        if POLAR_POINT_COUNT < len(ALPHA):
            alpha = np.linspace(ALPHA[0], ALPHA[-1], POLAR_POINT_COUNT)
            cl = np.interp(alpha, ALPHA, CL)
            cd = np.interp(alpha, ALPHA, CD)
            cdp = np.interp(alpha, ALPHA, CDP)
            cm = np.interp(alpha, ALPHA, CM)
        else:
            alpha = add_zeros(ALPHA)
            cl = add_zeros(CL)
            cd = add_zeros(CD)
            cdp = add_zeros(CDP)
            cm = add_zeros(CM)

        return alpha, cl, cd, cdp, cm, cl_max_2d, cl_min_2d, cd_min_2d

    def give_data_labels(
        self, alpha, cl, cd, cdp, cm, cl_max_2d, cl_min_2d, cd_min_2d, mach, reynolds
    ):
        """
        Five data labes and prepare list for later writing to the file
        """
        results = [
            np.array(mach),
            np.array(reynolds),
            np.array(cl_max_2d),
            np.array(cl_min_2d),
            np.array(cd_min_2d),
            str(self._reshape(alpha, alpha).tolist()),
            str(self._reshape(alpha, cl).tolist()),
            str(self._reshape(alpha, cd).tolist()),
            str(self._reshape(alpha, cdp).tolist()),
            str(self._reshape(alpha, cm).tolist()),
        ]
        labels = [
            "mach",
            "reynolds",
            "cl_max_2d",
            "cl_min_2d",
            "cd_min_2d",
            "alpha",
            "cl",
            "cd",
            "cdp",
            "cm",
        ]
        return results, labels

    def get_output_files(self, result_folder_path, tmp_result_file_path):
        if pth.exists(tmp_result_file_path):
            polar_file_path = pth.join(
                result_folder_path, self.options[OPTION_RESULT_POLAR_FILENAME]
            )
            shutil.move(tmp_result_file_path, polar_file_path)

        if pth.exists(self.stdout):
            stdout_file_path = pth.join(result_folder_path, _STDOUT_FILE_NAME)
            shutil.move(self.stdout, stdout_file_path)

        if pth.exists(self.stderr):
            stderr_file_path = pth.join(result_folder_path, _STDERR_FILE_NAME)
            shutil.move(self.stderr, stderr_file_path)

def string_to_array(arr):
    arr = np.array([float(v) for v in arr.strip('[]').split(',')])
    return arr

def add_zeros(arr):
    additional_zeros = list(np.zeros(POLAR_POINT_COUNT - len(arr)))
    if not isinstance(arr,list):
        arr = arr.tolist()
    arr.extend(additional_zeros)
    arr = np.asarray(arr)
    return arr