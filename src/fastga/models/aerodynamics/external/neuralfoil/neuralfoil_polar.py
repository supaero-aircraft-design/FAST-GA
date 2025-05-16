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
import numpy as np
import pandas as pd
import neuralfoil as nf

from openmdao.components.external_code_comp import ExternalCodeComp
from pathlib import Path
from fastga.models.aerodynamics import airfoil_folder
from fastga.command.api import string_to_array
from ...constants import POLAR_POINT_COUNT

OPTION_RESULT_POLAR_FILENAME = "result_polar_filename"
OPTION_RESULT_FOLDER_PATH = "result_folder_path"
OPTION_ALPHA_START = "alpha_start"
OPTION_ALPHA_END = "alpha_end"
OPTION_COMP_NEG_AIR_SYM = "activate_negative_angle"
ALPHA_STEP = 0.5

_DEFAULT_AIRFOIL_FILE = "naca23012.af"

_LOGGER = logging.getLogger(__name__)

_XFOIL_PATH_LIMIT = 64


class NeuralfoilPolar(ExternalCodeComp):
    """Runs a polar computation with XFOIL and returns the 2D max lift coefficient."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Column names in XFOIL polar result
        self._xfoil_output_names = ["alpha", "CL", "CD", "CDp", "CM", "Top_Xtr", "Bot_Xtr"]

    def initialize(self):
        self.options.declare("airfoil_folder_path", default=None, types=str, allow_none=True)
        self.options.declare("airfoil_file", default=_DEFAULT_AIRFOIL_FILE, types=str)
        self.options.declare(OPTION_RESULT_FOLDER_PATH, default="", types=str)
        self.options.declare(OPTION_RESULT_POLAR_FILENAME, default="polar_result.txt", types=str)
        self.options.declare(OPTION_ALPHA_START, default=0.0, types=float)
        self.options.declare(OPTION_ALPHA_END, default=90.0, types=float)
        self.options.declare(OPTION_COMP_NEG_AIR_SYM, default=False, types=bool)
        self.options.declare("inviscid_calculation", default=False, types=bool)
        self.options.declare(
            "single_AoA",
            default=False,
            types=bool,
            desc="If only one angle of attack is required, Cl_max_2D and Cl_min_2D won't be "
            "returned and results won't be written in the resources nor will they be read from "
            "them. In addition to that, options "
            + OPTION_ALPHA_START
            + " and "
            + OPTION_ALPHA_END
            + " must match",
        )

    def setup(self):
        """
        Set up inputs and outputs required for this operation
        """

        multiple_aoa = not self.options["single_AoA"]

        self.add_input("neuralfoil:reynolds", val=np.nan)
        self.add_input("neuralfoil:mach", val=np.nan)

        if multiple_aoa:
            self.add_output("neuralfoil:alpha", shape=POLAR_POINT_COUNT, units="deg")
            self.add_output("neuralfoil:CL", shape=POLAR_POINT_COUNT)
            self.add_output("neuralfoil:CD", shape=POLAR_POINT_COUNT)
            self.add_output("neuralfoil:CDp", val=np.zeros(POLAR_POINT_COUNT))
            self.add_output("neuralfoil:CM", shape=POLAR_POINT_COUNT)
            self.add_output("neuralfoil:CL_max_2D")
            self.add_output("neuralfoil:CL_min_2D")
            self.add_output("neuralfoil:CD_min_2D")

        else:
            self.add_output("neuralfoil:alpha", units="deg")
            self.add_output("neuralfoil:CL")
            self.add_output("neuralfoil:CD")
            self.add_output("neuralfoil:CDp", val=0.0)
            self.add_output("neuralfoil:CM")

        self.declare_partials("*", "*", method="fd")

    def check_config(self, logger):
        # let void to avoid logger error on "The command cannot be empty"
        pass

    def compute(self, inputs, outputs):
        """
        Function that computes airfoil aerodynamics with XFoil and returns the different 2D
        aerodynamic parameters.

        @param inputs: inputs parameters defined within FAST-OAD-GA
        @param outputs: outputs parameters defined within FAST-OAD-GA
        """

        # Define timeout for the function
        self.options["timeout"] = 15.0

        # Get inputs and initialise outputs
        reynolds = round(float(inputs["neuralfoil:reynolds"]))

        # Search if data already stored for this profile and mach with reynolds values bounding
        # current value. If so, use linear interpolation with the nearest upper/lower reynolds
        interpolated_result = None

        # Modify file type respect to negative AoA/inviscid/single AoA options
        result_file = self._define_result_file_path()
        multiple_aoa = not self.options["single_AoA"]
        alpha_start = self.options[OPTION_ALPHA_START]
        alpha_end = self.options[OPTION_ALPHA_END]

        if pth.exists(result_file):
            interpolated_result, data_saved = self._interpolation_for_exist_data(
                result_file, 0.0, reynolds
            )

        if interpolated_result is None:
            alpha = self.options[OPTION_ALPHA_START]
            if self.options["airfoil_folder_path"] is None:
                self.options["airfoil_folder_path"] = airfoil_folder.__path__[0]
            airfoil_path = Path(self.options["airfoil_folder_path"]) / self.options["airfoil_file"]

            if multiple_aoa:
                alpha = np.arange(
                    -alpha_end if self.options[OPTION_COMP_NEG_AIR_SYM] else alpha_start,
                    alpha_end + ALPHA_STEP,
                    ALPHA_STEP,
                )

            results = nf.get_aero_from_dat_file(airfoil_path, alpha=alpha, Re=reynolds)
            cl = results["CL"]
            cd = results["CD"]
            cm = results["CM"]

            if multiple_aoa:
                results["AoA"] = alpha
                cl_max_2d = np.max(cl)
                cl_min_2d = np.min(cl)
                cd_min_2d = np.min(cd)
                alpha, cl, cd, cm = self._fix_calculation_result_length(results)

        else:
            # adjust the interpolated results size for other model use

            (
                alpha,
                cl,
                cd,
                cdp,
                cm,
                cl_max_2d,
                cl_min_2d,
                cd_min_2d,
            ) = self._extract_fix_interpolated_result_length(interpolated_result)

        # Defining outputs -------------------------------------------------------------------------
        outputs["neuralfoil:alpha"] = alpha
        outputs["neuralfoil:CL"] = cl
        outputs["neuralfoil:CD"] = cd
        outputs["neuralfoil:CM"] = cm
        if multiple_aoa:
            outputs["neuralfoil:CL_max_2D"] = cl_max_2d
            outputs["neuralfoil:CL_min_2D"] = cl_min_2d
            outputs["neuralfoil:CD_min_2D"] = cd_min_2d

    def _define_result_file_path(self):
        """
        Each computation option (airfoil name, max AOA, single AOA, ...) will lead to unique file
        name for stored results. We can thus check if results already exists if the file name
        already exists. This function generate file name corresponding to the computation options.

        :return result_file: path to result file if it exists
        """

        # Generate tags for the different options
        if self.options[OPTION_COMP_NEG_AIR_SYM]:
            negative_angle_tag = "_" + str(int(np.ceil(self.options[OPTION_ALPHA_END]))) + "S"
        else:
            negative_angle_tag = ""

        if self.options["single_AoA"]:
            single_aoa_tag = "_1_AOA"
        else:
            single_aoa_tag = ""

        if self.options["inviscid_calculation"]:
            inviscid_tag = "_inv"
        else:
            inviscid_tag = ""

        naming = negative_angle_tag + single_aoa_tag + inviscid_tag

        result_file = pth.join(
            pth.split(os.path.realpath(__file__))[0],
            "resources",
            self.options["airfoil_file"].replace(".af", naming) + ".csv",
        )

        return result_file

    @staticmethod
    def _interpolation_for_exist_data(result_file, mach, reynolds):
        """
        If a result file exist, we then check if the proper Mach number exists. If it does we
        check if a Reynolds number below and above it exist, in which case we interpolate between
        the two. Even if interpolated results can't be obtained (Reynold too low or too high),
        if we enter this function, it means the corresponding airfoil exists ad the Xfoil run we
        are about to do should be added to the existing results.

        :return interpolated_result: the interpolated results if they exists, None otherwise.
        :return data_saved: existing results
        """

        interpolated_result = None

        data_saved = pd.read_csv(result_file)

        # Pre-processing of the dataframe
        values = data_saved.to_numpy()[:, 1 : len(data_saved.to_numpy()[0])]
        labels = data_saved.to_numpy()[:, 0].tolist()
        data_saved = pd.DataFrame(values, index=labels)

        # Look for existing mach or one close enough
        saved_mach_list = data_saved.loc["mach", :].to_numpy().astype(float)
        index_near_mach = np.where(abs(saved_mach_list - mach) < 0.03)[0]
        near_mach = []
        distance_to_mach = []
        # Check if there is a velocity (Mach) value that is close to this one
        for index in index_near_mach:
            if saved_mach_list[index] not in near_mach:
                near_mach.append(saved_mach_list[index])
                distance_to_mach.append(abs(saved_mach_list[index] - mach))
        if not near_mach:
            index_mach = np.where(data_saved.loc["mach", :].to_numpy() == str(mach))[0]
        else:
            selected_mach_index = distance_to_mach.index(min(distance_to_mach))
            index_mach = np.where(saved_mach_list == near_mach[selected_mach_index])[0]
        data_reduced = data_saved.loc[labels, index_mach]

        # Search if this exact reynolds has been computed and save results
        reynolds_vect = data_reduced.loc["reynolds", :].to_numpy().astype(float)

        index_reynolds = index_mach[np.where(reynolds_vect == reynolds)]
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
                alpha_lower = list(
                    string_to_array(lower_values.loc["alpha", index_lower_reynolds].to_numpy()[0])
                )
                alpha_upper = list(
                    string_to_array(upper_values.loc["alpha", index_upper_reynolds].to_numpy()[0])
                )
                alpha_shared = np.array(list(set(alpha_upper).intersection(alpha_lower)))
                interpolated_result.loc["alpha", index_lower_reynolds] = str(alpha_shared.tolist())

                labels.remove("alpha")

                # Calculate average values (cd, cl...) with linear interpolation
                for label in labels:
                    lower_value = string_to_array(
                        lower_values.loc[label, index_lower_reynolds].to_numpy()[0]
                    ).astype(float)
                    upper_value = string_to_array(
                        upper_values.loc[label, index_upper_reynolds].to_numpy()[0]
                    ).astype(float)

                    # If values relative to alpha vector, performs interpolation with shared
                    # vector
                    if np.size(lower_value) == len(alpha_lower):
                        lower_value = np.interp(
                            alpha_shared,
                            np.array(alpha_lower),
                            lower_value,
                        )
                        upper_value = np.interp(
                            alpha_shared,
                            np.array(alpha_upper),
                            upper_value,
                        )

                    value = list(lower_value * x_ratio + upper_value * (1 - x_ratio))

                    interpolated_result.loc[label, index_lower_reynolds] = str(value)

        return interpolated_result, data_saved

    @staticmethod
    def _extract_fix_interpolated_result_length(interpolated_result):
        """
        Format the size of results that need to be passed as array to the size declared to
        OpenMDAO if the original array is bigger we resize it, otherwise we complete with zeros.

        :param interpolated_result: result dataset with labels and values

        :return: length-modified aerodynamic characteristic array
        """

        # Extract results
        cl_max_2d = string_to_array(interpolated_result.loc["cl_max_2d", :].values[0])
        cl_min_2d = string_to_array(interpolated_result.loc["cl_min_2d", :].values[0])
        alpha = string_to_array(interpolated_result.loc["alpha", :].values[0])
        cl = string_to_array(interpolated_result.loc["cl", :].values[0])
        cd = string_to_array(interpolated_result.loc["cd", :].values[0])
        cdp = string_to_array(interpolated_result.loc["cdp", :].values[0])
        cm = string_to_array(interpolated_result.loc["cm", :].values[0])
        cd_min_2d = np.min(cd)

        # Modify vector length if necessary
        if POLAR_POINT_COUNT < len(alpha):
            alpha = np.linspace(alpha[0], alpha[-1], POLAR_POINT_COUNT)
            cl = np.interp(alpha, alpha, cl)
            cd = np.interp(alpha, alpha, cd)
            cdp = np.interp(alpha, alpha, cdp)
            cm = np.interp(alpha, alpha, cm)
        else:
            filler = np.zeros(POLAR_POINT_COUNT - len(alpha))
            alpha = np.append(alpha, filler)
            cl = np.append(cl, filler)
            cd = np.append(cd, filler)
            cdp = np.append(cdp, filler)
            cm = np.append(cm, filler)

        return alpha, cl, cd, cdp, cm, cl_max_2d, cl_min_2d, cd_min_2d

    @staticmethod
    def _fix_calculation_result_length(computed_result):
        """
        Format the size of results that need to be passed as array to the size declared to
        OpenMDAO if the original array is bigger we resize it, otherwise filled with zeros.

        :param interpolated_result: result dataset with labels and values

        :return: length-modified aerodynamic characteristic array
        """

        # Extract results

        alpha = computed_result["AoA"]
        cl = computed_result["CL"]
        cd = computed_result["CD"]
        cm = computed_result["CM"]

        # Modify vector length if necessary
        if POLAR_POINT_COUNT < len(alpha):
            alpha = np.linspace(alpha[0], alpha[-1], POLAR_POINT_COUNT)
            cl = np.interp(alpha, alpha, cl)
            cd = np.interp(alpha, alpha, cd)
            cm = np.interp(alpha, alpha, cm)
        else:
            filler = np.zeros(POLAR_POINT_COUNT - len(alpha))
            alpha = np.append(alpha, filler)
            cl = np.append(cl, filler)
            cd = np.append(cd, filler)
            cm = np.append(cm, filler)

        return alpha, cl, cd, cm
