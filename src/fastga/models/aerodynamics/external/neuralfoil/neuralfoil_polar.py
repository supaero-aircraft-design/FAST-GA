"""
Computation of the airfoil aerodynamic properties using Neuralfoil from :cite:`neuralfoil:2023`.
"""
#  This file is part of FAST-OAD_CS23 : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2025  ONERA & ISAE-SUPAERO
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
import numpy as np
import neuralfoil as nf
from typing import Tuple

from openmdao.components.external_code_comp import ExternalCodeComp
from pathlib import Path
from fastga.models.aerodynamics import airfoil_folder
from ...constants import (
    POLAR_POINT_COUNT,
    OPTION_ALPHA_START,
    OPTION_ALPHA_END,
    OPTION_COMP_NEG_AIR_SYM,
    _DEFAULT_AIRFOIL_FILE,
    ALPHA_STEP,
    DEFAULT_2D_CL_MAX,
    DEFAULT_2D_CL_MIN,
)


_LOGGER = logging.getLogger(__name__)


class NeuralfoilPolar(ExternalCodeComp):
    """Runs a polar computation with Neuralfoil and returns 2D aerodynamic coefficients."""

    def initialize(self):
        self.options.declare("airfoil_folder_path", default=None, types=str, allow_none=True)
        self.options.declare("airfoil_file", default=_DEFAULT_AIRFOIL_FILE, types=str)
        self.options.declare(OPTION_ALPHA_START, default=0.0, types=float)
        self.options.declare(OPTION_ALPHA_END, default=90.0, types=float)
        self.options.declare(OPTION_COMP_NEG_AIR_SYM, default=False, types=bool)
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
        Set up inputs and outputs required for this operation.The pressure drag coefficient ratio
        (CDp) is set to zero in all computations, as NeuralFoil does not provide it, ensuring
        backward compatibility.
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
        mach = round(float(inputs["neuralfoil:mach"]), 4)
        reynolds = round(float(inputs["neuralfoil:reynolds"]))

        # Compressibility correction
        beta = np.sqrt(1.0 - mach**2.0)

        # Modify file type respect to negative AoA/inviscid/single AoA options
        multiple_aoa = not self.options["single_AoA"]
        alpha_start = self.options[OPTION_ALPHA_START]
        alpha_end = self.options[OPTION_ALPHA_END]

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
        _LOGGER.info("Entering Neuralfoil computation")
        results = nf.get_aero_from_dat_file(airfoil_path, alpha=alpha, Re=reynolds)
        # Only apply for Cl and Cm based on the documentation of Neuralfoil
        cl = results["CL"] / (
            beta + mach**2.0 / beta * 0.5 * results["CL"] * (1.0 + 0.2 * mach**2.0)
        )
        cd = results["CD"]
        cm = results["CM"] / (
            beta + mach**2.0 / beta * 0.5 * results["CM"] * (1.0 + 0.2 * mach**2.0)
        )

        if multiple_aoa:
            results["AoA"] = alpha
            cl_max_2d = self._get_max_cl(alpha, cl)
            cl_min_2d = self._get_min_cl(alpha, cl)
            cd_min_2d = np.min(cd)
            alpha, cl, cd, cm = self._fix_calculation_result_length(results)

        # Defining outputs -------------------------------------------------------------------------
        outputs["neuralfoil:alpha"] = alpha
        outputs["neuralfoil:CL"] = cl
        outputs["neuralfoil:CD"] = cd
        outputs["neuralfoil:CM"] = cm
        if multiple_aoa:
            outputs["neuralfoil:CL_max_2D"] = cl_max_2d
            outputs["neuralfoil:CL_min_2D"] = cl_min_2d
            outputs["neuralfoil:CD_min_2D"] = cd_min_2d

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

                def lift_fct(x):
                    return (lift_coeff[1] - lift_coeff[0]) / (alpha[1] - alpha[0]) * (
                        x - alpha[0]
                    ) + lift_coeff[0]

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

                def lift_fct(x):
                    return (lift_coeff[1] - lift_coeff[0]) / (alpha[1] - alpha[0]) * (
                        x - alpha[0]
                    ) + lift_coeff[0]

                delta = np.abs(lift_coeff - lift_fct(alpha))
                return min(lift_coeff[delta <= 0.3]), False

        _LOGGER.warning(
            "2D CL min not found, less than 40%% of angle range computed: using default value %f",
            DEFAULT_2D_CL_MIN,
        )
        return DEFAULT_2D_CL_MIN, True
