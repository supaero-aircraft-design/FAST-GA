"""Computation of lift and drag increment due to high-lift devices."""
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

import math
from typing import Union

import numpy as np
import fastoad.api as oad

from .figure_digitization import FigureDigitization
from ..constants import SUBMODEL_DELTA_ELEVATOR


@oad.RegisterSubmodel(SUBMODEL_DELTA_ELEVATOR, "fastga.submodel.aerodynamics.elevator.delta.legacy")
class ComputeDeltaElevator(FigureDigitization):
    """
    Provides lift and drag increments due to high-lift devices.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.phase = None

    def setup(self):

        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:horizontal_tail:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:horizontal_tail:sweep_25", val=np.nan, units="rad")
        self.add_input("data:geometry:horizontal_tail:elevator_chord_ratio", val=np.nan)
        self.add_input("data:geometry:horizontal_tail:thickness_ratio", val=np.nan)
        self.add_input("data:aerodynamics:low_speed:mach", val=np.nan)
        self.add_input(
            "data:aerodynamics:horizontal_tail:airfoil:CL_alpha", val=np.nan, units="rad**-1"
        )
        self.add_input("data:mission:sizing:landing:elevator_angle", val=np.nan, units="deg")

        self.add_output("data:aerodynamics:elevator:low_speed:CL_delta", units="rad**-1")
        self.add_output("data:aerodynamics:elevator:low_speed:CD_delta", units="rad**-2")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        wing_area = inputs["data:geometry:wing:area"]
        htp_area = inputs["data:geometry:horizontal_tail:area"]

        # Computes elevator contribution during low speed operations (for different deflection
        # angle)
        outputs["data:aerodynamics:elevator:low_speed:CL_delta"] = self._get_elevator_delta_cl(
            inputs,
            25.0,
        )  # get derivative for 25° angle assuming it is linear when <= to 25 degree,
        # derivative wrt to the wing, multiplies the deflection angle squared
        outputs["data:aerodynamics:elevator:low_speed:CD_delta"] = (
            self.delta_cd_plain_flap(
                inputs["data:geometry:horizontal_tail:elevator_chord_ratio"],
                abs(inputs["data:mission:sizing:landing:elevator_angle"]),
            )
            / (abs(inputs["data:mission:sizing:landing:elevator_angle"]) * math.pi / 180.0) ** 2.0
            * math.cos(inputs["data:geometry:horizontal_tail:sweep_25"])
            * htp_area
            / wing_area
        )

    def _get_elevator_delta_cl(
        self, inputs, elevator_angle: Union[float, np.array]
    ) -> Union[float, np.array]:
        """
        Computes the elevator lift increment as a plain flap following the method presented in
        Roskam part 6, section 8.1.2.1.a.

        :param elevator_angle: elevator angle (in Degree).
        :return: lift coefficient derivative.
        """

        ht_area = inputs["data:geometry:horizontal_tail:area"]
        wing_area = inputs["data:geometry:wing:area"]
        elevator_chord_ratio = inputs["data:geometry:horizontal_tail:elevator_chord_ratio"]
        htp_thickness_ratio = inputs["data:geometry:horizontal_tail:thickness_ratio"]
        cl_alpha_airfoil_ht = inputs["data:aerodynamics:horizontal_tail:airfoil:CL_alpha"]

        # Elevator (plain flap). Default: maximum deflection (25deg)
        cl_delta_theory = self.cl_delta_theory_plain_flap(htp_thickness_ratio, elevator_chord_ratio)
        k = self.k_prime_plain_flap(abs(elevator_angle), elevator_chord_ratio)
        k_cl_delta = self.k_cl_delta_plain_flap(
            htp_thickness_ratio, cl_alpha_airfoil_ht, elevator_chord_ratio
        )
        cl_alpha_elev = (cl_delta_theory * k * k_cl_delta) * ht_area / wing_area
        cl_alpha_elev *= 0.9  # Correction for the central fuselage part (no elevator there)

        return cl_alpha_elev
