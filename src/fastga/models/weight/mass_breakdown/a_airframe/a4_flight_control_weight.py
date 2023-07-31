"""Estimation of flight controls weight."""
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
import openmdao.api as om

import fastoad.api as oad

from .constants import SUBMODEL_FLIGHT_CONTROLS_MASS

oad.RegisterSubmodel.active_models[
    SUBMODEL_FLIGHT_CONTROLS_MASS
] = "fastga.submodel.weight.mass.airframe.flight_controls.legacy"


@oad.RegisterSubmodel(
    SUBMODEL_FLIGHT_CONTROLS_MASS, "fastga.submodel.weight.mass.airframe.flight_controls.legacy"
)
class ComputeFlightControlsWeight(om.ExplicitComponent):
    """
    Flight controls weight estimation

    Based on a statistical analysis. See :cite:`raymer:2012` but can also be found in
    :cite:`gudmundsson:2013`
    """

    def setup(self):

        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="lb")
        self.add_input("data:mission:sizing:cs23:sizing_factor:ultimate_aircraft", val=np.nan)
        self.add_input("data:geometry:wing:span", val=np.nan, units="ft")
        self.add_input("data:geometry:fuselage:length", val=np.nan, units="ft")

        self.add_output("data:weight:airframe:flight_controls:mass", units="lb")

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        mtow = inputs["data:weight:aircraft:MTOW"]
        n_ult = inputs["data:mission:sizing:cs23:sizing_factor:ultimate_aircraft"]
        span = inputs["data:geometry:wing:span"]
        fus_length = inputs["data:geometry:fuselage:length"]

        a4 = 0.053 * (fus_length ** 1.536 * span ** 0.371 * (n_ult * mtow * 1e-4) ** 0.80)
        # mass formula in lb

        outputs["data:weight:airframe:flight_controls:mass"] = a4

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        mtow = inputs["data:weight:aircraft:MTOW"]
        n_ult = inputs["data:mission:sizing:cs23:sizing_factor:ultimate_aircraft"]
        span = inputs["data:geometry:wing:span"]
        fus_length = inputs["data:geometry:fuselage:length"]

        partials["data:weight:airframe:flight_controls:mass", "data:geometry:fuselage:length"] = (
            0.081408 * fus_length ** 0.536 * span ** 0.371 * (1.0e-4 * mtow * n_ult) ** 0.8
        )
        partials["data:weight:airframe:flight_controls:mass", "data:geometry:wing:span"] = (
            0.019663 * fus_length ** 1.536 * (1.0e-4 * mtow * n_ult) ** 0.8
        ) / span ** 0.629
        partials["data:weight:airframe:flight_controls:mass", "data:weight:aircraft:MTOW"] = (
            4.24e-6 * fus_length ** 1.536 * n_ult * span ** 0.371
        ) / (1.0e-4 * mtow * n_ult) ** 0.2
        partials[
            "data:weight:airframe:flight_controls:mass",
            "data:mission:sizing:cs23:sizing_factor:ultimate_aircraft",
        ] = (4.24e-6 * fus_length ** 1.536 * mtow * span ** 0.371) / (1.0e-4 * mtow * n_ult) ** 0.2
