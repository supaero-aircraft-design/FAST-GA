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

from ..constants import (
    SUBMODEL_CY_BETA,
    SUBMODEL_CY_BETA_WING,
    SUBMODEL_CY_BETA_VT,
    SUBMODEL_CY_BETA_FUSELAGE,
)


@oad.RegisterSubmodel(SUBMODEL_CY_BETA, "submodel.aerodynamics.aircraft.side_force_beta.legacy")
class ComputeCYBetaAircraft(om.Group):
    """
    Computation of the increase in side force due to a sideslip angle. Assumes the coefficient at
    aircraft level can be obtained by summing the contribution of the individual components. Only
    considers the contribution of wing, VTP and fuselage, does not take into account the effect
    of nacelles or propeller lateral forces.

    Based on :cite:`roskampart6:1985` section 10.2.4.1
    """

    def initialize(self):

        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):

        if self.options["low_speed_aero"]:
            self.add_subsystem(
                "wing_contribution",
                oad.RegisterSubmodel.get_submodel(SUBMODEL_CY_BETA_WING),
                promotes=["*"],
            )
            self.add_subsystem(
                "fuselage_contribution",
                oad.RegisterSubmodel.get_submodel(SUBMODEL_CY_BETA_FUSELAGE),
                promotes=["*"],
            )
        options = {
            "low_speed_aero": self.options["low_speed_aero"],
        }
        self.add_subsystem(
            "vt_contribution",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_CY_BETA_VT, options=options),
            promotes=["*"],
        )
        self.add_subsystem(
            "sum",
            _SumCYBetaContributions(low_speed_aero=self.options["low_speed_aero"]),
            promotes=["*"],
        )


class _SumCYBetaContributions(om.ExplicitComponent):
    """
    Sums the contribution of the various components to the increase in side force due to a
    sideslip angle.
    """

    def initialize(self):

        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):

        self.add_input("data:aerodynamics:fuselage:Cy_beta", val=np.nan, units="rad**-1")
        self.add_input("data:aerodynamics:wing:Cy_beta", val=np.nan, units="rad**-1")

        if self.options["low_speed_aero"]:

            self.add_input(
                "data:aerodynamics:vertical_tail:low_speed:Cy_beta", val=np.nan, units="rad**-1"
            )

            self.add_output("data:aerodynamics:aircraft:low_speed:Cy_beta", units="rad**-1")

        else:

            self.add_input(
                "data:aerodynamics:vertical_tail:cruise:Cy_beta", val=np.nan, units="rad**-1"
            )

            self.add_output("data:aerodynamics:aircraft:cruise:Cy_beta", units="rad**-1")

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        cy_beta_fus = inputs["data:aerodynamics:fuselage:Cy_beta"]
        cy_beta_wing = inputs["data:aerodynamics:wing:Cy_beta"]

        if self.options["low_speed_aero"]:
            outputs["data:aerodynamics:aircraft:low_speed:Cy_beta"] = (
                cy_beta_wing
                + cy_beta_fus
                + inputs["data:aerodynamics:vertical_tail:low_speed:Cy_beta"]
            )
        else:
            outputs["data:aerodynamics:aircraft:cruise:Cy_beta"] = (
                cy_beta_wing
                + cy_beta_fus
                + inputs["data:aerodynamics:vertical_tail:cruise:Cy_beta"]
            )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        if self.options["low_speed_aero"]:
            partials[
                "data:aerodynamics:aircraft:low_speed:Cy_beta",
                "data:aerodynamics:wing:Cy_beta",
            ] = 1.0
            partials[
                "data:aerodynamics:aircraft:low_speed:Cy_beta",
                "data:aerodynamics:fuselage:Cy_beta",
            ] = 1.0
            partials[
                "data:aerodynamics:aircraft:low_speed:Cy_beta",
                "data:aerodynamics:vertical_tail:low_speed:Cy_beta",
            ] = 1.0
        else:
            partials[
                "data:aerodynamics:aircraft:cruise:Cy_beta", "data:aerodynamics:wing:Cy_beta"
            ] = 1.0
            partials[
                "data:aerodynamics:aircraft:cruise:Cy_beta",
                "data:aerodynamics:fuselage:Cy_beta",
            ] = 1.0
            partials[
                "data:aerodynamics:aircraft:cruise:Cy_beta",
                "data:aerodynamics:vertical_tail:cruise:Cy_beta",
            ] = 1.0
