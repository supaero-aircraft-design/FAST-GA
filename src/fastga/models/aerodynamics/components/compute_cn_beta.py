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
    SUBMODEL_CN_BETA,
    SUBMODEL_CN_BETA_VT,
    SUBMODEL_CN_BETA_FUSELAGE,
)


@oad.RegisterSubmodel(SUBMODEL_CN_BETA, "submodel.aerodynamics.aircraft.yawing_moment_beta.legacy")
class ComputeCnBetaAircraft(om.Group):
    """
    Computation of the increase in yawing moment due to a sideslip angle. Assumes the coefficient
    at aircraft level can be obtained by summing the contribution of the individual components (
    fuselage and VT). Wing contribution is negligible up until high angles of attack. The
    convention from :cite:`roskampart6:1985` are used, meaning that for lateral derivative,
    the reference length is the wing span. Does not take into account the effect
    of nacelles or propeller lateral forces.

    Based on :cite:`roskampart6:1985` section 10.2.4.1
    """

    def initialize(self):

        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):

        if self.options["low_speed_aero"]:
            self.add_subsystem(
                "fuselage_contribution",
                oad.RegisterSubmodel.get_submodel(SUBMODEL_CN_BETA_FUSELAGE),
                promotes=["*"],
            )
        options = {
            "low_speed_aero": self.options["low_speed_aero"],
        }
        self.add_subsystem(
            "vt_contribution",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_CN_BETA_VT, options=options),
            promotes=["*"],
        )
        self.add_subsystem(
            "sum",
            _SumCNBetaContributions(low_speed_aero=self.options["low_speed_aero"]),
            promotes=["*"],
        )


class _SumCNBetaContributions(om.ExplicitComponent):
    """
    Sums the contribution of the various components to the increase in yawing moment due to a
    sideslip angle.
    """

    def initialize(self):

        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):

        self.add_input("data:aerodynamics:fuselage:Cn_beta", val=np.nan, units="rad**-1")

        if self.options["low_speed_aero"]:

            self.add_input(
                "data:aerodynamics:vertical_tail:low_speed:Cn_beta", val=np.nan, units="rad**-1"
            )

            self.add_output("data:aerodynamics:aircraft:low_speed:Cn_beta", units="rad**-1")

        else:

            self.add_input(
                "data:aerodynamics:vertical_tail:cruise:Cn_beta", val=np.nan, units="rad**-1"
            )

            self.add_output("data:aerodynamics:aircraft:cruise:Cn_beta", units="rad**-1")

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        cy_beta_fus = inputs["data:aerodynamics:fuselage:Cn_beta"]

        if self.options["low_speed_aero"]:
            outputs["data:aerodynamics:aircraft:low_speed:Cn_beta"] = (
                +cy_beta_fus + inputs["data:aerodynamics:vertical_tail:low_speed:Cn_beta"]
            )
        else:
            outputs["data:aerodynamics:aircraft:cruise:Cn_beta"] = (
                +cy_beta_fus + inputs["data:aerodynamics:vertical_tail:cruise:Cn_beta"]
            )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        if self.options["low_speed_aero"]:
            partials[
                "data:aerodynamics:aircraft:low_speed:Cn_beta",
                "data:aerodynamics:fuselage:Cn_beta",
            ] = 1.0
            partials[
                "data:aerodynamics:aircraft:low_speed:Cn_beta",
                "data:aerodynamics:vertical_tail:low_speed:Cn_beta",
            ] = 1.0
        else:
            partials[
                "data:aerodynamics:aircraft:cruise:Cn_beta",
                "data:aerodynamics:fuselage:Cn_beta",
            ] = 1.0
            partials[
                "data:aerodynamics:aircraft:cruise:Cn_beta",
                "data:aerodynamics:vertical_tail:cruise:Cn_beta",
            ] = 1.0
