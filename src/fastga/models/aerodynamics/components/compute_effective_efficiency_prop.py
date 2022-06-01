"""Estimation of effective propeller efficiency due to additional cowling drag."""
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

import warnings

import numpy as np
import openmdao.api as om
import fastoad.api as oad
from stdatm import Atmosphere

from ..constants import SUBMODEL_EFFECTIVE_EFFICIENCY_PROPELLER


@oad.RegisterSubmodel(
    SUBMODEL_EFFECTIVE_EFFICIENCY_PROPELLER,
    "fastga.submodel.aerodynamics.propeller.effective_efficiency.legacy",
)
class ComputeEffectiveEfficiencyPropeller(om.ExplicitComponent):
    """
    Estimation of the increase in cowling profile drag due to the effect of the propeller
    slipstream. It will be modelled as a decrease in efficiency.
    """

    def initialize(self):
        """Declaring the low_speed_aero options so we can use low speed and cruise conditions."""
        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):

        self.add_input("data:geometry:propulsion:engine:layout", val=np.nan)
        self.add_input("data:geometry:propeller:diameter", val=np.nan, units="m")
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")

        self.add_input("data:geometry:fuselage:wet_area", val=np.nan, units="m**2")
        self.add_input("data:geometry:propulsion:nacelle:wet_area", val=np.nan, units="m**2")

        if self.options["low_speed_aero"]:
            self.add_input("data:aerodynamics:nacelles:low_speed:CD0", val=np.nan)
            self.add_input("data:aerodynamics:fuselage:low_speed:CD0", val=np.nan)
            self.add_output(
                "data:aerodynamics:propeller:installation_effect:effective_efficiency:low_speed",
                val=1.0,
                desc="Value to multiply the uninstalled efficiency with to obtain the effective "
                "efficiency due to the presence of cowling (fuselage or nacelle) behind the "
                "propeller",
            )
        else:
            self.add_input("data:aerodynamics:nacelles:cruise:CD0", val=np.nan)
            self.add_input("data:aerodynamics:fuselage:cruise:CD0", val=np.nan)
            self.add_input("data:mission:sizing:main_route:cruise:altitude", val=np.nan, units="m")
            self.add_output(
                "data:aerodynamics:propeller:installation_effect:effective_efficiency:cruise",
                val=1.0,
                desc="Value to multiply the uninstalled efficiency with to obtain the effective "
                "efficiency due to the presence of cowling (fuselage or nacelle) behind the "
                "propeller",
            )

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        propeller_diameter = inputs["data:geometry:propeller:diameter"]
        wing_area = inputs["data:geometry:wing:area"]
        engine_layout = inputs["data:geometry:propulsion:engine:layout"]

        if self.options["low_speed_aero"]:
            altitude = 0.0
            if engine_layout == 3.0:
                wet_area_cowling = inputs["data:geometry:fuselage:wet_area"]
                friction_drag_coeff = inputs["data:aerodynamics:fuselage:low_speed:CD0"]
            elif engine_layout == 1.0 or engine_layout == 2.0:
                wet_area_cowling = inputs["data:geometry:propulsion:nacelle:wet_area"]
                friction_drag_coeff = inputs["data:aerodynamics:nacelles:low_speed:CD0"]
            else:
                wet_area_cowling = inputs["data:geometry:fuselage:wet_area"]
                friction_drag_coeff = inputs["data:aerodynamics:fuselage:low_speed:CD0"]
                warnings.warn(
                    "Propulsion layout {} not implemented in model, replaced by layout 3!".format(
                        engine_layout
                    )
                )
        else:
            altitude = inputs["data:mission:sizing:main_route:cruise:altitude"]
            if engine_layout == 3.0:
                wet_area_cowling = inputs["data:geometry:fuselage:wet_area"]
                friction_drag_coeff = inputs["data:aerodynamics:fuselage:cruise:CD0"]
            elif engine_layout == 1.0 or engine_layout == 2.0:
                wet_area_cowling = inputs["data:geometry:propulsion:nacelle:wet_area"]
                friction_drag_coeff = inputs["data:aerodynamics:nacelles:cruise:CD0"]
            else:
                wet_area_cowling = inputs["data:geometry:fuselage:wet_area"]
                friction_drag_coeff = inputs["data:aerodynamics:fuselage:cruise:CD0"]
                warnings.warn(
                    "Propulsion layout {} not implemented in model, replaced by layout 3!".format(
                        engine_layout
                    )
                )

        # All drag coefficient are given wrt the wing area but for this formula we need to have
        # this coefficient with respect to the cowling wet area
        friction_drag_coeff *= wing_area / wet_area_cowling

        density = Atmosphere(altitude, altitude_in_feet=False).density
        density_sl = Atmosphere(0.0, altitude_in_feet=False).density

        effective_efficiency_factor = (
            1.0
            - 1.558
            / propeller_diameter ** 2.0
            * density
            / density_sl
            * friction_drag_coeff
            * wet_area_cowling
        )

        if self.options["low_speed_aero"]:
            outputs[
                "data:aerodynamics:propeller:installation_effect:effective_efficiency:low_speed"
            ] = effective_efficiency_factor
        else:
            outputs[
                "data:aerodynamics:propeller:installation_effect:effective_efficiency:cruise"
            ] = effective_efficiency_factor
