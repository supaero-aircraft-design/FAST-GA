"""
Python module for flight control system weight calculation, part of the airframe mass computation.
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

import fastoad.api as oad
import numpy as np
import openmdao.api as om
from stdatm import AtmosphereWithPartials

from .constants import (
    SERVICE_FLIGHT_CONTROLS_MASS,
    SUBMODEL_FLIGHT_CONTROLS_MASS_LEGACY,
    SUBMODEL_FLIGHT_CONTROLS_MASS_FLOPS,
)

oad.RegisterSubmodel.active_models[SERVICE_FLIGHT_CONTROLS_MASS] = (
    SUBMODEL_FLIGHT_CONTROLS_MASS_LEGACY
)


@oad.RegisterSubmodel(SERVICE_FLIGHT_CONTROLS_MASS, SUBMODEL_FLIGHT_CONTROLS_MASS_LEGACY)
class ComputeFlightControlsWeight(om.ExplicitComponent):
    """
    Flight controls weight estimation

    Based on a statistical analysis. See :cite:`raymer:2012` but can also be found in
    :cite:`gudmundsson:2013`
    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="lb")
        self.add_input("data:mission:sizing:cs23:sizing_factor:ultimate_aircraft", val=np.nan)
        self.add_input("data:geometry:wing:span", val=np.nan, units="ft")
        self.add_input("data:geometry:fuselage:length", val=np.nan, units="ft")

        self.add_output("data:weight:airframe:flight_controls:mass", units="lb")

        self.declare_partials("*", "*", method="exact")

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        mtow = inputs["data:weight:aircraft:MTOW"]
        n_ult = inputs["data:mission:sizing:cs23:sizing_factor:ultimate_aircraft"]
        span = inputs["data:geometry:wing:span"]
        fus_length = inputs["data:geometry:fuselage:length"]

        a4 = 0.053 * (fus_length**1.536 * span**0.371 * (n_ult * mtow * 1e-4) ** 0.80)
        # mass formula in lb

        outputs["data:weight:airframe:flight_controls:mass"] = a4

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        mtow = inputs["data:weight:aircraft:MTOW"]
        n_ult = inputs["data:mission:sizing:cs23:sizing_factor:ultimate_aircraft"]
        span = inputs["data:geometry:wing:span"]
        fus_length = inputs["data:geometry:fuselage:length"]

        partials["data:weight:airframe:flight_controls:mass", "data:geometry:fuselage:length"] = (
            0.081408 * fus_length**0.536 * span**0.371 * (1.0e-4 * mtow * n_ult) ** 0.8
        )
        partials["data:weight:airframe:flight_controls:mass", "data:geometry:wing:span"] = (
            0.019663 * fus_length**1.536 * (1.0e-4 * mtow * n_ult) ** 0.8
        ) / span**0.629
        partials["data:weight:airframe:flight_controls:mass", "data:weight:aircraft:MTOW"] = (
            4.24e-6 * fus_length**1.536 * n_ult * span**0.371
        ) / (1.0e-4 * mtow * n_ult) ** 0.2
        partials[
            "data:weight:airframe:flight_controls:mass",
            "data:mission:sizing:cs23:sizing_factor:ultimate_aircraft",
        ] = (4.24e-6 * fus_length**1.536 * mtow * span**0.371) / (1.0e-4 * mtow * n_ult) ** 0.2


@oad.RegisterSubmodel(SERVICE_FLIGHT_CONTROLS_MASS, SUBMODEL_FLIGHT_CONTROLS_MASS_FLOPS)
class ComputeFlightControlsWeightFLOPS(om.ExplicitComponent):
    """
    Flight controls weight estimation.

    Based on a statistical analysis. See :cite:`wells:2017`.
    """

    # pylint: disable=missing-function-docstring
    # Overriding OpenMDAO setup
    def setup(self):
        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="lb")
        self.add_input("data:mission:sizing:cs23:sizing_factor:ultimate_aircraft", val=np.nan)
        self.add_input("data:mission:sizing:cs23:characteristic_speed:vd", val=np.nan, units="m/s")
        self.add_input("data:geometry:wing:area", val=np.nan, units="ft**2")

        self.add_output("data:weight:airframe:flight_controls:mass", units="lb")

        self.declare_partials("*", "*", method="exact")

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute, not all arguments are used
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        mtow = inputs["data:weight:aircraft:MTOW"]
        n_ult = inputs["data:mission:sizing:cs23:sizing_factor:ultimate_aircraft"]
        v_d = inputs["data:mission:sizing:cs23:characteristic_speed:vd"]
        wing_area = inputs["data:geometry:wing:area"]

        atm = AtmosphereWithPartials(altitude=0.0, altitude_in_feet=True)
        atm.equivalent_airspeed = v_d
        dynamic_pressure = 1.0 / 2.0 * atm.density * atm.true_airspeed**2.0 * 0.0208854
        # In lb/ft2

        a4 = (
            0.404
            * wing_area**0.317
            * (mtow / 1000.0) ** 0.602
            * n_ult**0.525
            * dynamic_pressure**0.345
        )
        # mass formula in lb

        outputs["data:weight:airframe:flight_controls:mass"] = a4

    # pylint: disable=missing-function-docstring, unused-argument
    # Overriding OpenMDAO compute_partials, not all arguments are used
    def compute_partials(self, inputs, partials, discrete_inputs=None):
        mtow = inputs["data:weight:aircraft:MTOW"]
        n_ult = inputs["data:mission:sizing:cs23:sizing_factor:ultimate_aircraft"]
        v_d = inputs["data:mission:sizing:cs23:characteristic_speed:vd"]
        wing_area = inputs["data:geometry:wing:area"]

        atm = AtmosphereWithPartials(altitude=0.0, altitude_in_feet=True)
        atm.equivalent_airspeed = v_d
        dynamic_pressure = 1.0 / 2.0 * atm.density * atm.true_airspeed**2.0 * 0.0208854

        partials["data:weight:airframe:flight_controls:mass", "data:weight:aircraft:MTOW"] = (
            0.404
            * wing_area**0.317
            * 0.001**0.602
            * 0.602
            * mtow ** (-0.398)
            * n_ult**0.525
            * dynamic_pressure**0.345
        )
        partials[
            "data:weight:airframe:flight_controls:mass",
            "data:mission:sizing:cs23:sizing_factor:ultimate_aircraft",
        ] = (
            0.2121 * dynamic_pressure**0.345 * wing_area**0.317 * (0.001 * mtow) ** 0.602
        ) / n_ult**0.475
        partials["data:weight:airframe:flight_controls:mass", "data:geometry:wing:area"] = (
            0.404
            * 0.317
            * wing_area ** (-0.683)
            * (mtow / 1000.0) ** 0.602
            * n_ult**0.525
            * dynamic_pressure**0.345
        )
        partials[
            "data:weight:airframe:flight_controls:mass",
            "data:mission:sizing:cs23:characteristic_speed:vd",
        ] = (
            0.13938
            * n_ult**0.525
            * wing_area**0.317
            * (0.001 * mtow) ** 0.602
            * dynamic_pressure ** (-0.655)
            * atm.density
            * v_d
            * 0.0208854
        )
