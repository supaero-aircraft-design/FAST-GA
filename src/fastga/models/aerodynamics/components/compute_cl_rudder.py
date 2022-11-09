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

from ..constants import SUBMODEL_CL_RUDDER


@oad.RegisterSubmodel(SUBMODEL_CL_RUDDER, "fastga.submodel.aerodynamics.rudder.roll_moment.legacy")
class ComputeClDeltaRudder(om.ExplicitComponent):
    """
    Class to compute the contribution of the rudder to the roll moment coefficient. Depends on
    the angle of attack, so the same remark as in ..compute_cy_yaw_rate.py holds. The convention
    from :cite:`roskampart6:1985` are used, meaning that for lateral derivative, the reference
    length is the wing span.

    Based on :cite:`roskampart6:1985` section 10.3.8
    """

    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):

        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:root:z", units="m", val=np.nan)
        self.add_input(
            "data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25", units="m", val=np.nan
        )
        self.add_input("data:geometry:vertical_tail:MAC:z", units="m", val=np.nan)
        self.add_input("data:geometry:vertical_tail:area", units="m**2", val=np.nan)
        self.add_input("data:geometry:fuselage:maximum_height", units="m", val=np.nan)

        if self.options["low_speed_aero"]:
            self.add_input(
                "settings:aerodynamics:reference_flight_conditions:low_speed:AOA",
                units="rad",
                val=5.0 * np.pi / 180.0,
            )
            self.add_input(
                "data:aerodynamics:rudder:low_speed:Cy_delta_r", val=np.nan, units="rad**-1"
            )

            self.add_output("data:aerodynamics:rudder:low_speed:Cl_delta_r", units="rad**-1")
        else:
            self.add_input(
                "settings:aerodynamics:reference_flight_conditions:cruise:AOA",
                units="rad",
                val=1.0 * np.pi / 180.0,
            )
            self.add_input(
                "data:aerodynamics:rudder:cruise:Cy_delta_r", val=np.nan, units="rad**-1"
            )
            self.add_output("data:aerodynamics:rudder:cruise:Cl_delta_r", units="rad**-1")

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        z_v = (
            inputs["data:geometry:wing:root:z"]
            + 0.5 * inputs["data:geometry:fuselage:maximum_height"]
            + inputs["data:geometry:vertical_tail:MAC:z"]
        )
        lp_vt = inputs["data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25"]
        wing_span = inputs["data:geometry:wing:span"]
        wing_area = inputs["data:geometry:wing:area"]
        vt_area = inputs["data:geometry:vertical_tail:area"]

        if self.options["low_speed_aero"]:
            aoa_ref = inputs["settings:aerodynamics:reference_flight_conditions:low_speed:AOA"]
            # Change of reference surface area
            cy_delta_r = (
                inputs["data:aerodynamics:rudder:low_speed:Cy_delta_r"] * vt_area / wing_area
            )
            outputs["data:aerodynamics:rudder:low_speed:Cl_delta_r"] = (
                cy_delta_r * (z_v * np.cos(aoa_ref) - lp_vt * np.sin(aoa_ref)) / wing_span
            )
        else:
            aoa_ref = inputs["settings:aerodynamics:reference_flight_conditions:cruise:AOA"]
            # Change of reference surface area
            cy_delta_r = inputs["data:aerodynamics:rudder:cruise:Cy_delta_r"] * vt_area / wing_area
            outputs["data:aerodynamics:rudder:cruise:Cl_delta_r"] = (
                cy_delta_r * (z_v * np.cos(aoa_ref) - lp_vt * np.sin(aoa_ref)) / wing_span
            )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        z_v = (
            inputs["data:geometry:wing:root:z"]
            + 0.5 * inputs["data:geometry:fuselage:maximum_height"]
            + inputs["data:geometry:vertical_tail:MAC:z"]
        )
        lp_vt = inputs["data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25"]
        wing_span = inputs["data:geometry:wing:span"]
        wing_area = inputs["data:geometry:wing:area"]
        vt_area = inputs["data:geometry:vertical_tail:area"]

        if self.options["low_speed_aero"]:
            aoa_ref = inputs["settings:aerodynamics:reference_flight_conditions:low_speed:AOA"]
            cy_delta_r = inputs["data:aerodynamics:rudder:low_speed:Cy_delta_r"]

            partials[
                "data:aerodynamics:rudder:low_speed:Cl_delta_r",
                "data:aerodynamics:rudder:low_speed:Cy_delta_r",
            ] = (
                (z_v * np.cos(aoa_ref) - lp_vt * np.sin(aoa_ref)) / wing_span * vt_area / wing_area
            )
            partials[
                "data:aerodynamics:rudder:low_speed:Cl_delta_r",
                "data:geometry:wing:root:z",
            ] = (
                cy_delta_r * np.cos(aoa_ref) / wing_span * vt_area / wing_area
            )
            partials[
                "data:aerodynamics:rudder:low_speed:Cl_delta_r",
                "data:geometry:fuselage:maximum_height",
            ] = (
                0.5 * cy_delta_r * np.cos(aoa_ref) / wing_span * vt_area / wing_area
            )
            partials[
                "data:aerodynamics:rudder:low_speed:Cl_delta_r",
                "data:geometry:vertical_tail:MAC:z",
            ] = (
                cy_delta_r * np.cos(aoa_ref) / wing_span * vt_area / wing_area
            )
            partials[
                "data:aerodynamics:rudder:low_speed:Cl_delta_r",
                "data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25",
            ] = (
                -cy_delta_r * np.sin(aoa_ref) / wing_span * vt_area / wing_area
            )
            partials[
                "data:aerodynamics:rudder:low_speed:Cl_delta_r",
                "data:geometry:vertical_tail:area",
            ] = (
                cy_delta_r
                * (z_v * np.cos(aoa_ref) - lp_vt * np.sin(aoa_ref))
                / wing_span
                / wing_area
            )
            partials[
                "data:aerodynamics:rudder:low_speed:Cl_delta_r",
                "data:geometry:wing:span",
            ] = (
                -cy_delta_r
                * (z_v * np.cos(aoa_ref) - lp_vt * np.sin(aoa_ref))
                / wing_span ** 2.0
                * vt_area
                / wing_area
            )
            partials[
                "data:aerodynamics:rudder:low_speed:Cl_delta_r",
                "data:geometry:wing:area",
            ] = (
                -cy_delta_r
                * (z_v * np.cos(aoa_ref) - lp_vt * np.sin(aoa_ref))
                / wing_span
                * vt_area
                / wing_area ** 2.0
            )
            partials[
                "data:aerodynamics:rudder:low_speed:Cl_delta_r",
                "settings:aerodynamics:reference_flight_conditions:low_speed:AOA",
            ] = (
                -cy_delta_r
                * (z_v * np.sin(aoa_ref) + lp_vt * np.cos(aoa_ref))
                / wing_span
                * vt_area
                / wing_area
            )

        else:
            aoa_ref = inputs["settings:aerodynamics:reference_flight_conditions:cruise:AOA"]
            cy_delta_r = inputs["data:aerodynamics:rudder:cruise:Cy_delta_r"]

            partials[
                "data:aerodynamics:rudder:cruise:Cl_delta_r",
                "data:aerodynamics:rudder:cruise:Cy_delta_r",
            ] = (
                (z_v * np.cos(aoa_ref) - lp_vt * np.sin(aoa_ref)) / wing_span * vt_area / wing_area
            )
            partials["data:aerodynamics:rudder:cruise:Cl_delta_r", "data:geometry:wing:root:z"] = (
                (cy_delta_r * np.cos(aoa_ref) / wing_span) * vt_area / wing_area
            )
            partials[
                "data:aerodynamics:rudder:cruise:Cl_delta_r",
                "data:geometry:fuselage:maximum_height",
            ] = (
                0.5 * cy_delta_r * np.cos(aoa_ref) / wing_span * vt_area / wing_area
            )
            partials[
                "data:aerodynamics:rudder:cruise:Cl_delta_r",
                "data:geometry:vertical_tail:MAC:z",
            ] = (
                cy_delta_r * np.cos(aoa_ref) / wing_span * vt_area / wing_area
            )
            partials[
                "data:aerodynamics:rudder:cruise:Cl_delta_r",
                "data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25",
            ] = (
                -cy_delta_r * np.sin(aoa_ref) / wing_span * vt_area / wing_area
            )
            partials[
                "data:aerodynamics:rudder:cruise:Cl_delta_r",
                "data:geometry:vertical_tail:area",
            ] = (
                cy_delta_r
                * (z_v * np.cos(aoa_ref) - lp_vt * np.sin(aoa_ref))
                / wing_span
                / wing_area
            )
            partials["data:aerodynamics:rudder:cruise:Cl_delta_r", "data:geometry:wing:span"] = (
                (-cy_delta_r * (z_v * np.cos(aoa_ref) - lp_vt * np.sin(aoa_ref)) / wing_span ** 2.0)
                * vt_area
                / wing_area
            )
            partials["data:aerodynamics:rudder:cruise:Cl_delta_r", "data:geometry:wing:area"] = (
                -cy_delta_r
                * (z_v * np.cos(aoa_ref) - lp_vt * np.sin(aoa_ref))
                / wing_span
                * vt_area
                / wing_area ** 2.0
            )
            partials[
                "data:aerodynamics:rudder:cruise:Cl_delta_r",
                "settings:aerodynamics:reference_flight_conditions:cruise:AOA",
            ] = (
                (-cy_delta_r * (z_v * np.sin(aoa_ref) + lp_vt * np.cos(aoa_ref)) / wing_span)
                * vt_area
                / wing_area
            )
