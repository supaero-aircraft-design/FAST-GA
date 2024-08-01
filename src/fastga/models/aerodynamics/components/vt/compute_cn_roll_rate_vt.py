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

from ...constants import SUBMODEL_CN_P_VT


@oad.RegisterSubmodel(
    SUBMODEL_CN_P_VT, "fastga.submodel.aerodynamics.vertical_tail.yaw_moment_roll_rate.legacy"
)
class ComputeCnRollRateVerticalTail(om.ExplicitComponent):
    """
    Class to compute the contribution of the vertical_tail to the yaw moment coefficient due to
    roll rate. Depends on the reference angle of attack, so the same remark as in
    ..compute_cy_yaw_rate.py holds. The convention from :cite:`roskampart6:1985` are used,
    meaning that for lateral derivative, the reference length is the wing span. Another important
    point is that, for the derivative with respect to yaw and roll, the rotation speed are made
    dimensionless by multiplying them by the wing span and dividing them by 2 times the airspeed.

    Based on :cite:`roskampart6:1985` section 10.2.8
    """

    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:z", units="m", val=np.nan)
        self.add_input(
            "data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25", units="m", val=np.nan
        )
        self.add_input("data:geometry:vertical_tail:MAC:z", units="m", val=np.nan)
        self.add_input("data:geometry:fuselage:maximum_height", units="m", val=np.nan)

        if self.options["low_speed_aero"]:
            self.add_input(
                "settings:aerodynamics:reference_flight_conditions:low_speed:AOA",
                units="rad",
                val=5.0 * np.pi / 180.0,
            )
            self.add_input(
                "data:aerodynamics:vertical_tail:low_speed:Cy_beta", val=np.nan, units="rad**-1"
            )

            self.add_output("data:aerodynamics:vertical_tail:low_speed:Cn_p", units="rad**-1")
        else:
            self.add_input(
                "settings:aerodynamics:reference_flight_conditions:cruise:AOA",
                units="rad",
                val=1.0 * np.pi / 180.0,
            )
            self.add_input(
                "data:aerodynamics:vertical_tail:cruise:Cy_beta", val=np.nan, units="rad**-1"
            )
            self.add_output("data:aerodynamics:vertical_tail:cruise:Cn_p", units="rad**-1")

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        z_v = (
            inputs["data:geometry:wing:root:z"]
            + 0.5 * inputs["data:geometry:fuselage:maximum_height"]
            + inputs["data:geometry:vertical_tail:MAC:z"]
        )
        lp_vt = inputs["data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25"]
        wing_span = inputs["data:geometry:wing:span"]

        if self.options["low_speed_aero"]:
            aoa_ref = inputs["settings:aerodynamics:reference_flight_conditions:low_speed:AOA"]
            cy_beta_vt = inputs["data:aerodynamics:vertical_tail:low_speed:Cy_beta"]
            outputs["data:aerodynamics:vertical_tail:low_speed:Cn_p"] = (
                -2.0
                * cy_beta_vt
                * (lp_vt * np.cos(aoa_ref) + z_v * np.sin(aoa_ref))
                * (z_v * np.cos(aoa_ref) - lp_vt * np.sin(aoa_ref) - z_v)
                / wing_span**2.0
            )
        else:
            aoa_ref = inputs["settings:aerodynamics:reference_flight_conditions:cruise:AOA"]
            cy_beta_vt = inputs["data:aerodynamics:vertical_tail:cruise:Cy_beta"]
            outputs["data:aerodynamics:vertical_tail:cruise:Cn_p"] = (
                -2.0
                * cy_beta_vt
                * (lp_vt * np.cos(aoa_ref) + z_v * np.sin(aoa_ref))
                * (z_v * np.cos(aoa_ref) - lp_vt * np.sin(aoa_ref) - z_v)
                / wing_span**2.0
            )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        z_v = (
            inputs["data:geometry:wing:root:z"]
            + 0.5 * inputs["data:geometry:fuselage:maximum_height"]
            + inputs["data:geometry:vertical_tail:MAC:z"]
        )
        lp_vt = inputs["data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25"]
        wing_span = inputs["data:geometry:wing:span"]

        if self.options["low_speed_aero"]:
            aoa_ref = inputs["settings:aerodynamics:reference_flight_conditions:low_speed:AOA"]
            cy_beta_vt = inputs["data:aerodynamics:vertical_tail:low_speed:Cy_beta"]

            partials[
                "data:aerodynamics:vertical_tail:low_speed:Cn_p",
                "data:aerodynamics:vertical_tail:low_speed:Cy_beta",
            ] = (
                -2.0
                * (lp_vt * np.cos(aoa_ref) + z_v * np.sin(aoa_ref))
                * (z_v * np.cos(aoa_ref) - lp_vt * np.sin(aoa_ref) - z_v)
                / wing_span**2.0
            )
            partials[
                "data:aerodynamics:vertical_tail:low_speed:Cn_p",
                "data:geometry:wing:root:z",
            ] = (
                -2.0
                * cy_beta_vt
                * (
                    lp_vt * np.cos(aoa_ref) ** 2.0
                    - lp_vt * np.cos(aoa_ref)
                    + 2.0 * z_v * np.cos(aoa_ref) * np.sin(aoa_ref)
                    - lp_vt * np.sin(aoa_ref) ** 2.0
                    - 2.0 * z_v * np.sin(aoa_ref)
                )
                / wing_span**2.0
            )
            partials[
                "data:aerodynamics:vertical_tail:low_speed:Cn_p",
                "data:geometry:fuselage:maximum_height",
            ] = (
                -cy_beta_vt
                * (
                    lp_vt * np.cos(aoa_ref) ** 2.0
                    - lp_vt * np.cos(aoa_ref)
                    + 2.0 * z_v * np.cos(aoa_ref) * np.sin(aoa_ref)
                    - lp_vt * np.sin(aoa_ref) ** 2.0
                    - 2.0 * z_v * np.sin(aoa_ref)
                )
                / wing_span**2.0
            )
            partials[
                "data:aerodynamics:vertical_tail:low_speed:Cn_p",
                "data:geometry:vertical_tail:MAC:z",
            ] = (
                -2.0
                * cy_beta_vt
                * (
                    lp_vt * np.cos(aoa_ref) ** 2.0
                    - lp_vt * np.cos(aoa_ref)
                    + 2.0 * z_v * np.cos(aoa_ref) * np.sin(aoa_ref)
                    - lp_vt * np.sin(aoa_ref) ** 2.0
                    - 2.0 * z_v * np.sin(aoa_ref)
                )
                / wing_span**2.0
            )
            partials[
                "data:aerodynamics:vertical_tail:low_speed:Cn_p",
                "data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25",
            ] = (
                -2.0
                * cy_beta_vt
                * (
                    z_v * np.cos(aoa_ref) ** 2.0
                    - 2.0 * lp_vt * np.cos(aoa_ref) * np.sin(aoa_ref)
                    - z_v * np.cos(aoa_ref)
                    - z_v * np.sin(aoa_ref) ** 2.0
                )
                / wing_span**2.0
            )
            partials[
                "data:aerodynamics:vertical_tail:low_speed:Cn_p",
                "data:geometry:wing:span",
            ] = (
                4.0
                * cy_beta_vt
                * (lp_vt * np.cos(aoa_ref) + z_v * np.sin(aoa_ref))
                * (z_v * np.cos(aoa_ref) - lp_vt * np.sin(aoa_ref) - z_v)
                / wing_span**3.0
            )
            partials[
                "data:aerodynamics:vertical_tail:low_speed:Cn_p",
                "settings:aerodynamics:reference_flight_conditions:low_speed:AOA",
            ] = (
                -2.0
                * cy_beta_vt
                * (
                    (z_v**2.0 - lp_vt**2.0) * np.cos(2.0 * aoa_ref)
                    + z_v * lp_vt * (np.sin(aoa_ref) - 2.0 * np.sin(2.0 * aoa_ref))
                    - z_v**2.0 * np.cos(aoa_ref)
                )
                / wing_span**2.0
            )

        else:
            aoa_ref = inputs["settings:aerodynamics:reference_flight_conditions:cruise:AOA"]
            cy_beta_vt = inputs["data:aerodynamics:vertical_tail:cruise:Cy_beta"]

            partials[
                "data:aerodynamics:vertical_tail:cruise:Cn_p",
                "data:aerodynamics:vertical_tail:cruise:Cy_beta",
            ] = (
                -2.0
                * (lp_vt * np.cos(aoa_ref) + z_v * np.sin(aoa_ref))
                * (z_v * np.cos(aoa_ref) - lp_vt * np.sin(aoa_ref) - z_v)
                / wing_span**2.0
            )
            partials[
                "data:aerodynamics:vertical_tail:cruise:Cn_p",
                "data:geometry:wing:root:z",
            ] = (
                -2.0
                * cy_beta_vt
                * (
                    lp_vt * np.cos(aoa_ref) ** 2.0
                    - lp_vt * np.cos(aoa_ref)
                    + 2.0 * z_v * np.cos(aoa_ref) * np.sin(aoa_ref)
                    - lp_vt * np.sin(aoa_ref) ** 2.0
                    - 2.0 * z_v * np.sin(aoa_ref)
                )
                / wing_span**2.0
            )
            partials[
                "data:aerodynamics:vertical_tail:cruise:Cn_p",
                "data:geometry:fuselage:maximum_height",
            ] = (
                -cy_beta_vt
                * (
                    lp_vt * np.cos(aoa_ref) ** 2.0
                    - lp_vt * np.cos(aoa_ref)
                    + 2.0 * z_v * np.cos(aoa_ref) * np.sin(aoa_ref)
                    - lp_vt * np.sin(aoa_ref) ** 2.0
                    - 2.0 * z_v * np.sin(aoa_ref)
                )
                / wing_span**2.0
            )
            partials[
                "data:aerodynamics:vertical_tail:cruise:Cn_p",
                "data:geometry:vertical_tail:MAC:z",
            ] = (
                -2.0
                * cy_beta_vt
                * (
                    lp_vt * np.cos(aoa_ref) ** 2.0
                    - lp_vt * np.cos(aoa_ref)
                    + 2.0 * z_v * np.cos(aoa_ref) * np.sin(aoa_ref)
                    - lp_vt * np.sin(aoa_ref) ** 2.0
                    - 2.0 * z_v * np.sin(aoa_ref)
                )
                / wing_span**2.0
            )
            partials[
                "data:aerodynamics:vertical_tail:cruise:Cn_p",
                "data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25",
            ] = (
                -2.0
                * cy_beta_vt
                * (
                    z_v * np.cos(aoa_ref) ** 2.0
                    - 2.0 * lp_vt * np.cos(aoa_ref) * np.sin(aoa_ref)
                    - z_v * np.cos(aoa_ref)
                    - z_v * np.sin(aoa_ref) ** 2.0
                )
                / wing_span**2.0
            )
            partials["data:aerodynamics:vertical_tail:cruise:Cn_p", "data:geometry:wing:span"] = (
                4.0
                * cy_beta_vt
                * (lp_vt * np.cos(aoa_ref) + z_v * np.sin(aoa_ref))
                * (z_v * np.cos(aoa_ref) - lp_vt * np.sin(aoa_ref) - z_v)
                / wing_span**3.0
            )
            partials[
                "data:aerodynamics:vertical_tail:cruise:Cn_p",
                "settings:aerodynamics:reference_flight_conditions:cruise:AOA",
            ] = (
                -2.0
                * cy_beta_vt
                * (
                    (z_v**2.0 - lp_vt**2.0) * np.cos(2.0 * aoa_ref)
                    + z_v * lp_vt * (np.sin(aoa_ref) - 2.0 * np.sin(2.0 * aoa_ref))
                    - z_v**2.0 * np.cos(aoa_ref)
                )
                / wing_span**2.0
            )
