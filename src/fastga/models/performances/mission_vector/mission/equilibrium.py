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
from scipy.constants import g
from stdatm import Atmosphere


class Equilibrium(om.ImplicitComponent):
    """Find the conditions necessary for the aircraft equilibrium."""

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be " "treated"
        )
        self.options.declare(
            "flaps_position",
            default="cruise",
            desc="position of the flaps for the computation of the equilibrium",
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input("d_vx_dt", val=np.full(number_of_points, 0.0), units="m/s**2")
        self.add_input("mass", val=np.full(number_of_points, 1500.0), units="kg")
        self.add_input("x_cg", val=np.full(number_of_points, 5.0), units="m")
        self.add_input("gamma", val=np.full(number_of_points, 0.0), units="deg")
        self.add_input("altitude", val=np.full(number_of_points, 0.0), units="m")
        self.add_input("true_airspeed", val=np.full(number_of_points, 50.0), units="m/s")

        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
        self.add_input(
            "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="m"
        )

        self.add_input("data:aerodynamics:aircraft:cruise:CD0", np.nan)
        self.add_input("data:aerodynamics:wing:cruise:CL_alpha", val=np.nan, units="rad**-1")
        self.add_input("data:aerodynamics:wing:cruise:CL0_clean", val=np.nan)
        self.add_input("data:aerodynamics:wing:cruise:induced_drag_coefficient", np.nan)
        self.add_input("data:aerodynamics:wing:cruise:CM0_clean", val=np.nan)
        self.add_input("data:aerodynamics:fuselage:cm_alpha", val=np.nan, units="rad**-1")
        self.add_input("data:aerodynamics:horizontal_tail:cruise:CL0", val=np.nan)
        self.add_input(
            "data:aerodynamics:horizontal_tail:cruise:CL_alpha", val=np.nan, units="rad**-1"
        )
        self.add_input("data:aerodynamics:horizontal_tail:cruise:induced_drag_coefficient", np.nan)
        self.add_input("data:aerodynamics:elevator:low_speed:CL_delta", val=np.nan, units="rad**-1")
        self.add_input("data:aerodynamics:elevator:low_speed:CD_delta", val=np.nan, units="rad**-2")
        if self.options["flaps_position"] == "takeoff":
            self.add_input("data:aerodynamics:flaps:takeoff:CL", val=np.nan)
            self.add_input("data:aerodynamics:flaps:takeoff:CD", val=np.nan)
            self.add_input("data:aerodynamics:flaps:takeoff:CM", val=np.nan)
        if self.options["flaps_position"] == "landing":
            self.add_input("data:aerodynamics:flaps:landing:CL", val=np.nan)
            self.add_input("data:aerodynamics:flaps:landing:CD", val=np.nan)
            self.add_input("data:aerodynamics:flaps:landing:CM", val=np.nan)

        self.add_input("delta_Cl", val=np.full(number_of_points, 0.0))
        self.add_input("delta_Cd", val=np.full(number_of_points, 0.0))
        self.add_input("delta_Cm", val=np.full(number_of_points, 0.0))

        self.add_output("alpha", val=np.full(number_of_points, 5.0), units="deg")
        self.add_output("thrust", val=np.full(number_of_points, 1000.0), units="N")
        self.add_output("delta_m", val=np.full(number_of_points, -5.0), units="deg")

        self.declare_partials(
            of="alpha",
            wrt=[
                "altitude",
            ],
            method="fd",
            form="central",
            step=1.0e2,
        )
        self.declare_partials(
            of="alpha",
            wrt=[
                "mass",
                "gamma",
                "true_airspeed",
                "data:geometry:wing:area",
                "data:aerodynamics:wing:cruise:CL0_clean",
                "data:aerodynamics:wing:cruise:CL_alpha",
                "data:aerodynamics:horizontal_tail:cruise:CL0",
                "data:aerodynamics:horizontal_tail:cruise:CL_alpha",
                "delta_Cl",
                "data:aerodynamics:elevator:low_speed:CL_delta",
                "thrust",
                "alpha",
                "delta_m",
            ],
            method="exact",
        )
        self.declare_partials(
            of="thrust",
            wrt=[
                "altitude",
            ],
            method="fd",
            form="central",
            step=1.0e2,
        )
        self.declare_partials(
            of="thrust",
            wrt=[
                "gamma",
                "d_vx_dt",
                "mass",
                "true_airspeed",
                "data:geometry:wing:area",
                "data:aerodynamics:aircraft:cruise:CD0",
                "data:aerodynamics:wing:cruise:CL0_clean",
                "data:aerodynamics:wing:cruise:CL_alpha",
                "data:aerodynamics:wing:cruise:induced_drag_coefficient",
                "data:aerodynamics:horizontal_tail:cruise:CL0",
                "data:aerodynamics:horizontal_tail:cruise:CL_alpha",
                "data:aerodynamics:horizontal_tail:cruise:induced_drag_coefficient",
                "data:aerodynamics:elevator:low_speed:CD_delta",
                "data:aerodynamics:elevator:low_speed:CL_delta",
                "delta_Cd",
                "alpha",
                "thrust",
                "delta_m",
            ],
            method="exact",
        )
        self.declare_partials(
            of="delta_m",
            wrt=[
                "x_cg",
                "data:geometry:wing:MAC:length",
                "data:geometry:wing:MAC:at25percent:x",
                "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25",
                "data:aerodynamics:fuselage:cm_alpha",
                "data:aerodynamics:wing:cruise:CL0_clean",
                "data:aerodynamics:wing:cruise:CL_alpha",
                "data:aerodynamics:wing:cruise:CM0_clean",
                "data:aerodynamics:horizontal_tail:cruise:CL0",
                "data:aerodynamics:horizontal_tail:cruise:CL_alpha",
                "data:aerodynamics:elevator:low_speed:CL_delta",
                "delta_Cl",
                "delta_Cm",
                "alpha",
                "delta_m",
            ],
            method="exact",
        )
        if self.options["flaps_position"] == "takeoff":
            self.declare_partials(
                of="alpha", wrt="data:aerodynamics:flaps:takeoff:CL", method="exact"
            )
            self.declare_partials(
                of="thrust", wrt="data:aerodynamics:flaps:takeoff:CD", method="exact"
            )
            self.declare_partials(
                of="delta_m", wrt="data:aerodynamics:flaps:takeoff:CM", method="exact"
            )
        if self.options["flaps_position"] == "landing":
            self.declare_partials(
                of="alpha", wrt="data:aerodynamics:flaps:landing:CL", method="exact"
            )
            self.declare_partials(
                of="thrust", wrt="data:aerodynamics:flaps:landing:CD", method="exact"
            )
            self.declare_partials(
                of="delta_m", wrt="data:aerodynamics:flaps:landing:CM", method="exact"
            )

    def linearize(self, inputs, outputs, partials):

        number_of_points = self.options["number_of_points"]

        mass = inputs["mass"]
        d_vx_dt = inputs["d_vx_dt"]
        true_airspeed = inputs["true_airspeed"]
        altitude = inputs["altitude"]
        gamma = inputs["gamma"] * np.pi / 180.0
        x_cg = inputs["x_cg"]

        l0_wing = inputs["data:geometry:wing:MAC:length"]
        x_wing = inputs["data:geometry:wing:MAC:at25percent:x"]
        x_htp = x_wing + inputs["data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"]
        wing_area = inputs["data:geometry:wing:area"]

        cd0 = inputs["data:aerodynamics:aircraft:cruise:CD0"]
        cm_alpha_fus = inputs["data:aerodynamics:fuselage:cm_alpha"]
        cm0_wing = inputs["data:aerodynamics:wing:cruise:CM0_clean"]
        cl0_wing = inputs["data:aerodynamics:wing:cruise:CL0_clean"]
        cl_alpha_wing = inputs["data:aerodynamics:wing:cruise:CL_alpha"]
        coeff_k_wing = inputs["data:aerodynamics:wing:cruise:induced_drag_coefficient"]
        cl0_htp = inputs["data:aerodynamics:horizontal_tail:cruise:CL0"]
        cl_alpha_htp = inputs["data:aerodynamics:horizontal_tail:cruise:CL_alpha"]
        coeff_k_htp = inputs["data:aerodynamics:horizontal_tail:cruise:induced_drag_coefficient"]
        cl_delta_m = inputs["data:aerodynamics:elevator:low_speed:CL_delta"]
        cd_delta_m = inputs["data:aerodynamics:elevator:low_speed:CD_delta"]

        alpha = outputs["alpha"] * np.pi / 180.0
        thrust = outputs["thrust"]
        delta_m = outputs["delta_m"] * np.pi / 180.0

        if self.options["flaps_position"] == "takeoff":
            delta_cl_flaps = inputs["data:aerodynamics:flaps:takeoff:CL"]
            delta_cd_flaps = inputs["data:aerodynamics:flaps:takeoff:CD"]
            delta_cm_flaps = inputs["data:aerodynamics:flaps:takeoff:CM"]
        elif self.options["flaps_position"] == "landing":
            delta_cl_flaps = inputs["data:aerodynamics:flaps:landing:CL"]
            delta_cd_flaps = inputs["data:aerodynamics:flaps:landing:CD"]
            delta_cm_flaps = inputs["data:aerodynamics:flaps:landing:CM"]
        else:  # Cruise conditions
            delta_cl_flaps = 0.0
            delta_cd_flaps = 0.0
            delta_cm_flaps = 0.0

        delta_cl = inputs["delta_Cl"]
        delta_cd = inputs["delta_Cd"]
        delta_cm = inputs["delta_Cm"]

        rho = Atmosphere(altitude, altitude_in_feet=False).density

        dynamic_pressure = 1.0 / 2.0 * rho * np.square(true_airspeed)

        cl_wing = cl0_wing + cl_alpha_wing * alpha + delta_cl + delta_cl_flaps
        cl_htp = cl0_htp + cl_alpha_htp * alpha + cl_delta_m * delta_m

        cd_tot = (
            cd0
            + delta_cd
            + delta_cd_flaps
            + coeff_k_wing * cl_wing ** 2
            + coeff_k_htp * cl_htp ** 2
            + (cd_delta_m * delta_m ** 2.0)
        )

        d_q_d_airspeed = rho * true_airspeed

        # ------------------ Derivatives wrt alpha residuals ------------------ #

        partials["alpha", "data:aerodynamics:wing:cruise:CL0_clean"] = np.ones(number_of_points)
        partials["alpha", "data:aerodynamics:wing:cruise:CL_alpha"] = alpha
        partials["alpha", "data:aerodynamics:horizontal_tail:cruise:CL0"] = np.ones(
            number_of_points
        )
        partials["alpha", "data:aerodynamics:horizontal_tail:cruise:CL_alpha"] = alpha
        partials["alpha", "delta_Cl"] = np.identity(number_of_points)
        partials["alpha", "data:aerodynamics:elevator:low_speed:CL_delta"] = delta_m
        d_alpha_d_mass_vector = -g * np.cos(gamma) / (dynamic_pressure * wing_area)
        partials["alpha", "mass"] = np.diag(d_alpha_d_mass_vector)
        d_alpha_d_thrust_vector = np.sin(alpha) / (dynamic_pressure * wing_area)
        partials["alpha", "thrust"] = np.diag(d_alpha_d_thrust_vector)
        d_alpha_d_gamma_vector = mass * g * np.sin(gamma) / (dynamic_pressure * wing_area)
        partials["alpha", "gamma"] = np.diag(d_alpha_d_gamma_vector)
        d_alpha_d_q_vector = -(thrust * np.sin(alpha) - mass * g * np.cos(gamma)) / (
            wing_area * dynamic_pressure ** 2.0
        )
        partials["alpha", "true_airspeed"] = np.diag(d_alpha_d_q_vector * d_q_d_airspeed)
        d_alpha_d_s_vector = -(thrust * np.sin(alpha) - mass * g * np.cos(gamma)) / (
            dynamic_pressure * wing_area ** 2.0
        )
        partials["alpha", "data:geometry:wing:area"] = d_alpha_d_s_vector
        d_alpha_d_alpha_vector = (
            cl_alpha_wing + cl_alpha_htp + thrust * np.cos(alpha) / (dynamic_pressure * wing_area)
        )
        partials["alpha", "alpha"] = np.diag(d_alpha_d_alpha_vector) * np.pi / 180.0
        partials["alpha", "delta_m"] = np.identity(number_of_points) * cl_delta_m * np.pi / 180.0
        if self.options["flaps_position"] == "takeoff":
            partials["alpha", "data:aerodynamics:flaps:takeoff:CL"] = np.ones(number_of_points)
        if self.options["flaps_position"] == "landing":
            partials["alpha", "data:aerodynamics:flaps:landing:CL"] = np.ones(number_of_points)

        # ------------------ Derivatives wrt thrust residuals ------------------ #

        d_thrust_d_cl_w = -2.0 * dynamic_pressure * wing_area * coeff_k_wing * cl_wing
        d_thrust_d_cl_h = -2.0 * dynamic_pressure * wing_area * coeff_k_htp * cl_htp

        d_cl_w_d_cl_alpha_w = alpha
        d_cl_h_d_cl_alpha_h = alpha
        d_cl_h_d_cl_delta = delta_m

        partials["thrust", "d_vx_dt"] = np.diag(-mass)
        partials["thrust", "gamma"] = np.diag(-mass * g * np.cos(gamma) * np.pi / 180.0)
        partials["thrust", "mass"] = np.diag(-d_vx_dt - g * np.sin(gamma))
        partials["thrust", "true_airspeed"] = -np.diag(wing_area * cd_tot * d_q_d_airspeed)
        partials["thrust", "data:geometry:wing:area"] = -dynamic_pressure * cd_tot
        partials["thrust", "data:aerodynamics:aircraft:cruise:CD0"] = -dynamic_pressure * wing_area
        partials["thrust", "data:aerodynamics:horizontal_tail:cruise:induced_drag_coefficient"] = (
            -dynamic_pressure * wing_area * cl_htp ** 2.0
        )
        partials["thrust", "data:aerodynamics:wing:cruise:induced_drag_coefficient"] = (
            -dynamic_pressure * wing_area * cl_wing ** 2.0
        )
        partials["thrust", "delta_Cd"] = -np.diag(dynamic_pressure * wing_area)
        partials["thrust", "data:aerodynamics:elevator:low_speed:CD_delta"] = (
            -dynamic_pressure * wing_area * delta_m ** 2.0
        )
        partials["thrust", "data:aerodynamics:elevator:low_speed:CL_delta"] = (
            d_thrust_d_cl_h * d_cl_h_d_cl_delta
        )
        partials["thrust", "data:aerodynamics:wing:cruise:CL0_clean"] = d_thrust_d_cl_w
        partials["thrust", "data:aerodynamics:wing:cruise:CL_alpha"] = (
            d_thrust_d_cl_w * d_cl_w_d_cl_alpha_w
        )
        partials["thrust", "data:aerodynamics:horizontal_tail:cruise:CL0"] = d_thrust_d_cl_h
        partials["thrust", "data:aerodynamics:horizontal_tail:cruise:CL_alpha"] = (
            d_thrust_d_cl_h * d_cl_h_d_cl_alpha_h
        )
        partials["thrust", "thrust"] = np.diag(np.cos(alpha))
        d_thrust_d_alpha_vector = (
            (
                -thrust * np.sin(alpha)
                + d_thrust_d_cl_w * cl_alpha_wing
                + d_thrust_d_cl_h * cl_alpha_htp
            )
            * np.pi
            / 180.0
        )
        partials["thrust", "alpha"] = np.diag(d_thrust_d_alpha_vector)
        d_thrust_d_delta_m_vector = (
            (
                d_thrust_d_cl_h * cl_delta_m
                - 2.0 * dynamic_pressure * wing_area * cd_delta_m * delta_m
            )
            * np.pi
            / 180.0
        )
        partials["thrust", "delta_m"] = np.diag(d_thrust_d_delta_m_vector)
        if self.options["flaps_position"] == "takeoff":
            partials["thrust", "data:aerodynamics:flaps:takeoff:CD"] = -dynamic_pressure * wing_area
        if self.options["flaps_position"] == "landing":
            partials["thrust", "data:aerodynamics:flaps:landing:CD"] = -dynamic_pressure * wing_area

        # ------------------ Derivatives wrt delta_m residuals ------------------ #

        partials["delta_m", "data:aerodynamics:wing:cruise:CM0_clean"] = l0_wing * np.ones(
            number_of_points
        )
        partials["delta_m", "data:aerodynamics:fuselage:cm_alpha"] = l0_wing * alpha
        partials["delta_m", "data:aerodynamics:wing:cruise:CL0_clean"] = x_cg - x_wing
        partials["delta_m", "data:aerodynamics:wing:cruise:CM0_clean"] = l0_wing * np.ones(
            number_of_points
        )
        partials["delta_m", "data:aerodynamics:horizontal_tail:cruise:CL0"] = (
            x_cg - x_htp
        ) * np.ones(number_of_points)
        partials["delta_m", "delta_Cl"] = np.diag(x_cg - x_wing)
        partials["delta_m", "delta_Cm"] = np.diag(l0_wing)
        d_delta_m_d_alpha = (
            (x_cg - x_wing) * cl_alpha_wing + (x_cg - x_htp) * cl_alpha_htp + cm_alpha_fus
        )
        partials["delta_m", "alpha"] = np.diag(d_delta_m_d_alpha) * np.pi / 180.0
        partials["delta_m", "data:aerodynamics:horizontal_tail:cruise:CL_alpha"] = alpha * (
            x_cg - x_htp
        )
        partials["delta_m", "data:aerodynamics:elevator:low_speed:CL_delta"] = (
            x_cg - x_htp
        ) * delta_m
        partials["delta_m", "delta_m"] = (
            np.identity(number_of_points) * (x_cg - x_htp) * cl_delta_m * np.pi / 180.0
        )
        partials["delta_m", "x_cg"] = (cl_wing + cl_htp) * np.identity(number_of_points)
        partials["delta_m", "data:geometry:wing:MAC:at25percent:x"] = -(cl_wing + cl_htp) * np.ones(
            number_of_points
        )
        partials[
            "delta_m", "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"
        ] = -cl_htp
        partials["delta_m", "data:geometry:wing:MAC:length"] = (
            cm_alpha_fus * alpha + cm0_wing + delta_cm + delta_cm_flaps
        )
        partials["delta_m", "data:aerodynamics:wing:cruise:CL_alpha"] = (x_cg - x_wing) * alpha
        if self.options["flaps_position"] == "takeoff":
            partials["delta_m", "data:aerodynamics:flaps:takeoff:CM"] = l0_wing * np.ones(
                number_of_points
            )
        if self.options["flaps_position"] == "landing":
            partials["delta_m", "data:aerodynamics:flaps:landing:CM"] = l0_wing * np.ones(
                number_of_points
            )

    def apply_nonlinear(
        self, inputs, outputs, residuals, discrete_inputs=None, discrete_outputs=None
    ):

        d_vx_dt = inputs["d_vx_dt"]
        mass = inputs["mass"]
        x_cg = inputs["x_cg"]
        gamma = inputs["gamma"] * np.pi / 180.0
        true_airspeed = inputs["true_airspeed"]
        altitude = inputs["altitude"]

        wing_area = inputs["data:geometry:wing:area"]
        l0_wing = inputs["data:geometry:wing:MAC:length"]
        x_wing = inputs["data:geometry:wing:MAC:at25percent:x"]
        x_htp = x_wing + inputs["data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"]

        cd0 = inputs["data:aerodynamics:aircraft:cruise:CD0"]
        cl0_wing = inputs["data:aerodynamics:wing:cruise:CL0_clean"]
        cl_alpha_wing = inputs["data:aerodynamics:wing:cruise:CL_alpha"]
        coeff_k_wing = inputs["data:aerodynamics:wing:cruise:induced_drag_coefficient"]
        cm0_wing = inputs["data:aerodynamics:wing:cruise:CM0_clean"]
        cl0_htp = inputs["data:aerodynamics:horizontal_tail:cruise:CL0"]
        cl_alpha_htp = inputs["data:aerodynamics:horizontal_tail:cruise:CL_alpha"]
        coeff_k_htp = inputs["data:aerodynamics:horizontal_tail:cruise:induced_drag_coefficient"]
        cm_alpha_fus = inputs["data:aerodynamics:fuselage:cm_alpha"]
        cl_delta_m = inputs["data:aerodynamics:elevator:low_speed:CL_delta"]
        cd_delta_m = inputs["data:aerodynamics:elevator:low_speed:CD_delta"]

        delta_cl = inputs["delta_Cl"]
        delta_cd = inputs["delta_Cd"]
        delta_cm = inputs["delta_Cm"]

        if self.options["flaps_position"] == "takeoff":
            delta_cl_flaps = inputs["data:aerodynamics:flaps:takeoff:CL"]
            delta_cd_flaps = inputs["data:aerodynamics:flaps:takeoff:CD"]
            delta_cm_flaps = inputs["data:aerodynamics:flaps:takeoff:CM"]
        elif self.options["flaps_position"] == "landing":
            delta_cl_flaps = inputs["data:aerodynamics:flaps:landing:CL"]
            delta_cd_flaps = inputs["data:aerodynamics:flaps:landing:CD"]
            delta_cm_flaps = inputs["data:aerodynamics:flaps:landing:CM"]
        else:  # Cruise conditions
            delta_cl_flaps = 0.0
            delta_cd_flaps = 0.0
            delta_cm_flaps = 0.0

        alpha = outputs["alpha"] * np.pi / 180.0
        thrust = outputs["thrust"]
        delta_m = outputs["delta_m"] * np.pi / 180.0

        rho = Atmosphere(altitude, altitude_in_feet=False).density

        dynamic_pressure = 1.0 / 2.0 * rho * np.square(true_airspeed)

        cl_wing_clean = cl0_wing + cl_alpha_wing * alpha
        cl_wing_flaps = cl_wing_clean + delta_cl_flaps
        cl_wing_slip = cl_wing_clean + delta_cl
        cl_wing = cl_wing_clean + delta_cl_flaps + delta_cl
        cl_htp = cl0_htp + cl_alpha_htp * alpha + cl_delta_m * delta_m

        cd_tot = (
            cd0
            + delta_cd
            + delta_cd_flaps
            + coeff_k_wing * cl_wing_flaps ** 2.0
            + coeff_k_htp * cl_htp ** 2.0
            + (cd_delta_m * delta_m ** 2.0)
        )

        residuals["alpha"] = (
            cl_wing
            + cl_htp
            + (thrust * np.sin(alpha) - mass * g * np.cos(gamma)) / (dynamic_pressure * wing_area)
        )
        residuals["thrust"] = (
            thrust * np.cos(alpha)
            - dynamic_pressure * wing_area * cd_tot
            - mass * g * np.sin(gamma)
            - mass * d_vx_dt
        )
        residuals["delta_m"] = (
            (x_cg - x_wing) * cl_wing_slip
            + (x_cg - x_htp) * cl_htp
            + (cm0_wing + delta_cm + delta_cm_flaps + cm_alpha_fus * alpha) * l0_wing
        )
