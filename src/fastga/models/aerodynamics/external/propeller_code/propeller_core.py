"""Computation of propeller aero properties."""
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

import logging

import numpy as np
from scipy.optimize import fsolve

import openmdao.api as om

from stdatm import Atmosphere

from fastga.models.aerodynamics.external.xfoil.xfoil_polar import POLAR_POINT_COUNT

_LOGGER = logging.getLogger(__name__)

THRUST_PTS_NB = 30
SPEED_PTS_NB = 10


class PropellerCoreModule(om.ExplicitComponent):
    """
    Core component for the computation of the propeller performance.

    Compressibility correction are taken from :cite:`hoyos:2022` for subsonic corrections.
    Reynolds effects correction are taken from yamauchi:1983.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.theta_min = 0.0
        self.theta_max = 0.0

    def initialize(self):
        self.options.declare("sections_profile_position_list", types=list)
        self.options.declare("sections_profile_name_list", types=list)
        self.options.declare("elements_number", default=20, types=int)

    def setup(self):
        self.add_input("reference_reynolds", val=1e6)
        self.add_input("data:geometry:propeller:diameter", val=np.nan, units="m")
        self.add_input("data:geometry:propeller:hub_diameter", val=np.nan, units="m")
        self.add_input("data:geometry:propeller:blades_number", val=np.nan)
        self.add_input(
            "data:geometry:propeller:average_rpm",
            val=2500,
            units="rpm",
        )
        self.add_input(
            "data:geometry:propeller:sweep_vect",
            val=np.nan,
            units="deg",
            shape_by_conn=True,
            copy_shape="data:geometry:propeller:radius_ratio_vect",
        )
        self.add_input(
            "data:geometry:propeller:chord_vect",
            val=np.nan,
            units="m",
            shape_by_conn=True,
            copy_shape="data:geometry:propeller:radius_ratio_vect",
        )
        self.add_input(
            "data:geometry:propeller:twist_vect",
            val=np.nan,
            units="deg",
            shape_by_conn=True,
            copy_shape="data:geometry:propeller:radius_ratio_vect",
        )
        self.add_input("data:geometry:propeller:radius_ratio_vect", val=np.nan, shape_by_conn=True)

        for profile in self.options["sections_profile_name_list"]:
            self.add_input(
                profile + "_polar:alpha", val=np.nan, units="deg", shape=POLAR_POINT_COUNT
            )
            self.add_input(profile + "_polar:CL", val=np.nan, shape=POLAR_POINT_COUNT)
            self.add_input(profile + "_polar:CD", val=np.nan, shape=POLAR_POINT_COUNT)

        self.declare_partials(of="*", wrt="*", method="fd")

    def compute_extreme_pitch(self, inputs, v_inf):
        """For a given flight speed computes the min and max possible value of theta at .75 r/R."""
        radius_ratio_vect = inputs["data:geometry:propeller:radius_ratio_vect"]
        omega = inputs["data:geometry:propeller:average_rpm"]
        phi_vect = (
            180.0
            / np.pi
            * np.arctan((v_inf + 60.0 / v_inf) / (omega * 2.0 * np.pi / 60.0 * radius_ratio_vect))
        )
        phi_75 = np.interp(0.75, radius_ratio_vect, phi_vect)
        self.theta_min = phi_75 - 10.0
        self.theta_max = phi_75 + 25.0

    def compute_pitch_performance(
        self, inputs, theta_75, v_inf, altitude, omega, radius, alpha_list, cl_list, cd_list
    ):

        """
        This function calculates the thrust, efficiency and power at a given flight speed,
        altitude h and propeller angular speed.

        :param inputs: structure of data relative to the blade geometry available from setup
        :param theta_75: pitch defined at r = 0.75*R radial position [deg].
        :param v_inf: flight speeds [m/s].
        :param altitude: flight altitude [m].
        :param omega: angular velocity of the propeller [RPM].
        :param radius: array of radius of discretized blade elements [m].
        :param alpha_list: angle of attack list for aerodynamic coefficient of profile at
        discretized blade element [deg].
        :param cl_list: cl list for aerodynamic coefficient of profile at discretized blade
        element [-].
        :param cd_list: cd list for aerodynamic coefficient of profile at discretized blade
        element [-].

        :return: thrust [N], eta (efficiency) [-] and power [W].
        """

        blades_number = inputs["data:geometry:propeller:blades_number"]
        radius_min = inputs["data:geometry:propeller:hub_diameter"] / 2.0
        radius_max = inputs["data:geometry:propeller:diameter"] / 2.0
        sweep_vect = inputs["data:geometry:propeller:sweep_vect"]
        chord_vect = inputs["data:geometry:propeller:chord_vect"]
        twist_vect = inputs["data:geometry:propeller:twist_vect"]
        radius_ratio_vect = inputs["data:geometry:propeller:radius_ratio_vect"]
        reference_reynolds = inputs["reference_reynolds"]
        length = radius_max - radius_min
        element_length = length / self.options["elements_number"]
        omega = omega * np.pi / 30.0
        atm = Atmosphere(altitude, altitude_in_feet=False)

        theta_75_ref = np.interp(0.75, radius_ratio_vect, twist_vect)

        # Initialise vectors
        vi_vect = np.zeros_like(radius)
        vt_vect = np.zeros_like(radius)
        thrust_element_vector = np.zeros_like(radius)
        torque_element_vector = np.zeros_like(radius)
        alpha_vect = np.zeros_like(radius)
        speed_vect = np.array([0.1 * float(v_inf), 1.0])

        chord = np.interp(radius / radius_max, radius_ratio_vect, chord_vect)

        theta = np.interp(radius / radius_max, radius_ratio_vect, twist_vect) + (
            theta_75 - theta_75_ref
        )
        sweep = np.interp(radius / radius_max, radius_ratio_vect, sweep_vect)

        # Loop on element number to compute equations
        for idx, _ in enumerate(radius):

            # Solve BEM vs. disk theory system of equations
            speed_vect = fsolve(
                self.delta,
                speed_vect,
                (
                    radius[idx],
                    radius_min,
                    radius_max,
                    chord[idx],
                    blades_number,
                    sweep[idx],
                    omega,
                    v_inf,
                    theta[idx],
                    alpha_list[idx, :],
                    cl_list[idx, :],
                    cd_list[idx, :],
                    atm,
                    reference_reynolds,
                ),
                xtol=1e-3,
            )
            vi_vect[idx] = speed_vect[0]
            vt_vect[idx] = speed_vect[1]
            results = self.bem_theory(
                speed_vect,
                radius[idx],
                chord[idx],
                blades_number,
                sweep[idx],
                omega,
                v_inf,
                theta[idx],
                alpha_list[idx, :],
                cl_list[idx, :],
                cd_list[idx, :],
                atm,
                reference_reynolds,
            )
            out_of_polars = results[3]
            if out_of_polars:
                thrust_element_vector[idx] = 0.0
                torque_element_vector[idx] = 0.0
            else:
                thrust_element_vector[idx] = results[0] * element_length * atm.density
                torque_element_vector[idx] = results[1] * element_length * atm.density
            alpha_vect[idx] = results[2]

        torque = np.sum(torque_element_vector)
        thrust = float(np.sum(thrust_element_vector))
        power = float(torque * omega)
        eta = float(v_inf * thrust / power)

        return thrust, eta, torque

    @staticmethod
    def bem_theory(
        speed_vect: np.array,
        radius: float,
        chord: float,
        blades_number: float,
        sweep: float,
        omega: float,
        v_inf: float,
        theta: float,
        alpha_element: np.array,
        cl_element: np.array,
        cd_element: np.array,
        atm: Atmosphere,
        reference_reynolds: float,
    ):
        """
        The core of the Propeller code. Given the geometry of a propeller element,
        its aerodynamic polars, flight conditions and axial/tangential velocities it computes the
        thrust and the torque produced using force and momentum with BEM theory.

        :param speed_vect: the vector of axial and tangential induced speed in m/s
        :param radius: radius position of the element center  [m]
        :param chord: chord at the center of element [m]
        :param blades_number: number of blades [-]
        :param sweep: sweep angle [deg.]
        :param omega: angular speed of propeller [rad/sec]
        :param v_inf: flight speed [m/s]
        :param theta: profile angle relative to aircraft airflow v_inf [deg.]
        :param alpha_element: reference angle vector for element polars [deg.]
        :param cl_element: cl vector for element [-]
        :param cd_element: cd vector for element [-]
        :param atm: atmosphere properties
        :param reference_reynolds: Reynolds number at which the aerodynamic properties were computed

        :return: The calculated dT/(rho*dr) and dQ/(rho*dr) increments with BEM method.
        """

        # Extract axial/tangential speeds
        v_i = speed_vect[0]
        v_t = speed_vect[1]

        # Calculate speed composition and relative air angle (in deg.)
        v_ax = v_inf + v_i
        v_t = (omega * radius - v_t) * np.cos(sweep * np.pi / 180.0)
        rel_fluid_speed = np.sqrt(v_ax ** 2.0 + v_t ** 2.0)
        phi = np.arctan(v_ax / v_t)
        alpha = theta - phi * 180.0 / np.pi

        # Compute local mach
        atm.true_airspeed = rel_fluid_speed
        mach_local = atm.mach

        # Apply the compressibility corrections for cl and cd
        out_of_polars = bool((alpha > max(alpha_element)) or (alpha < min(alpha_element)))

        c_l = np.interp(alpha, alpha_element, cl_element)
        c_d = np.interp(alpha, alpha_element, cd_element)

        if mach_local < 1:
            beta = np.sqrt(1 - mach_local ** 2.0)
            c_l = c_l / (beta + c_l * mach_local ** 2.0 / (2.0 + 2.0 * beta))
            c_d = c_d / (beta + c_d * mach_local ** 2.0 / (2.0 + 2.0 * beta))
        else:
            beta = np.sqrt(mach_local ** 2.0 - 1)
            c_l = c_l / beta
            c_d = c_d / beta

        reynolds = chord * atm.unitary_reynolds
        f_re = (3.46 * np.log(reynolds) - 5.6) ** -2
        f_re_t = (3.46 * np.log(reference_reynolds) - 5.6) ** -2
        c_d = c_d * f_re / f_re_t

        # Calculate force and momentum
        thrust_element = (
            0.5
            * blades_number
            * chord
            * rel_fluid_speed ** 2.0
            * (c_l * np.cos(phi) - c_d * np.sin(phi))
        )
        torque_element = (
            0.5
            * blades_number
            * chord
            * rel_fluid_speed ** 2.0
            * (c_l * np.sin(phi) + c_d * np.cos(phi))
            * radius
        )

        # Store results
        output = np.empty(4)
        output[0] = thrust_element
        output[1] = torque_element
        output[2] = alpha
        output[3] = out_of_polars

        return output

    # noinspection PyUnusedLocal
    @staticmethod
    def disk_theory(
        speed_vect: np.array,
        radius: float,
        radius_min: float,
        radius_max: float,
        blades_number: float,
        sweep: float,
        omega: float,
        v_inf: float,
    ):
        """
        The core of the Propeller code. Given the geometry of a propeller element,
        its aerodynamic polars, flight conditions and axial/tangential velocities it computes the
        thrust and the torque produced using force and momentum with disk theory.

        :param speed_vect: the vector of axial and tangential induced speed in m/s
        :param radius: radius position of the element center  [m]
        :param radius_min: Hub radius [m]
        :param radius_max: Max radius [m]
        :param blades_number: number of blades [-]
        :param sweep: sweep angle [deg.]
        :param omega: angular speed of propeller [rad/sec]
        :param v_inf: flight speed [m/s]

        :return: The calculated dT/(rho*dr) and dQ/(rho*dr) increments with disk theory method.
        """

        # Extract axial/tangential speeds
        v_i = speed_vect[0]
        v_t = speed_vect[1]

        # Calculate speed composition and relative air angle (in deg.)
        v_ax = v_inf + v_i
        # Needed for the computation of the hub lost factor
        # phi = math.atan(v_ax / (omega * radius - v_t) * math.cos(sweep * math.pi / 180.0))

        # f_tip is the tip loose factor
        f_tip = (
            2
            / np.pi
            * np.arccos(
                np.exp(
                    -blades_number
                    / 2
                    * (
                        (radius_max - radius)
                        / radius
                        * np.sqrt(1 + (omega * radius / (v_ax + 1e-12 * (v_ax == 0.0))) ** 2.0)
                    )
                )
            )
        )

        # f_hub is the hub loose factor FIXME: to be activated in future versions
        # if phi > 0.0:
        #     f_hub = min(
        #         1.0,
        #         2
        #         / math.pi
        #         * math.acos(
        #             math.exp(
        #                  -blades_number / 2 * (radius - radius_min) / (radius * math.sin(phi))
        #             )
        #         ),
        #     )
        # else:
        #     f_hub = 1.0
        f_hub = 1.0

        # Calculate force and momentum
        thrust_element = 4.0 * np.pi * radius * (v_inf + v_i) * v_i * f_tip * f_hub
        torque_element = 4.0 * np.pi * radius ** 2.0 * (v_inf + v_i) * v_t * f_tip * f_hub

        # Store results
        output = np.empty(2)
        output[0] = thrust_element
        output[1] = torque_element

        return output

    def delta(
        self,
        speed_vect: np.array,
        radius: float,
        radius_min: float,
        radius_max: float,
        chord: float,
        blades_number: float,
        sweep: float,
        omega: float,
        v_inf: float,
        theta: float,
        alpha_element: np.array,
        cl_element: np.array,
        cd_element: np.array,
        atm: Atmosphere,
        reference_reynolds: float,
    ):
        """
        The core of the Propeller code. Given the geometry of a propeller element,
        its aerodynamic polars, flight conditions and axial/tangential velocities it computes the
        thrust and the torque produced using force and momentum with disk theory.

        :param speed_vect: the vector of axial and tangential induced speed in m/s
        :param radius: radius position of the element center  [m]
        :param radius_min: Hub radius [m]
        :param radius_max: Max radius [m]
        :param chord: chord at the center of element [m]
        :param blades_number: number of blades [-]
        :param sweep: sweep angle [deg.]
        :param omega: angular speed of propeller [rad/sec]
        :param v_inf: flight speed [m/s]
        :param theta: profile angle relative to aircraft airflow v_inf [DEG]
        :param alpha_element: reference angle vector for element polars [DEG]
        :param cl_element: cl vector for element [-]
        :param cd_element: cd vector for element [-]
        :param atm: atmosphere properties
        :param reference_reynolds: Reynolds number at which the aerodynamic properties were computed

        :return: The difference between BEM dual methods for dT/(rho*dr) and dQ/ increments.
        """

        bem_result = self.bem_theory(
            speed_vect,
            radius,
            chord,
            blades_number,
            sweep,
            omega,
            v_inf,
            theta,
            alpha_element,
            cl_element,
            cd_element,
            atm,
            reference_reynolds,
        )

        adt_result = self.disk_theory(
            speed_vect, radius, radius_min, radius_max, blades_number, sweep, omega, v_inf
        )

        return bem_result[0:1] - adt_result

    @staticmethod
    def reshape_polar(alpha, c_l, c_d):
        """
        Reads the polar under the openmdao format (meaning with additional zeros and reshape
        so that only relevant angle are considered.

        Assumes that the AOA list is ordered.
        """
        idx_start = np.argmin(alpha)
        idx_end = np.argmax(alpha)

        return (
            alpha[idx_start : idx_end + 1],
            c_l[idx_start : idx_end + 1],
            c_d[idx_start : idx_end + 1],
        )
