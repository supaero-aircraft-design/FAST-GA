"""
Computation of propeller aero properties
"""
#  This file is part of FAST : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2020  ONERA & ISAE-SUPAERO
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

import os.path as pth
import numpy as np
import openmdao.api as om
import math
import logging
import pandas as pd
from scipy.optimize import fsolve

from fastoad.model_base import Atmosphere
import fastga.models.aerodynamics.external.xfoil as xfoil
from fastga.models.aerodynamics.external.xfoil.xfoil_polar import XfoilPolar

from fastoad.module_management.service_registry import RegisterOpenMDAOSystem
from fastoad.module_management.constants import ModelDomain

_LOGGER = logging.getLogger(__name__)

THRUST_PTS_NB = 30
SPEED_PTS_NB = 10


@RegisterOpenMDAOSystem("fastga.aerodynamics.propeller", domain=ModelDomain.AERODYNAMICS)
class ComputePropellerPerformance(om.Group):
    def initialize(self):
        self.options.declare(
            "sections_profile_position_list",
            default=[0.0, 0.25, 0.28, 0.35, 0.40, 0.45],
            types=list,
        )
        self.options.declare(
            "sections_profile_name_list",
            default=["naca4430", "naca4424", "naca4420", "naca4414", "naca4412", "naca4409"],
            types=list,
        )
        self.options.declare("elements_number", default=20, types=int)
        self.options.declare("vectors_length", default=10, types=int)

    def setup(self):
        ivc = om.IndepVarComp()
        ivc.add_output("data:aerodynamics:propeller:mach", val=0.0)
        ivc.add_output("data:aerodynamics:propeller:reynolds", val=1e6)
        self.add_subsystem("propeller_aero_conditions", ivc, promotes=["*"])
        for profile in self.options["sections_profile_name_list"]:
            self.add_subsystem(
                profile + "_polar",
                XfoilPolar(
                    airfoil_file=profile + ".af", alpha_end=30.0, activate_negative_angle=True,
                ),
                promotes=[],
            )
            self.connect("data:aerodynamics:propeller:mach", profile + "_polar.xfoil:mach")
            self.connect("data:aerodynamics:propeller:reynolds", profile + "_polar.xfoil:reynolds")
        self.add_subsystem(
            "propeller_aero",
            _ComputePropellerPerformance(
                sections_profile_position_list=self.options["sections_profile_position_list"],
                sections_profile_name_list=self.options["sections_profile_name_list"],
                elements_number=self.options["elements_number"],
                vectors_length=self.options["vectors_length"],
            ),
            promotes=["*"],
        )


class _ComputePropellerPerformance(om.ExplicitComponent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.theta_min = 0.0
        self.theta_max = 0.0

    def initialize(self):
        self.options.declare("sections_profile_position_list", types=list)
        self.options.declare("sections_profile_name_list", types=list)
        self.options.declare("average_rpm", default=2500.0, types=float)
        self.options.declare("elements_number", default=20, types=int)
        self.options.declare("vectors_length", default=10, types=int)

    def setup(self):
        self.add_input("data:geometry:propeller:diameter", val=np.nan, units="m")
        self.add_input("data:geometry:propeller:hub_diameter", val=np.nan, units="m")
        self.add_input("data:geometry:propeller:blades_number", val=np.nan)
        nans_array = np.full(self.options["vectors_length"], np.nan)
        self.add_input("data:geometry:propeller:sweep_vect", val=nans_array, units="deg")
        self.add_input("data:geometry:propeller:chord_vect", val=nans_array, units="m")
        self.add_input("data:geometry:propeller:twist_vect", val=nans_array, units="deg")
        self.add_input("data:geometry:propeller:radius_ratio_vect", val=nans_array)
        self.add_input("data:mission:sizing:main_route:cruise:altitude", val=np.nan, units="m")
        self.add_input("data:TLAR:v_cruise", val=np.nan, units="m/s")

        self.add_output(
            "data:aerodynamics:propeller:sea_level:efficiency", shape=(SPEED_PTS_NB, THRUST_PTS_NB)
        )
        self.add_output(
            "data:aerodynamics:propeller:sea_level:thrust", shape=THRUST_PTS_NB, units="N"
        )
        self.add_output(
            "data:aerodynamics:propeller:sea_level:thrust_limit", shape=SPEED_PTS_NB, units="N"
        )
        self.add_output(
            "data:aerodynamics:propeller:sea_level:speed", shape=SPEED_PTS_NB, units="m/s"
        )
        self.add_output(
            "data:aerodynamics:propeller:cruise_level:efficiency",
            shape=(SPEED_PTS_NB, THRUST_PTS_NB),
        )
        self.add_output(
            "data:aerodynamics:propeller:cruise_level:thrust", shape=THRUST_PTS_NB, units="N"
        )
        self.add_output(
            "data:aerodynamics:propeller:cruise_level:thrust_limit", shape=SPEED_PTS_NB, units="N"
        )
        self.add_output(
            "data:aerodynamics:propeller:cruise_level:speed", shape=SPEED_PTS_NB, units="m/s"
        )

        self.declare_partials(of="*", wrt="*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        _LOGGER.debug("Entering propeller computation")

        # Define init values
        omega = self.options["average_rpm"]
        v_min = 5.0
        v_max = inputs["data:TLAR:v_cruise"] * 1.2
        speed_interp = np.linspace(v_min, v_max, SPEED_PTS_NB)

        # Construct table for init of climb
        altitude = 0.0
        thrust_vect, theta_vect, eta_vect = self.construct_table(
            inputs, speed_interp, altitude, omega
        )
        # plt.show()
        # Reformat table
        thrust_limit, thrust_interp, efficiency_interp = self.reformat_table(thrust_vect, eta_vect)
        # # Plot graphs
        # X, Y = np.meshgrid(speed_interp, thrust_interp)
        # fig, ax = plt.subplots(1, 1)
        # cp = ax.contourf(X, Y, np.transpose(efficiency_interp))
        # fig.colorbar(cp)  # Add a colorbar to a plot
        # ax.set_title('Efficiency map @ 0m')
        # ax.set_xlabel('Air speed [m/s]')
        # ax.set_ylabel('Thrust [N]')
        # plt.plot(speed_interp, thrust_limit)
        # plt.show()
        # Save results
        outputs["data:aerodynamics:propeller:sea_level:efficiency"] = efficiency_interp
        outputs["data:aerodynamics:propeller:sea_level:thrust"] = thrust_interp
        outputs["data:aerodynamics:propeller:sea_level:thrust_limit"] = thrust_limit
        outputs["data:aerodynamics:propeller:sea_level:speed"] = speed_interp

        # construct table for cruise
        altitude = inputs["data:mission:sizing:main_route:cruise:altitude"]
        thrust_vect, theta_vect, eta_vect = self.construct_table(
            inputs, speed_interp, altitude, omega
        )
        # Reformat table
        thrust_limit, thrust_interp, efficiency_interp = self.reformat_table(thrust_vect, eta_vect)

        _LOGGER.debug("Finishing propeller computation")

        # Save results
        outputs["data:aerodynamics:propeller:cruise_level:efficiency"] = efficiency_interp
        outputs["data:aerodynamics:propeller:cruise_level:thrust"] = thrust_interp
        outputs["data:aerodynamics:propeller:cruise_level:thrust_limit"] = thrust_limit
        outputs["data:aerodynamics:propeller:cruise_level:speed"] = speed_interp

    def compute_extreme_pitch(self, inputs, v_inf):
        radius_ratio_vect = inputs["data:geometry:propeller:radius_ratio_vect"]
        omega = self.options["average_rpm"]
        phi_vect = (
            180.0
            / math.pi
            * np.arctan((v_inf + 60.0 / v_inf) / (omega * 2.0 * math.pi / 60.0 * radius_ratio_vect))
        )
        phi_75 = np.interp(0.75, radius_ratio_vect, phi_vect)
        self.theta_min = phi_75 - 10.0
        self.theta_max = phi_75 + 25.0

    def construct_table(self, inputs, speed_interp, altitude, omega):
        thrust_vect = []
        theta_vect = []
        eta_vect = []
        for v_inf in speed_interp:
            self.compute_extreme_pitch(inputs, v_inf)
            theta_interp = np.linspace(self.theta_min, self.theta_max, 100)
            local_thrust_vect = []
            local_theta_vect = []
            local_eta_vect = []
            for theta_75 in theta_interp:
                thrust, eta, _ = self.compute_pitch_performance(
                    inputs, theta_75, v_inf, altitude, omega, self.options["elements_number"]
                )
                if len(local_thrust_vect) != 0:
                    local_thrust_vect.append(thrust)
                    local_theta_vect.append(theta_75)
                    local_eta_vect.append(eta)
                else:
                    local_thrust_vect.append(thrust)
                    local_theta_vect.append(theta_75)
                    local_eta_vect.append(eta)

            # Find first the "monotone" zone (10 points of increase)
            idx_in_zone = 0
            thrust_difference = np.array(local_thrust_vect[1:]) - np.array(local_thrust_vect[0:-1])
            for idx in range(5, len(thrust_difference)):
                if np.sum(np.array(thrust_difference[idx - 5 : idx + 5]) > 0.0) == len(
                    thrust_difference[idx - 5 : idx + 5]
                ):
                    idx_in_zone = idx + 1
                    break
            # Erase end of the curve if thrust decreases
            if len(np.where(thrust_difference[idx_in_zone:] < 0.0)[0]) != 0:
                idx_end = np.min(np.where(thrust_difference[idx_in_zone:] < 0.0)) + idx_in_zone
                local_thrust_vect = local_thrust_vect[0 : idx_end + 1]
                local_theta_vect = local_theta_vect[0 : idx_end + 1]
                local_eta_vect = local_eta_vect[0 : idx_end + 1]
            # Erase start of the curve if thrust is negative or decreases
            idx_start = 0
            thrust_difference = np.array(local_thrust_vect[1:]) - np.array(local_thrust_vect[0:-1])
            if len(np.where(np.array(local_thrust_vect) < 0.0)[0]) != 0.0:
                idx_start = int(np.max(np.where(np.array(local_thrust_vect) < 0.0)))
            if len(np.where(thrust_difference < 0.0)[0]) != 0.0:
                idx_start = max(idx_start, int(np.max(np.where(thrust_difference < 0.0)) + 1))
            local_thrust_vect = local_thrust_vect[idx_start:]
            local_theta_vect = local_theta_vect[idx_start:]
            local_eta_vect = local_eta_vect[idx_start:]
            # Erase remaining points with negative or >1.0 efficiency
            idx_drop = np.where(
                (np.array(local_eta_vect) <= 0.0) + (np.array(local_eta_vect) > 1.0)
            )[0].tolist()
            for idx in sorted(idx_drop, reverse=True):
                del local_thrust_vect[idx]
                del local_theta_vect[idx]
                del local_eta_vect[idx]

            # Save vectors
            thrust_vect.append(local_thrust_vect)
            theta_vect.append(local_theta_vect)
            eta_vect.append(local_eta_vect)

            # # Plot graphs
            # thrust_vect.append(local_thrust_vect)
            # theta_vect.append(local_theta_vect)
            # eta_vect.append(local_eta_vect)
            # plt.figure(1)
            # plt.subplot(311)
            # plt.xlabel("0.75R pitch angle [°]")
            # plt.ylabel("Thrust [N]")
            # plt.plot(local_theta_vect, local_thrust_vect)
            # plt.subplot(312)
            # plt.xlabel("0.75R pitch angle [°]")
            # plt.ylabel("Efficiency [-]")
            # plt.plot(local_theta_vect, local_eta_vect)
            # plt.subplot(313)
            # plt.xlabel("0.75R pitch angle [°]")
            # plt.ylabel("Torque [-]")
            # plt.plot(local_theta_vect,
            #          v_inf * np.array(local_thrust_vect) / (np.array(local_eta_vect) * omega * math.pi / 30.0))

        return thrust_vect, theta_vect, eta_vect

    @staticmethod
    def reformat_table(thrust_vect, eta_vect):
        min_thrust = 0.0
        max_thrust = 0.0
        thrust_limit = []
        for idx in range(len(thrust_vect)):
            min_thrust = max(min_thrust, thrust_vect[idx][0])
            max_thrust = max(max_thrust, thrust_vect[idx][-1])
            thrust_limit.append(thrust_vect[idx][-1])
        thrust_interp = np.linspace(min_thrust, max_thrust, THRUST_PTS_NB)
        thrust_limit = np.array(thrust_limit)
        efficiency_interp = np.zeros((SPEED_PTS_NB, len(thrust_interp)))
        for idx_speed in range(SPEED_PTS_NB):
            local_thrust_vect = thrust_vect[idx_speed]
            local_eta_vect = eta_vect[idx_speed]
            efficiency_interp[idx_speed, :] = np.interp(
                thrust_interp, local_thrust_vect, local_eta_vect
            )

        return thrust_limit, thrust_interp, efficiency_interp

    def compute_pitch_performance(self, inputs, theta_75, v_inf, h, omega, elements_number):

        """

        This function calculates the thrust, efficiency and power at a given flight speed, altitude h and propeller
        angular speed.

        :param inputs: structure of data relative to the blade geometry available from setup
        :param theta_75: pitch defined at r = 0.75*R radial position [deg.]
        :param v_inf: flight speeds [m/s]
        :param h: flight altitude [m]
        :param omega: angular velocity of the propeller [RPM]
        :param elements_number: number of elements for discretization [-]

        :return: thrust [N], eta (efficiency) [-] and power [W]
        """

        blades_number = inputs["data:geometry:propeller:blades_number"]
        sections_profile_position_list = self.options["sections_profile_position_list"]
        sections_profile_name_list = self.options["sections_profile_name_list"]
        radius_min = inputs["data:geometry:propeller:hub_diameter"] / 2.0
        radius_max = inputs["data:geometry:propeller:diameter"] / 2.0
        sweep_vect = inputs["data:geometry:propeller:sweep_vect"]
        chord_vect = inputs["data:geometry:propeller:chord_vect"]
        twist_vect = inputs["data:geometry:propeller:twist_vect"]
        radius_ratio_vect = inputs["data:geometry:propeller:radius_ratio_vect"]
        length = radius_max - radius_min
        dr = length / elements_number
        omega = omega * math.pi / 30.0
        atm = Atmosphere(h, altitude_in_feet=False)

        # Initialise vectors
        vi_vect = np.zeros(elements_number)
        vt_vect = np.zeros(elements_number)
        radius_vect = np.zeros(elements_number)
        theta_vect = np.zeros(elements_number)
        dT_vect = np.zeros(elements_number)
        dQ_vect = np.zeros(elements_number)
        alpha_vect = np.zeros(elements_number)
        speed_vect = np.array([0.1 * float(v_inf), 1.0])

        # Loop on element number to compute equations
        for i in range(elements_number):

            # Calculate element center radius and chord
            radius = radius_min + (i + 0.5) * dr
            chord = np.interp(radius / radius_max, radius_ratio_vect, chord_vect)

            # Find related profile name
            index = np.where(sections_profile_position_list < (radius / radius_max))[0]
            if index is None:
                profile_name = sections_profile_name_list[0]
            else:
                profile_name = sections_profile_name_list[int(index[-1])]

            # Load profile polars
            alpha_element, cl_element, cd_element = self.read_polar_result(profile_name)

            # Search element angle to aircraft axial air (~v_inf) and sweep angle
            theta_75_ref = np.interp(0.75, radius_ratio_vect, twist_vect)
            theta = np.interp(radius / radius_max, radius_ratio_vect, twist_vect) + (
                theta_75 - theta_75_ref
            )
            theta_vect[i] = theta
            sweep = np.interp(radius / radius_max, radius_ratio_vect, sweep_vect)

            # Solve BEM vs. disk theory system of equations
            speed_vect = fsolve(
                self.delta,
                speed_vect,
                (
                    radius,
                    radius_min,
                    radius_max,
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
                ),
                xtol=1e-3,
            )
            vi_vect[i] = speed_vect[0]
            vt_vect[i] = speed_vect[1]
            radius_vect[i] = radius
            results = self.bem_theory(
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
            )
            # results = self.disk_theory(speed_vect, radius, radius_min, radius_max, blades_number, sweep, omega, v_inf)
            out_of_polars = results[3]
            if out_of_polars:
                dT_vect[i] = 0.0
                dQ_vect[i] = 0.0
                # warnings.warn("Propeller element out of calculated polars: contribution canceled!")
            else:
                dT_vect[i] = results[0] * dr * atm.density
                dQ_vect[i] = results[1] * dr * atm.density
            alpha_vect[i] = results[2]

        torque = np.sum(dQ_vect)
        thrust = float(np.sum(dT_vect))
        power = float(torque * omega)
        eta = float(v_inf * thrust / power)

        # if eta < 0.0 or eta > 1.0:
        #     warnings.warn("Propeller not working in propulsive mode!")

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
    ):
        """
        The core of the Propeller code. Given the geometry of a propeller element, its aerodynamic polars, flight
        conditions and axial/tangential velocities it computes the thrust and the torque produced using force and
        momentum with BEM theory.

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

        :return: The calculated dT/(rho*dr) and dQ/(rho*dr) increments with BEM method.
        """

        # Extract axial/tangential speeds
        v_i = speed_vect[0]
        v_t = speed_vect[1]

        # Calculate speed composition and relative air angle (in deg.)
        v_ax = v_inf + v_i
        v_t = (omega * radius - v_t) * math.cos(sweep * math.pi / 180.0)
        w = math.sqrt(v_ax ** 2.0 + v_t ** 2.0)
        phi = math.atan(v_ax / v_t)
        alpha = theta - phi * 180.0 / math.pi

        # Compute local mach
        atm.true_airspeed = w
        mach_local = atm.mach

        # Apply the compressibility corrections for cl and cd
        if (alpha > max(alpha_element)) or (alpha < min(alpha_element)):
            out_of_polars = True
        else:
            out_of_polars = False
        cl = np.interp(alpha, alpha_element, cl_element)
        cd = np.interp(alpha, alpha_element, cd_element)
        if mach_local < 1:
            beta = math.sqrt(1 - mach_local ** 2.0)
            # cl = cl / (beta + (1 - beta) * cl / 2)
            # Prandtl-Glauert correction as Karman-Tsien and Laitone only apply to the pressure coefficient distribution
            cl = cl / beta
        else:
            beta = math.sqrt(mach_local ** 2.0 - 1)
            cl = cl / beta
            cd = cd / beta

        # Calculate force and momentum
        dT = 0.5 * blades_number * chord * w ** 2.0 * (cl * math.cos(phi) - cd * math.sin(phi))
        dQ = (
            0.5
            * blades_number
            * chord
            * w ** 2.0
            * (cl * math.sin(phi) + cd * math.cos(phi))
            * radius
        )

        # Store results
        f = np.empty(4)
        f[0] = dT
        f[1] = dQ
        f[2] = alpha
        f[3] = out_of_polars

        return f

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
        The core of the Propeller code. Given the geometry of a propeller element, its aerodynamic polars, flight
        conditions and axial/tangential velocities it computes the thrust and the torque produced using force and
        momentum with disk theory.

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
        phi = math.atan(v_ax / (omega * radius - v_t) * math.cos(sweep * math.pi / 180.0))

        # f_tip is the tip loose factor
        f_tip = (
            2
            / math.pi
            * math.acos(
                math.exp(
                    -blades_number
                    / 2
                    * (
                        (radius_max - radius)
                        / radius
                        * math.sqrt(
                            1
                            + (omega * radius / ((v_inf + v_i) + 1e-12 * ((v_inf + v_i) == 0.0)))
                            ** 2.0
                        )
                    )
                )
            )
        )

        # f_hub is the hub loose factor FIXME: to be activated in future versions
        # if phi > 0.0:
        #     f_hub = min(1.0, 2 / math.pi \
        #         * math.acos(
        #             math.exp(
        #                     -blades_number / 2 * (radius - radius_min) / (radius * math.sin(phi))))
        #                 )
        # else:
        #     f_hub = 1.0
        f_hub = 1.0

        # Calculate force and momentum
        dT = 4.0 * math.pi * radius * (v_inf + v_i) * v_i * f_tip * f_hub
        dQ = 4.0 * math.pi * radius ** 2.0 * (v_inf + v_i) * v_t * f_tip * f_hub

        # Store results
        f = np.empty(2)
        f[0] = dT
        f[1] = dQ

        return f

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
    ):
        """
        The core of the Propeller code. Given the geometry of a propeller element, its aerodynamic polars, flight
        conditions and axial/tangential velocities it computes the thrust and the torque produced using force and
        momentum with disk theory.

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

        :return: The difference between BEM dual methods for dT/(rho*dr) and dQ/ increments.
        """

        f1 = self.bem_theory(
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
        )

        f2 = self.disk_theory(
            speed_vect, radius, radius_min, radius_max, blades_number, sweep, omega, v_inf
        )

        return f1[0:1] - f2

    @staticmethod
    def read_polar_result(airfoil_name):
        result_file = pth.join(xfoil.__path__[0], "resources", airfoil_name + "_30S.csv")
        mach = 0.0
        reynolds = 1e6
        data_saved = pd.read_csv(result_file)
        values = data_saved.to_numpy()[:, 1 : len(data_saved.to_numpy()[0])]
        labels = data_saved.to_numpy()[:, 0].tolist()
        data_saved = pd.DataFrame(values, index=labels)
        index_mach = np.where(data_saved.loc["mach", :].to_numpy() == str(mach))[0]
        data_reduced = data_saved.loc[labels, index_mach]
        # Search if this exact reynolds has been computed and save results
        reynolds_vect = np.array(
            [float(x) for x in list(data_reduced.loc["reynolds", :].to_numpy())]
        )
        index_reynolds = index_mach[np.where(reynolds_vect == reynolds)[0]]
        if len(index_reynolds) == 1:
            interpolated_result = data_reduced.loc[labels, index_reynolds]
        # Else search for lower/upper Reynolds
        else:
            lower_reynolds = reynolds_vect[np.where(reynolds_vect < reynolds)[0]]
            upper_reynolds = reynolds_vect[np.where(reynolds_vect > reynolds)[0]]
            if not (len(lower_reynolds) == 0 or len(upper_reynolds) == 0):
                index_lower_reynolds = index_mach[np.where(reynolds_vect == max(lower_reynolds))[0]]
                index_upper_reynolds = index_mach[np.where(reynolds_vect == min(upper_reynolds))[0]]
                interpolated_result = data_reduced.loc[labels, index_lower_reynolds]
                # Calculate reynolds interval ratio
                x_ratio = (min(upper_reynolds) - reynolds) / (
                    min(upper_reynolds) - max(lower_reynolds)
                )
                # Search for common alpha range
                alpha_lower = np.array(
                    np.matrix(data_reduced.loc["alpha", index_lower_reynolds].to_numpy()[0])
                ).ravel()
                alpha_upper = np.array(
                    np.matrix(data_reduced.loc["alpha", index_upper_reynolds].to_numpy()[0])
                ).ravel()
                alpha_shared = np.array(list(set(alpha_upper).intersection(alpha_lower)))
                data_reduced.loc["alpha", index_lower_reynolds] = str(alpha_shared.tolist())
                labels.remove("alpha")
                # Calculate average values (cd, cl...) with linear interpolation
                for label in labels:
                    lower_value = np.array(
                        np.matrix(data_reduced.loc[label, index_lower_reynolds].to_numpy()[0])
                    ).ravel()
                    upper_value = np.array(
                        np.matrix(data_reduced.loc[label, index_upper_reynolds].to_numpy()[0])
                    ).ravel()
                    # If values relative to alpha vector, performs interpolation with shared vector
                    if np.size(lower_value) == len(alpha_lower):
                        lower_value = np.interp(alpha_shared, np.array(alpha_lower), lower_value)
                        upper_value = np.interp(alpha_shared, np.array(alpha_upper), upper_value)
                    value = (lower_value * x_ratio + upper_value * (1 - x_ratio)).tolist()
                    interpolated_result.loc[label, index_lower_reynolds] = str(value)
        # Extract alpha, cl and cd vectors
        # noinspection PyUnboundLocalVariable
        alpha_vect = np.array(np.matrix(interpolated_result.loc["alpha", :].to_numpy()[0])).ravel()
        cl_vect = np.array(np.matrix(interpolated_result.loc["cl", :].to_numpy()[0])).ravel()
        cd_vect = np.array(np.matrix(interpolated_result.loc["cd", :].to_numpy()[0])).ravel()
        return alpha_vect, cl_vect, cd_vect
