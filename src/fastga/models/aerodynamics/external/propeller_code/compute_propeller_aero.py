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
import math

import numpy as np
from scipy.optimize import fsolve

import openmdao.api as om

import fastoad.api as oad
from fastoad.module_management.constants import ModelDomain

from stdatm import Atmosphere

from fastga.models.aerodynamics.external.xfoil.xfoil_polar import XfoilPolar, POLAR_POINT_COUNT

_LOGGER = logging.getLogger(__name__)

THRUST_PTS_NB = 30
SPEED_PTS_NB = 10


@oad.RegisterOpenMDAOSystem("fastga.aerodynamics.propeller", domain=ModelDomain.AERODYNAMICS)
class ComputePropellerPerformance(om.Group):

    """Computes propeller profiles aerodynamic coefficient and propeller behaviour."""

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

    def setup(self):
        ivc = om.IndepVarComp()
        ivc.add_output("data:aerodynamics:propeller:mach", val=0.0)
        ivc.add_output("data:aerodynamics:propeller:reynolds", val=1e6)
        self.add_subsystem("propeller_aero_conditions", ivc, promotes=["*"])
        for profile in self.options["sections_profile_name_list"]:
            self.add_subsystem(
                profile + "_polar",
                XfoilPolar(
                    airfoil_file=profile + ".af",
                    alpha_end=30.0,
                    activate_negative_angle=True,
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
            ),
            promotes_inputs=["data:*"],
            promotes_outputs=["*"],
        )

        for profile in self.options["sections_profile_name_list"]:
            self.connect(
                profile + "_polar.xfoil:alpha", "propeller_aero." + profile + "_polar:alpha"
            )
            self.connect(profile + "_polar.xfoil:CL", "propeller_aero." + profile + "_polar:CL")
            self.connect(profile + "_polar.xfoil:CD", "propeller_aero." + profile + "_polar:CD")


class _ComputePropellerPerformance(om.ExplicitComponent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.theta_min = 0.0
        self.theta_max = 0.0

    def initialize(self):
        self.options.declare("sections_profile_position_list", types=list)
        self.options.declare("sections_profile_name_list", types=list)
        self.options.declare("elements_number", default=20, types=int)

    def setup(self):
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
        self.add_input("data:mission:sizing:main_route:cruise:altitude", val=np.nan, units="m")
        self.add_input("data:TLAR:v_cruise", val=np.nan, units="m/s")

        for profile in self.options["sections_profile_name_list"]:
            self.add_input(
                profile + "_polar:alpha", val=np.nan, units="deg", shape=POLAR_POINT_COUNT
            )
            self.add_input(profile + "_polar:CL", val=np.nan, shape=POLAR_POINT_COUNT)
            self.add_input(profile + "_polar:CD", val=np.nan, shape=POLAR_POINT_COUNT)

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
        self.add_output("data:aerodynamics:propeller:cruise_level:altitude", units="m")

        self.declare_partials(of="*", wrt="*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        _LOGGER.debug("Entering propeller computation")

        # Define init values
        omega = inputs["data:geometry:propeller:average_rpm"]
        v_min = 5.0
        v_max = inputs["data:TLAR:v_cruise"] * 1.2
        speed_interp = np.linspace(v_min, v_max, SPEED_PTS_NB)

        # Construct table for init of climb
        altitude = 0.0
        thrust_vect, _, eta_vect = self.construct_table(inputs, speed_interp, altitude, omega)
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
        # theta_vect can be obtained with construct table, it is the second input, not used as of
        # now
        thrust_vect, _, eta_vect = self.construct_table(inputs, speed_interp, altitude, omega)
        # Reformat table
        thrust_limit, thrust_interp, efficiency_interp = self.reformat_table(thrust_vect, eta_vect)

        _LOGGER.debug("Finishing propeller computation")

        # Save results
        outputs["data:aerodynamics:propeller:cruise_level:efficiency"] = efficiency_interp
        outputs["data:aerodynamics:propeller:cruise_level:thrust"] = thrust_interp
        outputs["data:aerodynamics:propeller:cruise_level:thrust_limit"] = thrust_limit
        outputs["data:aerodynamics:propeller:cruise_level:speed"] = speed_interp
        outputs["data:aerodynamics:propeller:cruise_level:altitude"] = inputs[
            "data:mission:sizing:main_route:cruise:altitude"
        ]

    def compute_extreme_pitch(self, inputs, v_inf):
        """For a given flight speed computes the min and max possible value of theta at .75 r/R."""
        radius_ratio_vect = inputs["data:geometry:propeller:radius_ratio_vect"]
        omega = inputs["data:geometry:propeller:average_rpm"]
        phi_vect = (
            180.0
            / math.pi
            * np.arctan((v_inf + 60.0 / v_inf) / (omega * 2.0 * math.pi / 60.0 * radius_ratio_vect))
        )
        phi_75 = np.interp(0.75, radius_ratio_vect, phi_vect)
        self.theta_min = phi_75 - 10.0
        self.theta_max = phi_75 + 25.0

    def construct_table(self, inputs, speed_interp, altitude, omega):
        """
        Computes the propeller characteristics in the given flight conditions for various
        pitches. These tables will then be reformatted to fit the OpenMDAO formalism.

        :param inputs: the inputs containing the propeller geometry.
        :param speed_interp: the array containing the flight speed at which we compute the
        propeller thrust and efficiency, in m/s.
        :param altitude: the altitude for the propeller computation, in m.
        :param omega: the propeller rotation speed, in rpm.
        """
        thrust_vect = []
        theta_vect = []
        eta_vect = []

        radius_min = inputs["data:geometry:propeller:hub_diameter"] / 2.0
        radius_max = inputs["data:geometry:propeller:diameter"] / 2.0
        length = radius_max - radius_min
        elements_number = np.arange(self.options["elements_number"])
        element_length = length / self.options["elements_number"]
        radius = radius_min + (elements_number + 0.5) * element_length
        sections_profile_position_list = self.options["sections_profile_position_list"]
        sections_profile_name_list = self.options["sections_profile_name_list"]

        alpha_interp = np.array([0])

        for profile in self.options["sections_profile_name_list"]:
            alpha_interp = np.union1d(alpha_interp, inputs[profile + "_polar:alpha"])

        alpha_list = np.zeros((len(radius), len(alpha_interp)))
        cl_list = np.zeros((len(radius), len(alpha_interp)))
        cd_list = np.zeros((len(radius), len(alpha_interp)))

        for idx, _ in enumerate(radius):

            index = np.where(sections_profile_position_list < (radius[idx] / radius_max))[0]
            if index is None:
                profile_name = sections_profile_name_list[0]
            else:
                profile_name = sections_profile_name_list[int(index[-1])]

            # Load profile polars
            alpha_element, cl_element, cd_element = self.reshape_polar(
                inputs[profile_name + "_polar:alpha"],
                inputs[profile_name + "_polar:CL"],
                inputs[profile_name + "_polar:CD"],
            )

            alpha_list[idx, :] = alpha_interp
            cl_list[idx, :] = np.interp(alpha_interp, alpha_element, cl_element)
            cd_list[idx, :] = np.interp(alpha_interp, alpha_element, cd_element)

        for v_inf in speed_interp:
            self.compute_extreme_pitch(inputs, v_inf)
            theta_interp = np.linspace(self.theta_min, self.theta_max, 100)
            local_thrust_vect = []
            local_theta_vect = []
            local_eta_vect = []
            for theta_75 in theta_interp:
                thrust, eta, _ = self.compute_pitch_performance(
                    inputs, theta_75, v_inf, altitude, omega, radius, alpha_list, cl_list, cd_list
                )
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
            # Testing a non empty sequence with an if will return True if it is not empty see
            # PEP8 recommended method
            if list(np.where(thrust_difference[idx_in_zone:] < 0.0)[0]):
                idx_end = np.min(np.where(thrust_difference[idx_in_zone:] < 0.0)) + idx_in_zone
                local_thrust_vect = local_thrust_vect[0 : idx_end + 1]
                local_theta_vect = local_theta_vect[0 : idx_end + 1]
                local_eta_vect = local_eta_vect[0 : idx_end + 1]
            # Erase start of the curve if thrust is negative or decreases
            idx_start = 0
            thrust_difference = np.array(local_thrust_vect[1:]) - np.array(local_thrust_vect[0:-1])
            # Testing a non empty sequence with an if will return True if it is not empty see
            # PEP8 recommended method
            if list(np.where(np.array(local_thrust_vect) < 0.0)[0]):
                idx_start = int(np.max(np.where(np.array(local_thrust_vect) < 0.0)))
            # Testing a non empty sequence with an if will return True if it is not empty see
            # PEP8 recommended method
            if list(np.where(thrust_difference < 0.0)[0]):
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
            #          v_inf * np.array(local_thrust_vect)
            #          /
            #          (np.array(local_eta_vect) * omega * math.pi / 30.0))

        return thrust_vect, theta_vect, eta_vect

    @staticmethod
    def reformat_table(thrust_vect, eta_vect):
        """
        Reformat the table by intersecting the thrust array and interpolating the
        corresponding efficiencies.
        """
        min_thrust = 0.0
        max_thrust = 0.0
        thrust_limit = []
        for idx, _ in enumerate(thrust_vect):
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
        :param radius: radius of discretized blade element [m].
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
        length = radius_max - radius_min
        element_length = length / self.options["elements_number"]
        omega = omega * math.pi / 30.0
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

        :return: The calculated dT/(rho*dr) and dQ/(rho*dr) increments with BEM method.
        """

        # Extract axial/tangential speeds
        v_i = speed_vect[0]
        v_t = speed_vect[1]

        # Calculate speed composition and relative air angle (in deg.)
        v_ax = v_inf + v_i
        v_t = (omega * radius - v_t) * math.cos(sweep * math.pi / 180.0)
        rel_fluid_speed = math.sqrt(v_ax ** 2.0 + v_t ** 2.0)
        phi = math.atan(v_ax / v_t)
        alpha = theta - phi * 180.0 / math.pi

        # Compute local mach
        atm.true_airspeed = rel_fluid_speed
        mach_local = atm.mach

        # Apply the compressibility corrections for cl and cd
        out_of_polars = bool((alpha > max(alpha_element)) or (alpha < min(alpha_element)))

        c_l = np.interp(alpha, alpha_element, cl_element)
        c_d = np.interp(alpha, alpha_element, cd_element)
        if mach_local < 1:
            beta = math.sqrt(1 - mach_local ** 2.0)
            c_l = c_l / beta
        else:
            beta = math.sqrt(mach_local ** 2.0 - 1)
            c_l = c_l / beta
            c_d = c_d / beta

        # Calculate force and momentum
        thrust_element = (
            0.5
            * blades_number
            * chord
            * rel_fluid_speed ** 2.0
            * (c_l * math.cos(phi) - c_d * math.sin(phi))
        )
        torque_element = (
            0.5
            * blades_number
            * chord
            * rel_fluid_speed ** 2.0
            * (c_l * math.sin(phi) + c_d * math.cos(phi))
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
            / math.pi
            * math.acos(
                math.exp(
                    -blades_number
                    / 2
                    * (
                        (radius_max - radius)
                        / radius
                        * math.sqrt(1 + (omega * radius / (v_ax + 1e-12 * (v_ax == 0.0))) ** 2.0)
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
        thrust_element = 4.0 * math.pi * radius * (v_inf + v_i) * v_i * f_tip * f_hub
        torque_element = 4.0 * math.pi * radius ** 2.0 * (v_inf + v_i) * v_t * f_tip * f_hub

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
