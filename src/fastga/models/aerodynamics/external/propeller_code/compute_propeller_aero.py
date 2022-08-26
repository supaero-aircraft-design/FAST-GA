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

import openmdao.api as om

import fastoad.api as oad
from fastoad.module_management.constants import ModelDomain

from fastga.models.aerodynamics.external.xfoil.xfoil_polar import XfoilPolar

from .propeller_core import PropellerCoreModule

_LOGGER = logging.getLogger(__name__)

THRUST_PTS_NB = 30
SPEED_PTS_NB = 10


@oad.RegisterOpenMDAOSystem("fastga.aerodynamics.propeller", domain=ModelDomain.AERODYNAMICS)
class ComputePropellerPerformance(om.Group):

    """Computes propeller profiles aerodynamic coefficient and propeller behaviour."""

    def initialize(self):
        self.options.declare("airfoil_folder_path", default=None, types=str, allow_none=True)
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
        self.add_subsystem("propeller_efficiency_aero_conditions", ivc, promotes=["*"])
        for profile in self.options["sections_profile_name_list"]:
            self.add_subsystem(
                profile + "_polar_efficiency",
                XfoilPolar(
                    airfoil_folder_path=self.options["airfoil_folder_path"],
                    airfoil_file=profile + ".af",
                    alpha_end=30.0,
                    activate_negative_angle=True,
                ),
                promotes=[],
            )
            self.connect(
                "data:aerodynamics:propeller:mach",
                profile + "_polar_efficiency.xfoil:mach",
            )
            self.connect(
                "data:aerodynamics:propeller:reynolds",
                profile + "_polar_efficiency.xfoil:reynolds",
            )
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
                profile + "_polar_efficiency.xfoil:alpha",
                "propeller_aero." + profile + "_polar:alpha",
            )
            self.connect(
                profile + "_polar_efficiency.xfoil:CL", "propeller_aero." + profile + "_polar:CL"
            )
            self.connect(
                profile + "_polar_efficiency.xfoil:CD", "propeller_aero." + profile + "_polar:CD"
            )
        self.connect("data:aerodynamics:propeller:reynolds", "propeller_aero.reference_reynolds")


class _ComputePropellerPerformance(PropellerCoreModule):
    def setup(self):

        super().setup()
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

        # Reformat table
        thrust_limit, thrust_interp, efficiency_interp = self.reformat_table(thrust_vect, eta_vect)

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
