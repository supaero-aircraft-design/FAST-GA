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
from stdatm import Atmosphere

from fastga.models.aerodynamics.external.xfoil.xfoil_polar import XfoilPolar
from .propeller_core import PropellerCoreModule

_LOGGER = logging.getLogger(__name__)

J_POINTS_NUMBER = 30


@oad.RegisterOpenMDAOSystem(
    "fastga.aerodynamics.propeller.coeff_map", domain=ModelDomain.AERODYNAMICS
)
class ComputePropellerCoefficientMap(om.Group):

    """Computes propeller profiles aerodynamic coefficient and propeller coefficient map under
    the form Ct=f(J) and Cp=f(J) for various pitches."""

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
        ivc.add_output("data:aerodynamics:propeller:coefficient_map:mach", val=0.0)
        ivc.add_output("data:aerodynamics:propeller:coefficient_map:reynolds", val=1e6)
        self.add_subsystem("propeller_coeff_map_aero_conditions", ivc, promotes=["*"])
        for profile in self.options["sections_profile_name_list"]:
            self.add_subsystem(
                profile + "_polar_coeff_map",
                XfoilPolar(
                    airfoil_folder_path=self.options["airfoil_folder_path"],
                    airfoil_file=profile + ".af",
                    alpha_end=30.0,
                    activate_negative_angle=True,
                ),
                promotes=[],
            )
            self.connect(
                "data:aerodynamics:propeller:coefficient_map:mach",
                profile + "_polar_coeff_map.xfoil:mach",
            )
            self.connect(
                "data:aerodynamics:propeller:coefficient_map:reynolds",
                profile + "_polar_coeff_map.xfoil:reynolds",
            )
        self.add_subsystem(
            "propeller_coeff_map",
            _ComputePropellerCoefficientMap(
                sections_profile_position_list=self.options["sections_profile_position_list"],
                sections_profile_name_list=self.options["sections_profile_name_list"],
                elements_number=self.options["elements_number"],
            ),
            promotes_inputs=["data:*"],
            promotes_outputs=["*"],
        )

        for profile in self.options["sections_profile_name_list"]:
            self.connect(
                profile + "_polar_coeff_map.xfoil:alpha",
                "propeller_coeff_map." + profile + "_polar:alpha",
            )
            self.connect(
                profile + "_polar_coeff_map.xfoil:CL",
                "propeller_coeff_map." + profile + "_polar:CL",
            )
            self.connect(
                profile + "_polar_coeff_map.xfoil:CD",
                "propeller_coeff_map." + profile + "_polar:CD",
            )
        self.connect(
            "data:aerodynamics:propeller:coefficient_map:reynolds",
            "propeller_coeff_map.reference_reynolds",
        )


class _ComputePropellerCoefficientMap(PropellerCoreModule):
    def setup(self):

        super().setup()

        self.add_input(
            "data:aerodynamics:propeller:coefficient_map:twist_75", units="deg", val=np.nan
        )
        self.add_input(
            "data:aerodynamics:propeller:coefficient_map:altitude", units="m", val=np.nan
        )
        self.add_input(
            "data:aerodynamics:propeller:coefficient_map:max_speed", units="m/s", val=np.nan
        )
        self.add_input(
            "data:aerodynamics:propeller:coefficient_map:min_speed", units="m/s", val=np.nan
        )

        self.add_output(
            "data:aerodynamics:propeller:coefficient_map:advance_ratio", shape=J_POINTS_NUMBER
        )
        self.add_output(
            "data:aerodynamics:propeller:coefficient_map:power_coefficient",
            shape=J_POINTS_NUMBER,
        )
        self.add_output(
            "data:aerodynamics:propeller:coefficient_map:thrust_coefficient",
            shape=J_POINTS_NUMBER,
        )

        self.declare_partials(of="*", wrt="*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        _LOGGER.debug("Entering propeller computation")

        # Define init values
        omega = inputs["data:geometry:propeller:average_rpm"]
        altitude = inputs["data:aerodynamics:propeller:coefficient_map:altitude"]
        atm = Atmosphere(altitude, altitude_in_feet=False)
        v_min = inputs["data:aerodynamics:propeller:coefficient_map:min_speed"]
        v_max = inputs["data:aerodynamics:propeller:coefficient_map:max_speed"]
        speed_interp = np.linspace(v_min, v_max, J_POINTS_NUMBER)
        ct_list = np.zeros_like(speed_interp)
        cp_list = np.zeros_like(speed_interp)
        j_list = np.zeros_like(speed_interp)
        theta_75 = inputs["data:aerodynamics:propeller:coefficient_map:twist_75"]

        prop_diameter = inputs["data:geometry:propeller:diameter"]
        radius_min = inputs["data:geometry:propeller:hub_diameter"] / 2.0
        radius_max = prop_diameter / 2.0
        length = radius_max - radius_min
        elements_number = np.arange(self.options["elements_number"])
        element_length = length / self.options["elements_number"]
        radius = radius_min + (elements_number + 0.5) * element_length
        sections_profile_position_list = self.options["sections_profile_position_list"]
        sections_profile_name_list = self.options["sections_profile_name_list"]

        # Build table with aerodynamic coefficients for quicker computations down the line
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

        for idx, v_inf in enumerate(speed_interp):

            thrust, eta, _ = self.compute_pitch_performance(
                inputs, theta_75, v_inf, altitude, omega, radius, alpha_list, cl_list, cd_list
            )
            ct_local = thrust / (atm.density * (omega / 60.00) ** 2.0 * prop_diameter ** 4.0)
            ct_list[idx] = ct_local
            shaft_power = thrust * v_inf / eta
            cp_local = shaft_power / (atm.density * (omega / 60.0) ** 3.0 * prop_diameter ** 5.0)
            cp_list[idx] = cp_local
            j_local = v_inf / (omega / 60.0 * prop_diameter)
            j_list[idx] = j_local

        _LOGGER.debug("Finishing propeller computation")

        # Save results
        outputs["data:aerodynamics:propeller:coefficient_map:advance_ratio"] = j_list
        outputs["data:aerodynamics:propeller:coefficient_map:power_coefficient"] = cp_list
        outputs["data:aerodynamics:propeller:coefficient_map:thrust_coefficient"] = ct_list
