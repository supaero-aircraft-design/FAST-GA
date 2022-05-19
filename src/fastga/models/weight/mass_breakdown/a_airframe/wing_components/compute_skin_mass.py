"""
Computes the mass of the skin based on the model presented by Raquel ALONSO
in her MAE research project report.
"""
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

import openmdao.api as om
import numpy as np

from stdatm import Atmosphere

from scipy.interpolate import interp1d


class ComputeSkinMass(om.ExplicitComponent):
    def setup(self):

        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="m")
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:thickness_ratio", val=np.nan)
        self.add_input("data:geometry:wing:taper_ratio", val=np.nan)
        self.add_input("data:geometry:wing:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:sweep_25", val=np.nan, units="rad")
        self.add_input("data:geometry:wing:aileron:chord_ratio", val=np.nan)
        self.add_input("data:geometry:wing:aileron:span_ratio", val=np.nan)
        self.add_input("data:geometry:wing:aileron:max_deflection", val=np.nan, units="rad")
        self.add_input("data:geometry:propulsion:engine:count", val=np.nan)

        self.add_input("data:mission:sizing:cs23:characteristic_speed:va", val=np.nan, units="m/s")
        self.add_input("data:mission:sizing:cs23:characteristic_speed:vc", val=np.nan, units="m/s")
        self.add_input("data:mission:sizing:cs23:characteristic_speed:vd", val=np.nan, units="m/s")

        self.add_input("data:mission:sizing:main_route:cruise:altitude", val=np.nan, units="ft")

        self.add_input(
            "data:aerodynamics:aircraft:mach_interpolation:CL_alpha_vector",
            val=np.nan,
            units="rad**-1",
            shape_by_conn=True,
            copy_shape="data:aerodynamics:aircraft:mach_interpolation:mach_vector",
        )
        self.add_input(
            "data:aerodynamics:aircraft:mach_interpolation:mach_vector",
            val=np.nan,
            shape_by_conn=True,
        )

        self.add_input(
            "settings:materials:aluminium:density",
            val=2780.0,
            units="kg/m**3",
            desc="Aluminum material density",
        )
        self.add_input(
            "settings:materials:aluminium:shear_modulus",
            val=28e9,
            units="Pa",
            desc="Aluminum shear modulus",
        )
        self.add_input(
            "settings:wing:airfoil:skin:ka",
            val=0.92,
            desc="Correction coefficient needed to account for the hypothesis of a rectangular "
            "wingbox",
        )
        self.add_input(
            "settings:wing:airfoil:skin:d_wingbox",
            val=0.4,
            desc="ratio of the wingbox working depth/airfoil chord",
        )

        self.add_output("data:weight:airframe:wing:skin:mass", units="kg")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        """
        Component that computes the skin mass necessary to react to the given linear force
        vector, according to the methodology developed by Raquel Alonso Castilla.
        """
        fus_width = inputs["data:geometry:fuselage:maximum_width"]
        fus_height = inputs["data:geometry:fuselage:maximum_height"]
        wing_span = inputs["data:geometry:wing:span"]
        wing_area = inputs["data:geometry:wing:area"]
        root_chord = inputs["data:geometry:wing:root:chord"]
        thickness_ratio = inputs["data:geometry:wing:thickness_ratio"]
        sweep_25 = inputs["data:geometry:wing:sweep_25"]

        aileron_chord_ratio = inputs["data:geometry:wing:aileron:chord_ratio"]
        aileron_span_ratio = inputs["data:geometry:wing:aileron:span_ratio"]
        aileron_max_deflection = inputs["data:geometry:wing:aileron:max_deflection"]

        cruise_alt = inputs["data:mission:sizing:main_route:cruise:altitude"]

        rho_m = inputs["settings:materials:aluminium:density"]
        shear_modulus = inputs["settings:materials:aluminium:shear_modulus"]

        # There is a slight problem when taper ratio tends to one as it induces a 0 / 0 division
        # that could not be solved by hand, consequently when taper ratio gets too close to 1. it
        # will be taken as 0.97
        taper_ratio = inputs["data:geometry:wing:taper_ratio"]
        if taper_ratio > 0.97:
            taper_ratio = 0.97

        ka = inputs["settings:wing:airfoil:skin:ka"]  # Approximates the wingbox area by a rectangle
        kl = 0.97  # Approximates the wingbox perimeter by a rectangle
        d_wingbox = inputs[
            "settings:wing:airfoil:skin:d_wingbox"
        ]  # Ratio wingbox working depth/airfoil chord

        cl_aileron = 6.1
        rho_0 = Atmosphere(0.0).density

        va = inputs["data:mission:sizing:cs23:characteristic_speed:va"]
        vc = inputs["data:mission:sizing:cs23:characteristic_speed:vc"]
        vd = inputs["data:mission:sizing:cs23:characteristic_speed:vd"]

        atm = Atmosphere(cruise_alt, altitude_in_feet=True)
        atm.equivalent_airspeed = vc

        mach_interp = inputs["data:aerodynamics:aircraft:mach_interpolation:mach_vector"]
        v_interp = []
        for mach in mach_interp:
            v_interp.append(float(mach * atm.speed_of_sound))
        cl_alpha_interp = inputs["data:aerodynamics:aircraft:mach_interpolation:CL_alpha_vector"]
        cl_alpha_fct = interp1d(
            v_interp, cl_alpha_interp, fill_value="extrapolate", kind="quadratic"
        )

        cl_alpha_ac = cl_alpha_fct(atm.true_airspeed)

        fus_radius = np.sqrt(fus_height * fus_width) / 2.0

        sweep_e = np.arctan(
            np.tan(sweep_25)
            + (1.0 - taper_ratio)
            * root_chord
            / (wing_span / 2.0 - fus_radius)
            * (25.0 - 35.0)
            / 100.0
        )
        aileron_area = wing_area * (aileron_span_ratio * aileron_chord_ratio)

        k = (1.0 - taper_ratio) * root_chord / (wing_span / 2.0)
        xe_root = -root_chord * (0.1 / (wing_span / 2.0 - fus_radius) * wing_span / 2.0 - 0.75)
        f_phi_e = 1.0 / (
            np.cos(sweep_e) ** 4.0
            * (1.0 + np.tan(sweep_e) ** 2.0 + (1.0 - xe_root / root_chord) * k * np.tan(sweep_e))
            * (1.0 + np.tan(sweep_e) ** 2.0 - xe_root / root_chord * k * np.tan(sweep_e))
        )

        dist_aileron = wing_span / 2.0 * (1 - aileron_span_ratio / 2.0)
        # Assumes that the aileron are at the outward part of the wing
        theta_f = np.arccos(2.0 * aileron_chord_ratio - 1.0)
        delta_ratio = -(np.sin(theta_f)) / (8.0 * (np.pi - theta_f + np.sin(theta_f)))
        d_delta_l = (0.75 - abs(delta_ratio)) / aileron_chord_ratio - 1.0

        # Deformation dimensioning (Hyp: skin thickness constant)
        f_theta = (
            kl
            * (thickness_ratio + d_wingbox * np.cos(sweep_e) * f_phi_e)
            * (d_delta_l * 0.5 * rho_0 * aileron_area * cl_aileron * aileron_max_deflection)
            / (
                2.0
                * (ka * d_wingbox * thickness_ratio * np.cos(sweep_e) * f_phi_e) ** 2.0
                * shear_modulus
            )
        )

        f_theta_vc = f_theta * vc ** 2.0
        f_theta_vd = f_theta * vd ** 2.0
        f_theta_va = f_theta * va ** 2.0

        k_max1 = (vc ** 3.0 * f_theta_vc - va ** 3.0 * f_theta_va) / (
            (vc - va) * (vc ** 2.0 + vc * va + va ** 2.0)
        )
        k_max2 = (27.0 * vd ** 3.0 * f_theta_vd - va ** 3.0 * f_theta_va) / (
            (3.0 * vd - va) * (9.0 * vd ** 2.0 + 3.0 * vd * va + va ** 2.0)
        )
        k_max = max(k_max1, k_max2)

        q1 = (
            6.0 * np.log(taper_ratio) * (wing_span / 2.0 - fus_radius * taper_ratio)
            + wing_span
            / 2.0
            * (2.0 * taper_ratio ** 3.0 - 3.0 * taper_ratio ** 2.0 - 6.0 * taper_ratio + 7.0)
            + fus_radius * (taper_ratio ** 3.0 + 3.0 * taper_ratio - 4.0)
        )
        q2 = (
            dist_aileron
            * aileron_area
            * cl_aileron
            * aileron_max_deflection
            * np.cos(sweep_e)
            * root_chord ** 2.0
            * (taper_ratio - 1.0) ** 3.0
        )
        e_def = (
            k_max / 12.0 * cl_alpha_ac * (wing_span / 2.0 - fus_radius) ** 2.0 * q1 / q2
        )  # skin thickness with deformation criteria

        skin_mass = (
            2.0
            * rho_m
            * e_def
            * kl
            * (thickness_ratio + d_wingbox * np.cos(sweep_e) * f_phi_e)
            * (wing_span / 2.0 - fus_radius)
            * (1.0 + taper_ratio)
            * root_chord
            / np.cos(sweep_e)
        )

        if inputs["data:geometry:propulsion:engine:count"] > 4:
            skin_mass *= 1.1

        outputs["data:weight:airframe:wing:skin:mass"] = skin_mass
