"""
Computes the mass of the web based on the model presented by Raquel ALONSO
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

from scipy.integrate import trapz

from fastga.models.load_analysis.wing.aerostructural_loads import AerostructuralLoad

from stdatm import Atmosphere


class ComputeWebMass(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("min_fuel_in_wing", default=False, types=bool)

    def setup(self):

        self.add_input("data:geometry:flap:chord_ratio", val=np.nan)
        self.add_input("data:geometry:wing:aileron:chord_ratio", val=np.nan)
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="m")
        self.add_input("data:geometry:landing_gear:y", val=np.nan, units="m")
        self.add_input("data:geometry:landing_gear:type", val=np.nan)
        self.add_input("data:geometry:propulsion:engine:layout", val=np.nan)
        self.add_input("data:geometry:propulsion:engine:count", val=np.nan)
        self.add_input(
            "data:geometry:propulsion:engine:y_ratio",
            shape_by_conn=True,
        )
        self.add_input("data:geometry:propulsion:tank:y_ratio_tank_end", val=np.nan)
        self.add_input("data:geometry:propulsion:tank:y_ratio_tank_beginning", val=np.nan)
        self.add_input("data:geometry:propulsion:tank:LE_chord_percentage", val=np.nan)
        self.add_input("data:geometry:propulsion:tank:TE_chord_percentage", val=np.nan)
        self.add_input("data:geometry:propulsion:nacelle:width", val=np.nan, units="m")
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:thickness_ratio", val=np.nan)
        self.add_input("data:geometry:wing:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:thickness_ratio", val=np.nan)
        self.add_input("data:geometry:wing:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:thickness_ratio", val=np.nan)
        self.add_input("data:geometry:wing:taper_ratio", val=np.nan)
        self.add_input("data:geometry:wing:sweep_25", val=np.nan, units="rad")

        self.add_input("data:mission:sizing:main_route:cruise:altitude", val=np.nan, units="ft")
        self.add_input("data:mission:sizing:cs23:characteristic_speed:vc", val=np.nan, units="m/s")

        self.add_input(
            "data:aerodynamics:wing:low_speed:Y_vector",
            val=np.nan,
            shape_by_conn=True,
            units="m",
        )
        self.add_input(
            "data:aerodynamics:wing:low_speed:chord_vector",
            val=np.nan,
            shape_by_conn=True,
            copy_shape="data:aerodynamics:wing:low_speed:Y_vector",
            units="m",
        )
        self.add_input(
            "data:aerodynamics:wing:low_speed:CL_vector",
            val=np.nan,
            shape_by_conn=True,
            copy_shape="data:aerodynamics:wing:low_speed:Y_vector",
        )
        self.add_input(
            "data:aerodynamics:slipstream:wing:cruise:only_prop:CL_vector",
            val=np.nan,
            shape_by_conn=True,
            copy_shape="data:aerodynamics:slipstream:wing:cruise:prop_on:Y_vector",
        )
        self.add_input(
            "data:aerodynamics:slipstream:wing:cruise:prop_on:Y_vector",
            val=np.nan,
            shape_by_conn=True,
            units="m",
        )
        self.add_input("data:aerodynamics:wing:low_speed:CL0_clean", val=np.nan)
        self.add_input(
            "data:aerodynamics:slipstream:wing:cruise:prop_on:velocity", val=np.nan, units="m/s"
        )

        self.add_input("data:weight:propulsion:engine:mass", val=np.nan, units="kg")
        self.add_input("data:weight:airframe:landing_gear:main:mass", val=np.nan, units="kg")
        self.add_input(
            "data:weight:airframe:wing:punctual_mass:y_ratio",
            shape_by_conn=True,
            val=0.0,
        )
        self.add_input(
            "data:weight:airframe:wing:punctual_mass:mass",
            shape_by_conn=True,
            copy_shape="data:weight:airframe:wing:punctual_mass:y_ratio",
            units="kg",
            val=0.0,
        )

        self.add_input("data:mission:sizing:cs23:safety_factor", val=np.nan)

        self.add_input("settings:geometry:fuel_tanks:depth", val=np.nan)

        self.add_input(
            "settings:materials:aluminium:density",
            val=2780.0,
            units="kg/m**3",
            desc="Aluminum material density",
        )
        self.add_input(
            "settings:materials:aluminium:max_shear_stress",
            val=165e6,
            units="Pa",
            desc="Aluminum maximum shear stress",
        )

        self.add_input("data:weight:airframe:wing:mass", val=np.nan, units="kg")

        self.add_input("data:weight:aircraft:MZFW", val=np.nan, units="kg")

        if not self.options["min_fuel_in_wing"]:
            self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="kg")
            self.add_input(
                "data:mission:sizing:cs23:sizing_factor:ultimate_mtow:positive", val=np.nan
            )
            self.add_input(
                "data:mission:sizing:cs23:sizing_factor:ultimate_mtow:negative", val=np.nan
            )

            self.add_output("data:weight:airframe:wing:web:mass:max_fuel_in_wing", units="kg")
        else:
            self.add_input(
                "data:mission:sizing:cs23:sizing_factor:ultimate_mzfw:positive", val=np.nan
            )
            self.add_input(
                "data:mission:sizing:cs23:sizing_factor:ultimate_mzfw:negative", val=np.nan
            )

            self.add_output("data:weight:airframe:wing:web:mass:min_fuel_in_wing", units="kg")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        """
        Component that computes the wing web mass necessary to react to the given linear force
        vector, according to the methodology developed by Raquel Alonso Castilla.
        """
        wing_mass = inputs["data:weight:airframe:wing:mass"]

        if not self.options["min_fuel_in_wing"]:
            mass = inputs["data:weight:aircraft:MTOW"]
            load_factor_pos = inputs[
                "data:mission:sizing:cs23:sizing_factor:ultimate_mtow:positive"
            ]
            load_factor_neg = inputs[
                "data:mission:sizing:cs23:sizing_factor:ultimate_mtow:negative"
            ]
        else:
            mass = inputs["data:weight:aircraft:MZFW"]
            load_factor_pos = inputs[
                "data:mission:sizing:cs23:sizing_factor:ultimate_mzfw:positive"
            ]
            load_factor_neg = inputs[
                "data:mission:sizing:cs23:sizing_factor:ultimate_mzfw:negative"
            ]

        mzfw = inputs["data:weight:aircraft:MZFW"]
        fuel_mass = max(0, mass - mzfw)
        load_factor = max(load_factor_pos, abs(load_factor_neg))

        fus_width = inputs["data:geometry:fuselage:maximum_width"]
        fus_height = inputs["data:geometry:fuselage:maximum_height"]
        wing_span = inputs["data:geometry:wing:span"]
        wing_area = inputs["data:geometry:wing:area"]
        root_chord = inputs["data:geometry:wing:root:chord"]
        tip_chord = inputs["data:geometry:wing:tip:chord"]
        taper_ratio = inputs["data:geometry:wing:taper_ratio"]
        sweep_25 = inputs["data:geometry:wing:sweep_25"]

        y_vector = inputs["data:aerodynamics:wing:low_speed:Y_vector"]
        y_vector_slip = inputs["data:aerodynamics:slipstream:wing:cruise:prop_on:Y_vector"]
        cl_vector = inputs["data:aerodynamics:wing:low_speed:CL_vector"]
        cl_vector_slip = inputs["data:aerodynamics:slipstream:wing:cruise:only_prop:CL_vector"]
        cl_0 = inputs["data:aerodynamics:wing:low_speed:CL0_clean"]
        chord_vector = inputs["data:aerodynamics:wing:low_speed:chord_vector"]
        v_ref = inputs["data:aerodynamics:slipstream:wing:cruise:prop_on:velocity"]

        cruise_alt = inputs["data:mission:sizing:main_route:cruise:altitude"]

        rho_m = inputs["settings:materials:aluminium:density"]
        max_shear_stress = inputs["settings:materials:aluminium:max_shear_stress"]

        safety_factor = inputs["data:mission:sizing:cs23:safety_factor"]

        # STEP 1 - DELETE THE ADDITIONAL ZEROS WE HAD TO PUT TO FIT OPENMDAO AND ADD A POINT AT
        # THE ROOT (Y=0) AND AT THE VERY TIP (Y=SPAN/2) TO GET THE WHOLE SPAN OF THE WING IN THE
        # INTERPOLATION WE WILL DO LATER

        semi_span = wing_span / 2.0

        atm = Atmosphere(cruise_alt, altitude_in_feet=True)
        atm.equivalent_airspeed = inputs["data:mission:sizing:cs23:characteristic_speed:vc"]

        # We delete the zeros
        y_vector = AerostructuralLoad.delete_additional_zeros(y_vector)
        y_vector_slip = AerostructuralLoad.delete_additional_zeros(y_vector_slip)
        cl_vector = AerostructuralLoad.delete_additional_zeros(cl_vector)
        cl_vector_slip = AerostructuralLoad.delete_additional_zeros(cl_vector_slip)
        chord_vector = AerostructuralLoad.delete_additional_zeros(chord_vector)

        # We add the first point at the root
        y_vector, _ = AerostructuralLoad.insert_in_sorted_array(y_vector, 0.0)
        y_vector_slip, _ = AerostructuralLoad.insert_in_sorted_array(y_vector_slip, 0.0)
        cl_vector = np.insert(cl_vector, 0, cl_vector[0])
        cl_vector_slip = np.insert(cl_vector_slip, 0, cl_vector_slip[0])
        chord_vector = np.insert(chord_vector, 0, root_chord)

        # And the last point at the tip
        y_vector_orig, _ = AerostructuralLoad.insert_in_sorted_array(y_vector, semi_span)
        y_vector_slip_orig, _ = AerostructuralLoad.insert_in_sorted_array(y_vector_slip, semi_span)
        cl_vector_orig = np.append(cl_vector, 0.0)
        cl_vector_slip = np.append(cl_vector_slip, 0.0)
        chord_vector_orig = np.append(chord_vector, tip_chord)

        fus_radius = np.sqrt(fus_height * fus_width) / 2.0

        sweep_e = np.arctan(
            np.tan(sweep_25)
            + (1.0 - taper_ratio)
            * root_chord
            / (wing_span / 2.0 - fus_radius)
            * (25.0 - 35.0)
            / 100.0
        )

        v_c_tas = atm.true_airspeed

        dynamic_pressure = 1.0 / 2.0 * atm.density * v_c_tas ** 2.0

        y_vector, weight_array_orig = AerostructuralLoad.compute_relief_force(
            inputs, y_vector_orig, chord_vector_orig, wing_mass, fuel_mass
        )
        cl_s = AerostructuralLoad.compute_cl_s(
            y_vector_orig, y_vector_orig, y_vector, cl_vector_orig, chord_vector_orig
        )
        cl_s_slip = AerostructuralLoad.compute_cl_s(
            y_vector_slip_orig, y_vector_orig, y_vector, cl_vector_slip, chord_vector_orig
        )

        cl_wing = 1.05 * (load_factor * mass * 9.81) / (dynamic_pressure * wing_area)
        cl_s_actual = cl_s * cl_wing / cl_0
        cl_s_slip_actual = safety_factor * cl_s_slip * (v_ref / v_c_tas) ** 2.0
        lift_section = dynamic_pressure * (cl_s_actual + cl_s_slip_actual)
        weight_array = weight_array_orig * load_factor
        tot_force_array = lift_section + weight_array

        shear_vector = AerostructuralLoad.compute_shear_diagram(y_vector, tot_force_array)
        web_surface = shear_vector / max_shear_stress
        web_mass = abs(2.0 * rho_m / np.cos(sweep_e) * trapz(web_surface, y_vector))

        if inputs["data:geometry:propulsion:engine:count"] > 4:
            web_mass *= 1.1

        if not self.options["min_fuel_in_wing"]:
            outputs["data:weight:airframe:wing:web:mass:max_fuel_in_wing"] = web_mass
        else:
            outputs["data:weight:airframe:wing:web:mass:min_fuel_in_wing"] = web_mass
