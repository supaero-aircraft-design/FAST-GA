"""
Computes the aerostructural loads on the wing of the aircraft.
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

import numpy as np
import openmdao.api as om
from scipy.integrate import trapz
from scipy.interpolate import interp1d

from stdatm import Atmosphere
import fastoad.api as oad

from fastga.models.aerodynamics.constants import SPAN_MESH_POINT
from fastga.models.geometry.geom_components.wing_tank.compute_mfw_advanced import (
    tank_volume_distribution,
)

from .constants import SUBMODEL_AEROSTRUCTURAL_LOADS, NB_POINTS_POINT_MASS, POINT_MASS_SPAN_RATIO

SPAN_MESH_POINT_LOADS = int(1.5 * SPAN_MESH_POINT)


@oad.RegisterSubmodel(
    SUBMODEL_AEROSTRUCTURAL_LOADS, "fastga.submodel.loads.wings.aerostructural.legacy"
)
class AerostructuralLoad(om.ExplicitComponent):
    def setup(self):

        self.add_input("data:TLAR:category", val=3.0)
        self.add_input("data:TLAR:level", val=2.0)
        self.add_input("data:TLAR:v_max_sl", val=np.nan, units="kn")
        self.add_input("data:TLAR:v_cruise", val=np.nan, units="m/s")
        self.add_input("data:TLAR:v_approach", val=np.nan, units="m/s")

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
        self.add_input(
            "data:aerodynamics:slipstream:wing:cruise:prop_on:velocity", val=np.nan, units="m/s"
        )
        self.add_input("data:aerodynamics:wing:low_speed:CL0_clean", val=np.nan)
        self.add_input("data:aerodynamics:wing:cruise:CL0_clean", val=np.nan)
        self.add_input("data:aerodynamics:wing:cruise:CM0_clean", val=np.nan)
        self.add_input("data:aerodynamics:horizontal_tail:efficiency", val=np.nan)
        self.add_input("data:aerodynamics:aircraft:landing:CL_max", val=np.nan)
        self.add_input("data:aerodynamics:wing:low_speed:CL_max_clean", val=np.nan)
        self.add_input("data:aerodynamics:wing:low_speed:CL_min_clean", val=np.nan)
        self.add_input("data:aerodynamics:wing:low_speed:CL_alpha", val=np.nan, units="rad**-1")
        self.add_input("data:aerodynamics:wing:cruise:CL_alpha", val=np.nan, units="rad**-1")
        self.add_input(
            "data:aerodynamics:horizontal_tail:low_speed:CL_alpha", val=np.nan, units="rad**-1"
        )
        self.add_input(
            "data:aerodynamics:horizontal_tail:cruise:CL_alpha", val=np.nan, units="rad**-1"
        )
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

        self.add_input("data:weight:aircraft:MZFW", val=np.nan, units="kg")
        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="kg")
        self.add_input("data:weight:propulsion:engine:mass", val=np.nan, units="kg")
        self.add_input("data:weight:airframe:landing_gear:main:mass", val=np.nan, units="kg")
        self.add_input("data:weight:airframe:wing:mass", val=np.nan, units="kg")
        self.add_input("data:weight:aircraft:CG:aft:x", val=np.nan, units="m")
        self.add_input("data:weight:aircraft:CG:fwd:x", val=np.nan, units="m")

        self.add_input("data:geometry:wing:root:virtual_chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:thickness_ratio", val=np.nan)
        self.add_input("data:geometry:wing:tip:thickness_ratio", val=np.nan)
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:MAC:leading_edge:x:local", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
        self.add_input(
            "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="m"
        )
        self.add_input("data:geometry:flap:chord_ratio", val=np.nan)
        self.add_input("data:geometry:wing:aileron:chord_ratio", val=np.nan)
        self.add_input("data:geometry:fuselage:length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:landing_gear:y", val=np.nan, units="m")
        self.add_input("data:geometry:landing_gear:type", val=np.nan)
        self.add_input("data:geometry:propulsion:engine:layout", val=np.nan)
        self.add_input("data:geometry:propulsion:engine:count", val=np.nan)
        self.add_input(
            "data:geometry:propulsion:engine:y_ratio",
            shape_by_conn=True,
        )
        self.add_input("data:geometry:propulsion:nacelle:width", val=np.nan, units="m")
        self.add_input("data:geometry:propulsion:tank:y_ratio_tank_end", val=np.nan)
        self.add_input("data:geometry:propulsion:tank:y_ratio_tank_beginning", val=np.nan)
        self.add_input("data:geometry:propulsion:tank:LE_chord_percentage", val=np.nan)
        self.add_input("data:geometry:propulsion:tank:TE_chord_percentage", val=np.nan)

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

        self.add_input("data:mission:sizing:fuel", val=np.nan, units="kg")
        self.add_input("data:mission:sizing:main_route:cruise:altitude", val=np.nan, units="ft")

        self.add_input("data:mission:sizing:cs23:characteristic_speed:vc", val=np.nan, units="m/s")
        self.add_input("data:mission:sizing:cs23:safety_factor", val=np.nan)
        self.add_input("data:mission:sizing:cs23:sizing_factor:ultimate_mtow:positive", val=np.nan)
        self.add_input("data:mission:sizing:cs23:sizing_factor:ultimate_mtow:negative", val=np.nan)
        self.add_input("data:mission:sizing:cs23:sizing_factor:ultimate_mzfw:positive", val=np.nan)
        self.add_input("data:mission:sizing:cs23:sizing_factor:ultimate_mzfw:negative", val=np.nan)

        self.add_input("settings:geometry:fuel_tanks:depth", val=np.nan)

        self.add_output("data:loads:max_shear:mass", units="kg")
        self.add_output("data:loads:max_shear:load_factor")
        self.add_output("data:loads:max_shear:lift_shear", units="N", shape=SPAN_MESH_POINT_LOADS)
        self.add_output("data:loads:max_shear:weight_shear", units="N", shape=SPAN_MESH_POINT_LOADS)

        self.add_output("data:loads:max_rbm:mass", units="kg")
        self.add_output("data:loads:max_rbm:load_factor")
        self.add_output("data:loads:max_rbm:lift_rbm", units="N*m", shape=SPAN_MESH_POINT_LOADS)
        self.add_output("data:loads:max_rbm:weight_rbm", units="N*m", shape=SPAN_MESH_POINT_LOADS)

        self.add_output("data:loads:y_vector", units="m", shape=SPAN_MESH_POINT_LOADS)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        # STEP 1/XX - DEFINE OR CALCULATE INPUT DATA FOR LOAD COMPUTATION ##########################
        ############################################################################################

        y_vector = inputs["data:aerodynamics:wing:low_speed:Y_vector"]
        y_vector_slip = inputs["data:aerodynamics:slipstream:wing:cruise:prop_on:Y_vector"]
        cl_vector = inputs["data:aerodynamics:wing:low_speed:CL_vector"]
        cl_vector_slip = inputs["data:aerodynamics:slipstream:wing:cruise:only_prop:CL_vector"]
        cl_0 = inputs["data:aerodynamics:wing:low_speed:CL0_clean"]
        chord_vector = inputs["data:aerodynamics:wing:low_speed:chord_vector"]
        v_ref = inputs["data:aerodynamics:slipstream:wing:cruise:prop_on:velocity"]

        semi_span = inputs["data:geometry:wing:span"] / 2.0
        wing_area = inputs["data:geometry:wing:area"]

        mtow = inputs["data:weight:aircraft:MTOW"]
        mzfw = inputs["data:weight:aircraft:MZFW"]
        wing_mass = inputs["data:weight:airframe:wing:mass"]

        cruise_alt = inputs["data:mission:sizing:main_route:cruise:altitude"]

        cruise_v_tas = inputs["data:TLAR:v_cruise"]
        v_c = inputs["data:mission:sizing:cs23:characteristic_speed:vc"]

        factor_of_safety = float(inputs["data:mission:sizing:cs23:safety_factor"])
        shear_max_conditions = []
        rbm_max_conditions = []

        atm = Atmosphere(cruise_alt)

        shear_max = 0.0
        rbm_max = 0.0

        # STEP 2/XX - DELETE THE ADDITIONAL ZEROS WE HAD TO PUT TO FIT OPENMDAO AND ADD A POINT
        # AT THE ROOT (Y=0) AND AT THE VERY TIP (Y=SPAN/2) TO GET THE WHOLE SPAN OF THE WING IN
        # THE INTERPOLATION WE WILL DO LATER

        # We delete the zeros
        y_vector = AerostructuralLoad.delete_additional_zeros(y_vector)
        y_vector_slip = AerostructuralLoad.delete_additional_zeros(y_vector_slip)
        cl_vector = AerostructuralLoad.delete_additional_zeros(cl_vector, len(y_vector))
        cl_vector_slip = AerostructuralLoad.delete_additional_zeros(
            cl_vector_slip, len(y_vector_slip)
        )
        chord_vector = AerostructuralLoad.delete_additional_zeros(chord_vector, len(y_vector))

        # We add the first point at the root
        y_vector, _ = AerostructuralLoad.insert_in_sorted_array(y_vector, 0.0)
        y_vector_slip, _ = AerostructuralLoad.insert_in_sorted_array(y_vector_slip, 0.0)
        cl_vector = np.insert(cl_vector, 0, cl_vector[0])
        cl_vector_slip = np.insert(cl_vector_slip, 0, cl_vector_slip[0])
        chord_vector = np.insert(chord_vector, 0, chord_vector[0])

        # And the last point at the tip
        y_vector_orig, _ = AerostructuralLoad.insert_in_sorted_array(y_vector, semi_span)
        y_vector_slip_orig, _ = AerostructuralLoad.insert_in_sorted_array(y_vector_slip, semi_span)
        cl_vector = np.append(cl_vector, 0.0)
        cl_vector_slip = np.append(cl_vector_slip, 0.0)
        chord_vector = np.append(chord_vector, chord_vector[-1])

        # STEP 3/XX - WE COMPUTE THE BASELINE LIFT THAT WE ASSUME WILL SCALE WITH THE LOAD
        # FACTOR, THAT IS WHY WE COMPUTE It OUT OF THE LOOPS

        y_vector, _ = self.compute_relief_force(inputs, y_vector_orig, chord_vector, wing_mass, 0.0)
        cl_s = self.compute_cl_s(y_vector_orig, y_vector_orig, y_vector, cl_vector, chord_vector)
        cl_s_slip = self.compute_cl_s(
            y_vector_slip_orig, y_vector_orig, y_vector, cl_vector_slip, chord_vector
        )

        lift_shear_diagram = np.full(len(y_vector), 0.0)
        lift_bending_diagram = np.full(len(y_vector), 0.0)
        weight_shear_diagram = np.full(len(y_vector), 0.0)
        weight_bending_diagram = np.full(len(y_vector), 0.0)

        # STEP 4/XX - WE INITIALIZE THE LOOPS ON THE DIFFERENT SIZING CASE THAT WE DEFINED AND
        # THEN LAUNCH THEM

        mass_tag_array = ["mtow", "mzfw"]

        for mass_tag in mass_tag_array:

            if mass_tag == "mtow":
                mass = mtow
                load_factor_list = [
                    float(inputs["data:mission:sizing:cs23:sizing_factor:ultimate_mtow:positive"])
                    / factor_of_safety,
                    float(inputs["data:mission:sizing:cs23:sizing_factor:ultimate_mtow:negative"])
                    / factor_of_safety,
                ]
            else:
                mass = min(mzfw, mtow)
                load_factor_list = [
                    float(inputs["data:mission:sizing:cs23:sizing_factor:ultimate_mzfw:positive"])
                    / factor_of_safety,
                    float(inputs["data:mission:sizing:cs23:sizing_factor:ultimate_mzfw:negative"])
                    / factor_of_safety,
                ]

            if abs(mtow - mzfw) < 5.0:
                fuel_mass = 0.0
            else:
                fuel_mass = mass - mzfw

            y_vector, weight_array_orig = self.compute_relief_force(
                inputs, y_vector_orig, chord_vector, wing_mass, fuel_mass
            )
            atm.true_airspeed = cruise_v_tas

            atm.equivalent_airspeed = v_c
            v_c_tas = atm.true_airspeed
            dynamic_pressure = 1.0 / 2.0 * atm.density * v_c_tas ** 2.0

            for load_factor in load_factor_list:

                # STEP 4.2/XX - WE COMPUTE THE REAL CONDITIONS EXPERIENCED IN TERMS OF LIFT AND
                # WEIGHT AND SCALE THE INITIAL VECTOR ACCORDING TO LOAD FACTOR AND LIFT EQUILIBRIUM

                cl_wing = 1.05 * (load_factor * mass * 9.81) / (dynamic_pressure * wing_area)
                cl_s_actual = cl_s * cl_wing / cl_0
                cl_s_slip_actual = cl_s_slip * (v_ref / v_c_tas) ** 2.0
                lift_section = (
                    factor_of_safety * dynamic_pressure * (cl_s_actual + cl_s_slip_actual)
                )
                weight_array = weight_array_orig * factor_of_safety * load_factor

                # STEP 4.3/XX - WE COMPUTE THE SHEAR AND WEIGHT DIAGRAM WITH THE APPROPRIATE
                # FUNCTION, IDENTIFY THE MOST EXTREME CONSTRAINTS AND SAVE THE CONDITIONS IN
                # WHICH THEY ARE EXPERIENCED FOR LATER USE IN THE POST-PROCESSING PHASE

                tot_shear_diagram = AerostructuralLoad.compute_shear_diagram(
                    y_vector, weight_array + lift_section
                )
                tot_bending_moment_diagram = AerostructuralLoad.compute_bending_moment_diagram(
                    y_vector, weight_array + lift_section
                )
                root_shear_force = tot_shear_diagram[0]
                root_bending_moment = tot_bending_moment_diagram[0]

                if abs(root_shear_force) > shear_max:
                    shear_max_conditions = [mass, load_factor]
                    lift_shear_diagram = AerostructuralLoad.compute_shear_diagram(
                        y_vector, lift_section
                    )
                    weight_shear_diagram = AerostructuralLoad.compute_shear_diagram(
                        y_vector, weight_array
                    )
                    shear_max = abs(root_shear_force)

                if abs(root_bending_moment) > rbm_max:
                    rbm_max_conditions = [mass, load_factor]
                    lift_bending_diagram = AerostructuralLoad.compute_bending_moment_diagram(
                        y_vector, lift_section
                    )
                    weight_bending_diagram = AerostructuralLoad.compute_bending_moment_diagram(
                        y_vector, weight_array
                    )
                    rbm_max = abs(root_bending_moment)

        # STEP 5/XX - WE ADD ZEROS TO THE RESULTS ARRAYS TO MAKE THEM FIT THE OPENMDAO FORMAT

        additional_zeros = np.zeros(SPAN_MESH_POINT_LOADS - len(y_vector))
        lift_shear_diagram = np.concatenate([lift_shear_diagram, additional_zeros])
        weight_shear_diagram = np.concatenate([weight_shear_diagram, additional_zeros])
        y_vector = np.concatenate([y_vector, additional_zeros])

        lift_bending_diagram = np.concatenate([lift_bending_diagram, additional_zeros])
        weight_bending_diagram = np.concatenate([weight_bending_diagram, additional_zeros])

        outputs["data:loads:max_shear:mass"] = shear_max_conditions[0]
        outputs["data:loads:max_shear:load_factor"] = shear_max_conditions[1]
        outputs["data:loads:max_shear:lift_shear"] = lift_shear_diagram
        outputs["data:loads:max_shear:weight_shear"] = weight_shear_diagram

        outputs["data:loads:max_rbm:mass"] = rbm_max_conditions[0]
        outputs["data:loads:max_rbm:load_factor"] = rbm_max_conditions[1]
        outputs["data:loads:max_rbm:lift_rbm"] = lift_bending_diagram
        outputs["data:loads:max_rbm:weight_rbm"] = weight_bending_diagram

        outputs["data:loads:y_vector"] = y_vector

    @staticmethod
    def compute_shear_diagram(y_vector, force_array):
        """
        Function that computes the shear diagram of a given array with linear forces in them

        @param y_vector: an array containing the position of the different station at which the
        linear forces are given
        @param force_array: an array containing the linear forces
        @return: shear_force_diagram an array representing the shear diagram of the linear forces
        given in input
        """

        # We first create the array to fill
        shear_force_diagram = np.zeros(len(y_vector))

        # Each station of the shear diagram is equal to the integral of the forces on all
        # subsequent station
        for i, _ in enumerate(y_vector):
            shear_force_diagram[i] = trapz(force_array[i:], y_vector[i:])

        return shear_force_diagram

    @staticmethod
    def compute_bending_moment_diagram(y_vector, force_array):
        """
        Function that computes the root bending diagram of a given array with linear forces in them

        @param y_vector: an array containing the position of the different station at which the
        linear forces are given
        @param force_array: an array containing the linear forces
        @return: bending_moment_diagram an array representing the root bending diagram of the
        linear forces given in
        input
        """

        # We first create the array to fill
        bending_moment_diagram = np.zeros(len(y_vector))

        # Each station of the shear diagram is equal to the root bending moment created by all
        # subsequent stations
        for i, _ in enumerate(y_vector):
            lever_arm = y_vector - y_vector[i]
            bending_moment_diagram[i] = trapz(force_array[i:] * lever_arm[i:], y_vector[i:])

        return bending_moment_diagram

    @staticmethod
    def compute_cl_s(y_vector_cl_orig, y_vector_chord_orig, y_vector, cl_list, chord_list):
        """
        Function that computes linear lift on all section of y_vector based on an original cl
        distribution

        @param y_vector_cl_orig: an array containing the position of the different station at which
        the original lift distribution was computed, typically a result of OpenVSP or VLM
        @param y_vector_chord_orig: an array containing the position of the different station at
        which the chord distribution was computed, typically a result of OpenVSP or VLM
        @param y_vector: an array containing the position of the different station at which the
        linear forces are given
        @param cl_list: an array containing the original lift coefficient distribution
        @param chord_list: an array containing the original wing chord length at the different
        station
        @return: lift_chord an array representing the linear lift at the different station of
        y_vector, integrating this vector along the wing span and multiplying it by the dynamic
        pressure will give you the actual lift distribution
        """

        # We create the interpolation function
        cl_inter = interp1d(y_vector_cl_orig, cl_list)
        chord_inter = interp1d(y_vector_chord_orig, chord_list)

        # We compute the new lift coefficient and
        cl_fin = cl_inter(y_vector)
        chord_fin = chord_inter(y_vector)
        lift_chord = np.multiply(cl_fin, chord_fin)

        return lift_chord

    @staticmethod
    def compute_relief_force(inputs, y_vector, chord_vector, wing_mass, fuel_mass, point_mass=True):
        """
        Function that computes the baseline weight distribution and modify the y_vector to
        account for point masses. We chose to represent point masses as linear masses on finite
        length and to do this we need to modify the y_vector

        @param inputs: inputs parameters defined within FAST-OAD-GA
        @param y_vector: an array containing the original position of the different station at which
        the chords are given
        @param chord_vector: an array containing the chord of the wing at different span station
        @param wing_mass: a float containing the mass of the wing
        @param fuel_mass: a float containing the mass of the fuel
        @param point_mass: a boolean, if it's FALSE all point mass will be equal to zero used in the
        post-processing
        @return: y_vector an array containing the position of the wing span at which the wing mass
        are sampled
        @return: weight_array an array containing linear masses of all structural components on the
        wing
        """

        # STEP 1/XX - DEFINE OR CALCULATE INPUT DATA FOR WEIGHT COMPUTATION

        # For post-processing we need to be able to compute the linear mass of each component
        # separately, to do this we chose to render this function able to set each mass to 0 to
        # nullify its influence
        if point_mass:
            tot_engine_mass = inputs["data:weight:propulsion:engine:mass"]
            tot_lg_mass = inputs["data:weight:airframe:landing_gear:main:mass"]
        else:
            tot_engine_mass = 0.0
            tot_lg_mass = 0.0

        engine_config = inputs["data:geometry:propulsion:engine:layout"]
        engine_count = inputs["data:geometry:propulsion:engine:count"]
        semi_span = inputs["data:geometry:wing:span"] / 2.0
        y_lg = inputs["data:geometry:landing_gear:y"]
        if engine_config != 1.0:
            y_ratio = 0.0
        else:
            y_ratio = inputs["data:geometry:propulsion:engine:y_ratio"]

        y_ratio_punctual_mass = inputs["data:weight:airframe:wing:punctual_mass:y_ratio"]
        punctual_mass_array = inputs["data:weight:airframe:wing:punctual_mass:mass"]

        g = 9.81

        # STEP 2/XX - REARRANGE THE DATA TO FIT ON ONE WING AS WE ASSUME SYMMETRICAL LOADING

        # Computing the mass of the components for one wing
        single_engine_mass = tot_engine_mass / engine_count
        single_lg_mass = tot_lg_mass / 2.0  # We assume 2 MLG

        # STEP 3/XX - AS MENTIONED BEFORE, ADDING POINT MASS WILL ADD Y STATIONS TO THE Y VECTOR
        # SO WE DO THEM BEFORE ANYTHING ELSE

        # We create the array that will store the "point mass" which we chose to represent as
        # distributed mass over a small finite interval
        point_mass_array = np.zeros(len(y_vector))

        # Adding the motor weight
        if engine_config == 1.0:
            for y_ratio_mot in y_ratio:
                y_eng = y_ratio_mot * semi_span
                y_vector, chord_vector, point_mass_array = AerostructuralLoad.add_point_mass(
                    y_vector, chord_vector, point_mass_array, y_eng, single_engine_mass, inputs
                )

        if len(y_ratio_punctual_mass) > 1 or (
            len(y_ratio_punctual_mass) == 1 and punctual_mass_array != 0
        ):
            # TODO: Can be done as a zip
            for y_ratio_punctual in y_ratio_punctual_mass:
                y_punctual_mass = y_ratio_punctual * semi_span
                punctual_mass = punctual_mass_array[
                    np.where(y_ratio_punctual_mass == y_ratio_punctual)[0]
                ]
                y_vector, chord_vector, point_mass_array = AerostructuralLoad.add_point_mass(
                    y_vector, chord_vector, point_mass_array, y_punctual_mass, punctual_mass, inputs
                )

        # Adding the LG weight
        y_vector, chord_vector, point_mass_array = AerostructuralLoad.add_point_mass(
            y_vector, chord_vector, point_mass_array, y_lg, single_lg_mass, inputs
        )

        # STEP 4/XX - WE CAN NOW ADD THE DISTRIBUTED MASS, I.E THE WING AND THE FUEL. A
        # HARD-CODED VAlUE ENABLE TO CHANGE FROM ONE MASS DISTRIBUTION TO ANOTHER

        # We first compute the shape of the distribution regardless of the amplitude which we
        # will later adjust to make sure that the integration of the mass distribution gives the
        # actual mass
        distribution_type = 0.0

        if distribution_type == 1.0:
            y_ratio = y_vector / semi_span
            struct_weight_distribution = 4.0 / np.pi * np.sqrt(1.0 - y_ratio ** 2.0)
        else:
            struct_weight_distribution = chord_vector / max(chord_vector)

        readjust_struct = trapz(struct_weight_distribution, y_vector)

        fuel_weight_distribution = tank_volume_distribution(inputs, y_vector)

        readjust_fuel = trapz(fuel_weight_distribution, y_vector)

        # We readjust to make sure that the integration of the mass distribution gives the actual
        # mass
        wing_mass_array = wing_mass * struct_weight_distribution / (2.0 * readjust_struct)
        fuel_mass_array = fuel_mass * fuel_weight_distribution / (2.0 * readjust_fuel)

        # STEP 4/XX - WE CAN NOW ADD ALL THE MASS TOGETHER AND RETURN ALL VALUES
        mass_array = wing_mass_array + fuel_mass_array + point_mass_array
        weight_array = -mass_array * g

        return y_vector, weight_array

    @staticmethod
    def insert_in_sorted_array(array, element):
        """
        Function that insert an element in a sorted array so as to keep it sorted

        @param array: a sorted array in which we want to insert an element
        @param element: the element we want to insert in the sorted array
        @return: final_array a sorted array based on the input array with the argument float
        inserted in it
        @return: index the location at which we add to insert the element ot keep the initial
        array sorted
        """

        tmp_array = np.append(array, element)
        final_array = np.sort(tmp_array)
        index = np.where(final_array == element)

        return final_array, index

    @staticmethod
    def delete_additional_zeros(array, length: int = None):
        """
        Function that delete the additional zeros we had to add to fit the format imposed by
        OpenMDAO

        @param array: an array with additional zeros we want to delete
        @param length: if len is specified leave zeros up until the length of the array is len
        @return: final_array an array containing the same elements of the initial array but with
        the additional zeros deleted
        """

        last_zero = np.amax(np.where(array != 0.0)) + 1
        if length is not None:
            final_array = array[: max(int(last_zero), length)]
        else:
            final_array = array[: int(last_zero)]

        return final_array

    @staticmethod
    def add_point_mass(y_vector, chord_vector, point_mass_array, y_point_mass, point_mass, inputs):
        """
        Function that add a point mass to an already created point_mass_array. Modify the y
        station sampling and chord sampling to account for the additional station added.

        @param y_vector: the original y_vector which will be modified by adding
        NB_POINTS_POINT_MASS + 2 points to represent the location of the new point mass
        @param chord_vector: the original chord vector which will be modified by adding
        NB_POINTS_POINT_MASS + 2 points to represent the chord at the newly added location
        @param point_mass_array: the original point mass vector on which we will add the point mass
        @param y_point_mass: the y station of the point mass
        @param point_mass: the value of the mass which we want to add
        @param inputs: inputs parameters defined within FAST-OAD-GA
        @return: y_vector_new : the new vector contains the y station at which we sample the point
        mass array with the newly added point mass
        @return: chord_vector_new : the new vector contains the chord at the new y_station
        @return: point_mass_array_new : the new vector contains the sampled point mass
        """

        # STEP 1/XX - WE EXTRACT THE NECESSARY OUTPUT FROM THE INPUTS

        semi_span = float(inputs["data:geometry:wing:span"]) / 2.0

        # STEP 2/XX - WE CREATE THE INTERPOLATION VECTOR FOR THE CHORD AND THE MASSES. WE ALSO
        # CREATE A VECTOR TO STOCK WHERE THE MASS ARE ADDED AND HAVE A WAY TO READJUST THE
        # AMPLITUDE

        fake_point_mass_array = np.zeros(len(point_mass_array))
        present_mass_interp = interp1d(y_vector, point_mass_array)
        present_chord_interp = interp1d(y_vector, chord_vector)

        # STEP 3/XX - WE ALSO STOCK WHERE WE ADD THE Y STATION SINCE IT'LL BE LATER NECESSARY TO
        # READJUST THE AMPLITUDE

        interval_len = POINT_MASS_SPAN_RATIO * semi_span / NB_POINTS_POINT_MASS
        nb_point_side = (NB_POINTS_POINT_MASS - 1.0) / 2.0
        y_added = []

        # STEP 4/XX - WE ADD THE NB_POINTS_POINT_MASS AND 2 MORE POINT JUST BEFORE AND AFTER TO
        # GET THE PROPER SQUARE SHAPE AND NOT A TRAPEZE. WE ALSO ADD THE CORRESPONDING POINT IN
        # THE CHORD VECTOR

        for i in range(NB_POINTS_POINT_MASS):
            y_current = y_point_mass + (i - nb_point_side) * interval_len
            if (y_current >= 0.0) and (y_current <= semi_span):
                y_added.append(y_current)
                y_vector, idx = AerostructuralLoad.insert_in_sorted_array(y_vector, y_current)
                index = int(float(idx[0]))
                chord_vector = np.insert(chord_vector, index, present_chord_interp(y_current))
                point_mass_array = np.insert(
                    point_mass_array, index, present_mass_interp(y_current)
                )
                fake_point_mass_array = np.insert(fake_point_mass_array, index, 0.0)

        y_min = min(y_added) - 1e-3
        y_vector, idx = AerostructuralLoad.insert_in_sorted_array(y_vector, y_min)
        index = int(float(idx[0]))
        chord_vector = np.insert(chord_vector, index, present_chord_interp(y_min))
        point_mass_array = np.insert(point_mass_array, index, present_mass_interp(y_min))
        fake_point_mass_array = np.insert(fake_point_mass_array, index, 0.0)

        y_max = max(y_added) + 1e-3
        y_vector, idx = AerostructuralLoad.insert_in_sorted_array(y_vector, y_max)
        index = int(float(idx[0]))
        chord_vector = np.insert(chord_vector, index, present_chord_interp(y_max))
        point_mass_array = np.insert(point_mass_array, index, present_mass_interp(y_max))
        fake_point_mass_array = np.insert(fake_point_mass_array, index, 0.0)

        # STEP 5/XX - WE NOW HAVE THE RIGHT WE JUST NEED TO SCALE IT PROPERLY WHICH IS THE POINT
        # OF THIS STEP

        where_add_mass_grt = np.greater_equal(y_vector, min(y_added))
        where_add_mass_lss = np.less_equal(y_vector, max(y_added))
        where_add_mass = np.logical_and(where_add_mass_grt, where_add_mass_lss)
        where_add_mass_index = np.where(where_add_mass)

        # The fake mass array we use to readjust contains the mass at the station between the
        # first one we added and the last one which might contains the y station added for other
        # point mass if two are on top of each other

        for idx in where_add_mass_index:
            fake_point_mass_array[idx] = 1.0

        readjust = trapz(fake_point_mass_array, y_vector)

        for idx in where_add_mass_index:
            point_mass_array[idx] += point_mass / readjust

        y_vector_new = y_vector
        point_mass_array_new = point_mass_array
        chord_vector_new = chord_vector

        return y_vector_new, chord_vector_new, point_mass_array_new
