"""
Computes the aerostructural loads on the wing of the aircraft
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

import math

import numpy as np

from ..aerodynamics.constants import SPAN_MESH_POINT, MACH_NB_PTS
from ..aerodynamics.external.openvsp.compute_vn import ComputeVNopenvsp
from fastga.utils.physics.atmosphere import Atmosphere
from scipy.integrate import trapz
from scipy.interpolate import interp1d
from ..aerodynamics.lift_equilibrium import AircraftEquilibrium

NB_POINTS_POINT_MASS = 5
# MUST BE AN EVEN NUMBER
POINT_MASS_SPAN_RATIO = 0.01
SPAN_MESH_POINT_LOADS = int(2.0 * SPAN_MESH_POINT)


class AerostructuralLoadX57(ComputeVNopenvsp):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def initialize(self):
        super().initialize()

    def setup(self):

        self.add_input("data:TLAR:category", val=3.0)
        self.add_input("data:TLAR:level", val=1.0)
        self.add_input("data:TLAR:v_max_sl", val=np.nan, units="kn")
        self.add_input("data:TLAR:v_cruise", val=np.nan, units="m/s")
        self.add_input("data:TLAR:v_approach", val=np.nan, units="m/s")

        nans_array_ov = np.full(SPAN_MESH_POINT, np.nan)
        nans_array_m = np.full(MACH_NB_PTS + 1, np.nan)
        self.add_input("data:aerodynamics:wing:low_speed:Y_vector", val=nans_array_ov, shape=SPAN_MESH_POINT,
                       units="m")
        self.add_input("data:aerodynamics:wing:low_speed:chord_vector", val=nans_array_ov, shape=SPAN_MESH_POINT,
                       units="m")
        self.add_input("data:aerodynamics:wing:low_speed:CL_vector", val=nans_array_ov, shape=SPAN_MESH_POINT)
        self.add_input("data:aerodynamics:slipstream:wing:only_prop:CL_vector", val=nans_array_ov,
                       shape=SPAN_MESH_POINT)
        self.add_input("data:aerodynamics:slipstream:wing:prop_on:Y_vector", val=nans_array_ov,
                       shape=SPAN_MESH_POINT, units="m")
        self.add_input("data:aerodynamics:slipstream:wing:prop_on:velocity", val=np.nan, units="m/s")
        self.add_input("data:aerodynamics:wing:low_speed:CL0_clean", val=np.nan)
        self.add_input("data:aerodynamics:wing:cruise:CL0_clean", val=np.nan)
        self.add_input("data:aerodynamics:wing:cruise:CM0_clean", val=np.nan)
        self.add_input("data:aerodynamics:horizontal_tail:efficiency", val=np.nan)
        self.add_input("data:aerodynamics:aircraft:landing:CL_max", val=np.nan)
        self.add_input("data:aerodynamics:wing:low_speed:CL_max_clean", val=np.nan)
        self.add_input("data:aerodynamics:wing:low_speed:CL_min_clean", val=np.nan)
        self.add_input("data:aerodynamics:wing:low_speed:CL_alpha", val=np.nan, units="rad**-1")
        self.add_input("data:aerodynamics:wing:cruise:CL_alpha", val=np.nan, units="rad**-1")
        self.add_input("data:aerodynamics:horizontal_tail:low_speed:CL_alpha", val=np.nan, units="rad**-1")
        self.add_input("data:aerodynamics:horizontal_tail:cruise:CL_alpha", val=np.nan, units="rad**-1")
        self.add_input("data:aerodynamics:aircraft:mach_interpolation:CL_alpha_vector", val=nans_array_m,
                       units="rad**-1",
                       shape=MACH_NB_PTS + 1)
        self.add_input("data:aerodynamics:aircraft:mach_interpolation:mach_vector", val=nans_array_m,
                       shape=MACH_NB_PTS + 1)

        self.add_input("data:weight:aircraft:MZFW", val=np.nan, units="kg")
        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="kg")
        self.add_input("data:weight:propulsion:engine:mass", val=np.nan, units="kg")
        self.add_input("data:weight:airframe:landing_gear:main:mass", val=np.nan, units="kg")
        self.add_input("data:weight:airframe:wing:mass", val=np.nan, units="kg")
        self.add_input("data:weight:aircraft_empty:CG:z", val=np.nan, units="m")
        self.add_input("data:weight:aircraft:CG:aft:x", val=np.nan, units="m")
        self.add_input("data:weight:aircraft:CG:fwd:x", val=np.nan, units="m")

        self.add_input("data:geometry:wing:root:virtual_chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:MAC:leading_edge:x:local", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:landing_gear:height", val=np.nan, units="m")
        self.add_input("data:geometry:landing_gear:type", val=np.nan)
        self.add_input("data:geometry:propulsion:layout", val=np.nan)
        self.add_input("data:geometry:propulsion:count", val=np.nan)
        self.add_input("data:geometry:propulsion:y_ratio", val=np.nan)
        self.add_input("data:geometry:propulsion:nacelle:width", val=np.nan, units="m")

        self.add_input("data:mission:sizing:fuel", val=np.nan, units="kg")
        self.add_input("data:mission:sizing:main_route:cruise:altitude", val=np.nan, units="ft")

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

        y_vector = inputs["data:aerodynamics:wing:low_speed:Y_vector"]
        y_vector_slip = inputs["data:aerodynamics:slipstream:wing:prop_on:Y_vector"]
        cl_vector = inputs["data:aerodynamics:wing:low_speed:CL_vector"]
        cl_vector_slip = inputs["data:aerodynamics:slipstream:wing:only_prop:CL_vector"]
        cl_0 = inputs["data:aerodynamics:wing:low_speed:CL0_clean"]
        chord_vector = inputs["data:aerodynamics:wing:low_speed:chord_vector"]
        v_ref = inputs["data:aerodynamics:slipstream:wing:prop_on:velocity"]

        semi_span = inputs["data:geometry:wing:span"] / 2.0
        wing_area = inputs["data:geometry:wing:area"]

        mtow = inputs["data:weight:aircraft:MTOW"]
        mzfw = inputs["data:weight:aircraft:MZFW"]
        wing_mass = inputs["data:weight:airframe:wing:mass"]

        cruise_alt = inputs["data:mission:sizing:main_route:cruise:altitude"]

        cruise_v_tas = inputs["data:TLAR:v_cruise"]

        # We delete the zeros we had to add to fit the size we set in the aerodynamics module and add the physic
        # extrema that are missing, the root and the full span,
        y_vector = AerostructuralLoadX57.delete_additional_zeros(y_vector)
        y_vector_slip = AerostructuralLoadX57.delete_additional_zeros(y_vector_slip)
        cl_vector = AerostructuralLoadX57.delete_additional_zeros(cl_vector)
        cl_vector_slip = AerostructuralLoadX57.delete_additional_zeros(cl_vector_slip)
        chord_vector = AerostructuralLoadX57.delete_additional_zeros(chord_vector)
        y_vector, _ = AerostructuralLoadX57.insert_in_sorted_array(y_vector, 0.)
        y_vector_slip, _ = AerostructuralLoadX57.insert_in_sorted_array(y_vector_slip, 0.)
        cl_vector = np.insert(cl_vector, 0, cl_vector[0])
        cl_vector_slip = np.insert(cl_vector_slip, 0, cl_vector_slip[0])
        chord_vector = np.insert(chord_vector, 0, chord_vector[0])
        y_vector_orig, _ = AerostructuralLoadX57.insert_in_sorted_array(y_vector, semi_span)
        y_vector_slip_orig, _ = AerostructuralLoadX57.insert_in_sorted_array(y_vector_slip, semi_span)
        cl_vector = np.append(cl_vector, 0.)
        cl_vector_slip = np.append(cl_vector_slip, 0.)
        chord_vector = np.append(chord_vector, chord_vector[-1])

        FoS = 1.5
        shear_max_conditions = []
        rbm_max_conditions = []

        y_vector, _ = self.compute_relief_force_x57(inputs, y_vector_orig, chord_vector,
                                                    wing_mass, 0.0)
        cl_s = self.compute_Cl_S(y_vector_orig, y_vector, cl_vector, chord_vector)
        cl_s_slip = self.compute_Cl_S(y_vector_slip_orig, y_vector, cl_vector_slip, chord_vector)

        mass_array = np.array([mtow, min(mzfw, mtow)])
        # In case fully electric
        # TODO Account for tanks in the fuselage of the wing

        atm = Atmosphere(cruise_alt)

        shear_max = 0.0
        rbm_max = 0.0

        for mass in mass_array:

            if abs(mtow - mzfw) < 5.:
                fuel_mass = 0.0
            else:
                fuel_mass = mass - mzfw

            y_vector, weight_array_orig = self.compute_relief_force_x57(inputs, y_vector_orig, chord_vector,
                                                                        wing_mass, fuel_mass)

            cruise_v_keas = atm.get_equivalent_airspeed(cruise_v_tas)

            velocity_array, load_factor_array, _ = self.flight_domain(inputs, outputs, mass, cruise_alt, cruise_v_keas)

            v_c = float(velocity_array[6])

            load_factor_list = np.array([max(load_factor_array), min(load_factor_array)])

            v_c_tas = atm.get_true_airspeed(v_c)
            dynamic_pressure = 1. / 2. * atm.density * v_c_tas ** 2.0

            for load_factor in load_factor_list:

                cl_wing = 1.05 * (mass * load_factor * 9.81) / (dynamic_pressure * wing_area)
                cl_s_actual = cl_s * cl_wing / cl_0
                cl_s_slip_actual = cl_s_slip * (v_ref / v_c_tas) ** 2.0
                lift_section = FoS * dynamic_pressure * (cl_s_actual + cl_s_slip_actual)
                weight_array = weight_array_orig * FoS * load_factor

                tot_shear_diagram = AerostructuralLoadX57.compute_shear_diagram(y_vector, weight_array + lift_section)
                tot_bending_moment_diagram = AerostructuralLoadX57.compute_bending_moment_diagram(
                    y_vector, weight_array + lift_section)
                root_shear_force = tot_shear_diagram[0]
                root_bending_moment = tot_bending_moment_diagram[0]

                if abs(root_shear_force) > shear_max:
                    shear_max_conditions = [mass, load_factor]
                    lift_shear_diagram = AerostructuralLoadX57.compute_shear_diagram(y_vector, lift_section)
                    weight_shear_diagram = AerostructuralLoadX57.compute_shear_diagram(y_vector, weight_array)
                    shear_max = abs(root_shear_force)

                if abs(root_bending_moment) > rbm_max:
                    rbm_max_conditions = [mass, load_factor]
                    lift_bending_diagram = AerostructuralLoadX57.compute_bending_moment_diagram(y_vector, lift_section)
                    weight_bending_diagram = AerostructuralLoadX57.compute_bending_moment_diagram(
                        y_vector, weight_array)
                    rbm_max = abs(root_bending_moment)

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

        shear_force_diagram = np.zeros(len(y_vector))

        for i in range(len(y_vector)):
            shear_force_diagram[i] = trapz(force_array[i:], y_vector[i:])

        return shear_force_diagram

    @staticmethod
    def compute_bending_moment_diagram(y_vector, force_array):

        bending_moment_diagram = np.zeros(len(y_vector))
        for i in range(len(y_vector)):
            lever_arm = y_vector - y_vector[i]
            test = lever_arm[i:]
            bending_moment_diagram[i] = trapz(force_array[i:] * lever_arm[i:], y_vector[i:])

        return bending_moment_diagram

    @staticmethod
    def compute_Cl_S(y_vector_orig, y_vector, cl_list, chord_list):

        cl_inter = interp1d(y_vector_orig, cl_list)
        chord_inter = interp1d(y_vector_orig, chord_list)
        cl_fin = cl_inter(y_vector)
        chord_fin = chord_inter(y_vector)
        lift_chord = np.multiply(cl_fin, chord_fin)

        return lift_chord

    @staticmethod
    def compute_relief_force_x57(inputs, y_vector, chord_vector, wing_mass, fuel_mass, point_mass=True):

        # Recuperating the data necessary for the computation
        if point_mass:
            eng_mass_vec = [6.8, 6.8, 6.8, 6.8, 6.8, 6.8, 53.1]
            tot_lg_mass = inputs["data:weight:airframe:landing_gear:main:mass"]
        else:
            eng_mass_vec = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            tot_lg_mass = 0.0

        z_cg = inputs["data:weight:aircraft_empty:CG:z"]

        lg_height = inputs["data:geometry:landing_gear:height"]
        lg_type = inputs["data:geometry:landing_gear:type"]
        engine_config = inputs["data:geometry:propulsion:layout"]
        engine_count = inputs["data:geometry:propulsion:count"]
        nacelle_width = inputs["data:geometry:propulsion:nacelle:width"]
        semi_span = inputs["data:geometry:wing:span"] / 2.0
        y_ratio_vec = [0.1863354, 0.30310559, 0.42236025, 0.54161491, 0.66086957, 0.79710145, 0.99171843]

        g = 9.81

        single_lg_mass = tot_lg_mass / 2.0  # We assume 2 MLG

        # Before computing the continued weight distribution we first take care of the point masses and modify the
        # y_vector accordingly

        # We create the array that will store the "point mass" which we chose to represent as distributed mass over a
        # small finite interval
        point_mass_array = np.zeros(len(y_vector))

        # Adding the motor weight
        if engine_config == 1.0:
            for i in range(len(y_ratio_vec)):
                y_ratio = y_ratio_vec[i]
                eng_mass = eng_mass_vec[i]
                y_eng = y_ratio * semi_span
                y_vector, chord_vector, point_mass_array = AerostructuralLoadX57.add_point_mass(
                    y_vector, chord_vector, point_mass_array, y_eng, eng_mass, inputs)
                test = 1.0

        y_eng_array = y_ratio_vec * semi_span

        # Computing and adding the lg weight
        # Overturn angle set as a fixed value, it is recommended to take over 25° and check that we can fit both LG in
        # the fuselage
        phi_ot = 35. * np.pi / 180.
        y_lg_1 = math.tan(phi_ot) * z_cg
        y_lg = max(y_lg_1, lg_height)

        y_vector, chord_vector, point_mass_array = AerostructuralLoadX57.add_point_mass(
            y_vector, chord_vector, point_mass_array, y_lg, single_lg_mass, inputs)

        # We can now choose what type of mass distribution we want for the mass and the fuel
        distribution_type = 0.0
        if distribution_type == 1.0:
            Y = y_vector / semi_span
            struct_weight_distribution = 4. / np.pi * np.sqrt(1. - Y ** 2.0)
        else:
            Y = y_vector / semi_span
            struct_weight_distribution = chord_vector / max(chord_vector)

        reajust_struct = trapz(struct_weight_distribution, y_vector)

        in_eng_nacelle = np.full(len(y_vector), False)
        for y_eng in y_eng_array:
            for i in np.where(abs(y_vector - y_eng) <= nacelle_width / 2.):
                in_eng_nacelle[i] = True
        where_engine = np.where(in_eng_nacelle)

        if distribution_type == 1.0:
            Y = y_vector / semi_span
            fuel_weight_distribution = 4. / np.pi * np.sqrt(1. - Y ** 2.0)
        else:
            Y = y_vector / semi_span
            fuel_weight_distribution = chord_vector / max(chord_vector)
            if lg_type == 1.0:
                for i in np.where(y_vector < y_lg):
                    # For now 80% size reduction in the fuel tank capacity due to the landing gear
                    fuel_weight_distribution[i] = fuel_weight_distribution[i] * 0.2
            if engine_config == 1.0:
                for i in where_engine:
                    # For now 50% size reduction in the fuel tank capacity due to the engine
                    fuel_weight_distribution[i] = fuel_weight_distribution[i] * 0.5

        reajust_fuel = trapz(fuel_weight_distribution, y_vector)

        wing_mass_array = wing_mass * struct_weight_distribution / (2. * reajust_struct)
        fuel_mass_array = fuel_mass * fuel_weight_distribution / (2. * reajust_fuel)

        mass_array = wing_mass_array + fuel_mass_array + point_mass_array
        weight_array = - mass_array * g

        return y_vector, weight_array

    @staticmethod
    def insert_in_sorted_array(array, element):

        tmp_array = np.append(array, element)
        final_array = np.sort(tmp_array)
        index = np.where(final_array == element)

        return final_array, index

    @staticmethod
    def delete_additional_zeros(array):

        last_zero = np.amax(np.where(array != 0.)) + 1
        final_array = array[:int(last_zero)]

        return final_array

    @staticmethod
    def add_point_mass(y_vector, chord_vector, point_mass_array, y_point_mass, point_mass, inputs):

        semi_span = float(inputs["data:geometry:wing:span"]) / 2.0
        fake_point_mass_array = np.zeros(len(point_mass_array))
        present_mass_interp = interp1d(y_vector, point_mass_array)
        present_chord_interp = interp1d(y_vector, chord_vector)

        interval_len = POINT_MASS_SPAN_RATIO * semi_span / NB_POINTS_POINT_MASS
        nb_point_side = (NB_POINTS_POINT_MASS - 1.) / 2.
        y_added = []

        for i in range(NB_POINTS_POINT_MASS):
            y_current = y_point_mass + (i - nb_point_side) * interval_len
            if (y_current >= 0.0) and (y_current <= semi_span):
                y_added.append(y_current)
                y_vector, idx = AerostructuralLoadX57.insert_in_sorted_array(y_vector, y_current)
                index = int(float(idx[0]))
                chord_vector = np.insert(chord_vector, index, present_chord_interp(y_current))
                point_mass_array = np.insert(point_mass_array, index, present_mass_interp(y_current))
                fake_point_mass_array = np.insert(fake_point_mass_array, index, 0.)

        y_min = min(y_added) - 1e-3
        y_vector, idx = AerostructuralLoadX57.insert_in_sorted_array(y_vector, y_min)
        index = int(float(idx[0]))
        chord_vector = np.insert(chord_vector, index, present_chord_interp(y_min))
        point_mass_array = np.insert(point_mass_array, index, present_mass_interp(y_min))
        fake_point_mass_array = np.insert(fake_point_mass_array, index, 0.)

        y_max = max(y_added) + 1e-3
        y_vector, idx = AerostructuralLoadX57.insert_in_sorted_array(y_vector, y_max)
        index = int(float(idx[0]))
        chord_vector = np.insert(chord_vector, index, present_chord_interp(y_max))
        point_mass_array = np.insert(point_mass_array, index, present_mass_interp(y_max))
        fake_point_mass_array = np.insert(fake_point_mass_array, index, 0.)

        where_add_mass_grt = np.greater_equal(y_vector, min(y_added))
        where_add_mass_lss = np.less_equal(y_vector, max(y_added))
        where_add_mass = np.logical_and(where_add_mass_grt, where_add_mass_lss)
        where_add_mass_index = np.where(where_add_mass)

        for idx in where_add_mass_index:
            fake_point_mass_array[idx] = 1.0

        reajust = trapz(fake_point_mass_array, y_vector)

        for idx in where_add_mass_index:
            point_mass_array[idx] += point_mass / reajust

        y_vector_new = y_vector
        point_mass_array_new = point_mass_array
        chord_vector_new = chord_vector

        return y_vector_new, chord_vector_new, point_mass_array_new
