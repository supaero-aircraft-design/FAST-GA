"""
    FAST - Copyright (c) 2016 ONERA ISAE
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
import scipy.interpolate as inter

from openmdao.core.explicitcomponent import ExplicitComponent
from fastoad.model_base import Atmosphere


class Compute2DHingeMomentsTail(ExplicitComponent):
    """
    Based on : Roskam, Jan. Airplane Design: Part 6-Preliminary Calculation of Aerodynamic, Thrust and Power
    Characteristics. DAR corporation, 1985. Section 10.4.1, 10.4.2
    """

    def setup(self):
        self.add_input("data:geometry:horizontal_tail:elevator_chord_ratio", val=np.nan)
        self.add_input("data:geometry:horizontal_tail:thickness_ratio", val=np.nan)
        self.add_input("data:geometry:horizontal_tail:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:aerodynamics:horizontal_tail:cruise:CL_alpha", val=np.nan, units="rad**-1")
        self.add_input("data:TLAR:v_cruise", val=np.nan, units="m/s")
        self.add_input("data:mission:sizing:main_route:cruise:altitude", val=np.nan, units="m")

        self.add_output("data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_alpha_2D",
                        units="rad**-1")
        self.add_output("data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_delta_2D",
                        units="rad**-1")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        elevator_chord_ratio = inputs["data:geometry:horizontal_tail:elevator_chord_ratio"]
        tail_thickness_ratio = inputs["data:geometry:horizontal_tail:thickness_ratio"]
        cl_alpha_ht = inputs["data:aerodynamics:horizontal_tail:cruise:CL_alpha"]
        wing_area = inputs["data:geometry:wing:area"]
        ht_area = inputs["data:geometry:horizontal_tail:area"]
        v_cruise = inputs["data:TLAR:v_cruise"]
        cruise_alt = inputs["data:mission:sizing:main_route:cruise:altitude"]

        # Section 10.4.1.1
        # Step 1.
        y = lambda x: tail_thickness_ratio / 0.2 * (
                0.2969 * x ** 0.5 - 0.126 * x - 0.3516 * x ** 2.0 + 0.2843 * x ** 3.0 - 0.105 * x ** 4.0)

        y_90 = y(0.90)
        y_95 = y(0.95)
        y_99 = y(0.99)

        tan_0_5_phi_te = (y(1. - 1.e-6) - y(1.)) / 1e-6
        tan_0_5_phi_te_prime = (y_90 / 2.0 - y_99 / 2.0) / 9.0
        tan_0_5_phi_te_prime_prime = (y_95 / 2.0 - y_99 / 2.0) / 9.0

        if (tan_0_5_phi_te == tan_0_5_phi_te_prime) and (tan_0_5_phi_te_prime == tan_0_5_phi_te_prime_prime) and (
                tan_0_5_phi_te_prime_prime == tail_thickness_ratio):
            condition = True
        else:
            condition = False

        # Step 2.
        cl_alpha_ht *= wing_area / ht_area
        cl_alpha_ht_th = 6.3 + tail_thickness_ratio / 0.2 * (7.3 - 6.3)

        k_cl_alpha_inter = inter.interp1d([0., 0.2], [0.9, 0.69])
        k_cl_alpha = k_cl_alpha_inter(tail_thickness_ratio)[0]

        k_cl_alpha_array = [0.7, 0.72, 0.74, 0.76, 0.78, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0]
        k_ch_alpha_min_array = [-0.1, 0.0, 0.08, 0.19, 0.28, 0.36, 0.43, 0.5, 0.58, 0.65, 0.70, 0.75, 0.82, 0.87, 0.93,
                                1.0]
        k_ch_alpha_max_array = [0.12, 0.22, 0.30, 0.38, 0.46, 0.53, 0.6, 0.65, 0.70, 0.75, 0.80, 0.84, 0.88, 0.92, 0.96,
                                1.0]
        k_ch_alpha_min_inter = inter.interp1d(k_cl_alpha_array, k_ch_alpha_min_array)
        k_ch_alpha_max_inter = inter.interp1d(k_cl_alpha_array, k_ch_alpha_max_array)

        k_ch_alpha_min = k_ch_alpha_min_inter(k_cl_alpha)
        k_ch_alpha_max = k_ch_alpha_max_inter(k_cl_alpha)

        if elevator_chord_ratio < 0.1:
            k_ch_alpha = k_ch_alpha_min
        elif elevator_chord_ratio > 0.4:
            k_ch_alpha = k_ch_alpha_max
        else:
            k_ch_alpha_inter = inter.interp1d([0.1, 0.4], [k_ch_alpha_min, k_ch_alpha_max])
            k_ch_alpha = k_ch_alpha_inter(elevator_chord_ratio)

        thickness_ratio_array = [0., 0.4, 0.6, 0.8, 0.10, 0.12, 0.16]
        ch_alpha_min_array = [-0.225, -0.25, -0.27, -0.295, -0.305, -0.315, -0.3]
        ch_alpha_max_array = [-0.65, -0.67, -0.685, -0.7, -0.71, -0.72, -0.75]

        ch_alpha_min_inter = inter.interp1d(thickness_ratio_array, ch_alpha_min_array)
        ch_alpha_max_inter = inter.interp1d(thickness_ratio_array, ch_alpha_max_array)

        ch_alpha_min = ch_alpha_min_inter(tail_thickness_ratio)[0]
        ch_alpha_max = ch_alpha_max_inter(tail_thickness_ratio)[0]

        if elevator_chord_ratio < 0.1:
            ch_alpha = ch_alpha_min
        elif elevator_chord_ratio > 0.4:
            ch_alpha = ch_alpha_max
        else:
            ch_alpha_inter = inter.interp1d([0.1, 0.4], [ch_alpha_min, ch_alpha_max])
            ch_alpha = ch_alpha_inter(elevator_chord_ratio)

        ch_prime_alpha = k_ch_alpha * ch_alpha

        # Step 3.

        if condition:
            ch_prime_prime_alpha = ch_prime_alpha

        else:
            ch_prime_prime_alpha = ch_prime_alpha + \
                                   2. * cl_alpha_ht_th * (1. - k_cl_alpha) * \
                                   (tan_0_5_phi_te_prime_prime - tail_thickness_ratio)

        # Step 4.
        # The overhang (cb/cf will we taken equal to 3 as we assume a 1/4, 3/4 chord repartition for the hinge line)
        # We will also assume that the thickness ratio of the elevator is the same as the tail and that it has a
        # round nose

        balance_ratio = math.sqrt((1. / 3.) ** 2.0 - (tail_thickness_ratio * 5. / 4) ** 2.0)

        if balance_ratio < 0.15:
            balance_ratio = 0.15
        elif balance_ratio > 0.5:
            balance_ratio = 0.5

        k_ch_alpha_balance_inter = inter.interp1d([0.15, 0.50], [0.93, 0.2])
        k_ch_alpha_balance = k_ch_alpha_balance_inter(balance_ratio)

        ch_alpha_balance = k_ch_alpha_balance * ch_prime_prime_alpha

        # Step 5.

        sos_cruise = Atmosphere(cruise_alt, altitude_in_feet=False).speed_of_sound
        mach = v_cruise / sos_cruise
        beta = math.sqrt(1. - mach ** 2.0)

        ch_alpha_fin = ch_alpha_balance / beta

        # Section 10.4.1.2
        # Step 1., same as step 1. in previous section

        # Step 2.

        k_cl_alpha_array = [0.6, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]
        k_ch_delta_min = [0.64, 0.70, 0.75, 0.80, 0.84, 0.88, 0.92, 0.96, 1.0]
        k_ch_delta_avg = [0.56, 0.65, 0.725, 0.775, 0.83, 0.87, 0.91, 0.955, 1.0]
        k_ch_delta_max = [0.415, 0.55, 0.66, 0.745, 0.80, 0.85, 0.90, 0.95, 1.0]

        k_ch_delta_min_inter = inter.interp1d(k_cl_alpha_array, k_ch_delta_min)
        k_ch_delta_avg_inter = inter.interp1d(k_cl_alpha_array, k_ch_delta_avg)
        k_ch_delta_max_inter = inter.interp1d(k_cl_alpha_array, k_ch_delta_max)

        k_ch_delta_min = k_ch_delta_min_inter(k_cl_alpha)
        k_ch_delta_avg = k_ch_delta_avg_inter(k_cl_alpha)
        k_ch_delta_max = k_ch_delta_max_inter(k_cl_alpha)

        if elevator_chord_ratio < 0.1:
            k_ch_delta = k_ch_delta_min
        elif elevator_chord_ratio > 0.4:
            k_ch_delta = k_ch_delta_max
        else:
            k_ch_delta_inter = inter.interp1d([0.1, 0.25, 0.4], [k_ch_delta_min, k_ch_delta_avg, k_ch_delta_max],
                                              kind='quadratic')
            k_ch_delta = k_ch_delta_inter(elevator_chord_ratio)

        thickness_ratio_array = [0., 0.4, 0.6, 0.8, 0.10, 0.12, 0.16]
        ch_delta_min_array = [-0.88, -0.83, -0.8, -0.77, -0.735, -0.7, -0.64]
        ch_delta_max_array = [-1.02, -0.995, -0.985, -0.97, -0.955, -0.94, -0.925]

        ch_delta_min_inter = inter.interp1d(thickness_ratio_array, ch_delta_min_array)
        ch_delta_max_inter = inter.interp1d(thickness_ratio_array, ch_delta_max_array)

        ch_delta_min = ch_delta_min_inter(tail_thickness_ratio)[0]
        ch_delta_max = ch_delta_max_inter(tail_thickness_ratio)[0]

        if elevator_chord_ratio < 0.1:
            ch_delta = ch_delta_min
        elif elevator_chord_ratio > 0.4:
            ch_delta = ch_delta_max
        else:
            ch_delta_inter = inter.interp1d([0.1, 0.4], [ch_delta_min, ch_delta_max])
            ch_delta = ch_delta_inter(elevator_chord_ratio)

        ch_prime_delta = k_ch_delta * ch_delta

        # Step 3.

        if condition:
            ch_prime_prime_delta = ch_prime_delta

        else:
            thickness_ratio_array = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15]
            cl_delta_th_min_array = [1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75]
            cl_delta_th_avg_array = [4.125, 4.1875, 4.25, 4.3125, 4.375, 4.4375, 4.5, 4.5625]
            cl_delta_th_max_array = [5.15, 5.25, 5.35, 5.45, 5.55, 5.7, 5.8, 5.95]

            cl_delta_th_min_inter = inter.interp1d(thickness_ratio_array, cl_delta_th_min_array)
            cl_delta_th_avg_inter = inter.interp1d(thickness_ratio_array, cl_delta_th_avg_array)
            cl_delta_th_max_inter = inter.interp1d(thickness_ratio_array, cl_delta_th_max_array)

            cl_delta_th_min = cl_delta_th_min_inter(tail_thickness_ratio)[0]
            cl_delta_th_avg = cl_delta_th_avg_inter(tail_thickness_ratio)[0]
            cl_delta_th_max = cl_delta_th_max_inter(tail_thickness_ratio)[0]

            if elevator_chord_ratio < 0.05:
                cl_delta_th = cl_delta_th_min
            elif elevator_chord_ratio > 0.5:
                cl_delta_th = cl_delta_th_max
            else:
                cl_delta_th_inter = inter.interp1d([0.05, 0.3, 0.5],
                                                   [cl_delta_th_min, cl_delta_th_avg, cl_delta_th_max],
                                                   kind='quadratic')
                cl_delta_th = cl_delta_th_inter(elevator_chord_ratio)

            k_cl_alpha_array = [0.7, 0.72, 0.74, 0.76, 0.78, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98,
                                1.0]
            k_cl_delta_min_array = [0.36, 0.4, 0.44, 0.48, 0.53, 0.575, 0.615, 0.655, 0.7, 0.735, 0.78, 0.83, 0.86,
                                    0.91, 0.95, 1.0]
            k_cl_delta_max_array = [0.55, 0.59, 0.62, 0.655, 0.695, 0.733, 0.766, 0.8, 0.82, 0.845, 0.87, 0.9, 0.925,
                                    0.95, 0.975, 1.0]

            k_cl_delta_min_array_inter = inter.interp1d(k_cl_alpha_array, k_cl_delta_min_array)
            k_cl_delta_max_array_inter = inter.interp1d(k_cl_alpha_array, k_cl_delta_max_array)

            k_cl_delta_min = k_cl_delta_min_array_inter(k_cl_alpha)
            k_cl_delta_max = k_cl_delta_max_array_inter(k_cl_alpha)

            if elevator_chord_ratio < 0.05:
                k_cl_delta = k_cl_delta_min
            elif elevator_chord_ratio > 0.5:
                k_cl_delta = k_cl_delta_max
            else:
                k_cl_delta_inter = inter.interp1d([0.05, 0.5], [k_cl_delta_min, k_cl_delta_max])
                k_cl_delta = k_cl_delta_inter(elevator_chord_ratio)

            ch_prime_prime_delta = ch_prime_delta + (
                    2. * cl_delta_th * (1. - k_cl_delta) *
                    (tan_0_5_phi_te_prime_prime - tail_thickness_ratio))

        # Step 4. Same assumption as the step 4. of the previous section. Only this time we only take the curve for
        # the NACA 0009
        k_ch_delta_balance = inter.interp1d([0.15, 0.50], [0.93, 0.2])(balance_ratio)

        ch_delta_balance = k_ch_delta_balance * ch_prime_prime_delta

        # Step 5.
        ch_delta_fin = ch_delta_balance / beta

        outputs["data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_alpha_2D"] = ch_alpha_fin
        outputs["data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_delta_2D"] = ch_delta_fin


class Compute3DHingeMomentsTail(ExplicitComponent):
    """
    Based on : Roskam, Jan. Airplane Design: Part 6-Preliminary Calculation of Aerodynamic, Thrust and Power
    Characteristics. DAR corporation, 1985. Section 10.4.1
    """

    def setup(self):
        self.add_input("data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_alpha_2D", val=np.nan, units="rad**-1")
        self.add_input("data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_delta_2D", val=np.nan, units="rad**-1")
        self.add_input("data:geometry:horizontal_tail:sweep_25", val=np.nan, units="deg")
        self.add_input("data:geometry:horizontal_tail:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:horizontal_tail:elevator_chord_ratio", val=np.nan)

        self.add_output("data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_alpha",
                        units="rad**-1")
        self.add_output("data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_delta",
                        units="rad**-1")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        ch_alpha_2d = inputs["data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_alpha_2D"]
        ch_delta_2d = inputs["data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_delta_2D"]
        sweep_25_ht = inputs["data:geometry:horizontal_tail:sweep_25"]*math.pi/180.
        ar_ht = inputs["data:geometry:horizontal_tail:aspect_ratio"]
        elevator_chord_ratio = inputs["data:geometry:horizontal_tail:elevator_chord_ratio"]

        # Section 10.4.2.1
        # Step 1. : delta_Ch_alpha we will ignore it for now, same for delta_Ch_delta

        ch_alpha_3d = ((ar_ht * math.cos(sweep_25_ht))/(ar_ht + 2. * math.cos(sweep_25_ht))) * ch_alpha_2d

        # Section 10.4.2.2
        # We assume the same sweep angle for hl and tail
        sweep_hl = sweep_25_ht

        # We'll compute the elevator effectiveness factor in the worst case scenario, i.e, with the highest deflection
        # angle which we will take at 25 degree
        chord_ratio_array = [0.15, 0.20, 0.25, 0.3, 0.4]
        a_delta_array = [0.35, 0.415, 0.48, 0.52, 0.57]

        a_delta_inter = inter.interp1d(chord_ratio_array, a_delta_array)

        a_delta = a_delta_inter(elevator_chord_ratio)

        ch_delta_3d = math.cos(sweep_25_ht) * math.cos(sweep_hl) * (
            ch_delta_2d +
            a_delta * ((2.0 * math.cos(sweep_25_ht))/(ar_ht + 2. * math.cos(sweep_25_ht))) * ch_alpha_2d
        )

        outputs["data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_alpha"] = ch_alpha_3d
        outputs["data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_delta"] = ch_delta_3d
