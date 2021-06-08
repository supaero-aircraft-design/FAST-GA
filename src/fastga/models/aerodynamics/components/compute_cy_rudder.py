"""
    Estimation of yawing moment du to the ruddder
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
from scipy import interpolate
from openmdao.core.explicitcomponent import ExplicitComponent
from .high_lift_aero import ComputeDeltaHighLift, KB_FLAPS
from . import resources


class ComputeCyDeltaRudder(ExplicitComponent):
    """
    Yawing moment due to rudder estimation

    Based on : Roskam, Jan. Airplane Design: Part 6-Preliminary Calculation of Aerodynamic, Thrust and Power *
    Characteristics. DARcorporation, 1985.
    """

    def setup(self):
        self.add_input("data:aerodynamics:vertical_tail:low_speed:CL_alpha", val=np.nan, units="rad**-1")
        self.add_input("data:geometry:vertical_tail:taper_ratio", val=np.nan)
        self.add_input("data:geometry:vertical_tail:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:vertical_tail:rudder:chord_ratio", val=np.nan)
        self.add_input("data:geometry:vertical_tail:rudder:max_deflection", val=np.nan, units="deg")

        self.add_output("data:aerodynamics:rudder:low_speed:Cy_delta_r", units="rad**-1")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        calculator = ComputeDeltaHighLift()

        taper_ratio_vt = inputs["data:geometry:vertical_tail:taper_ratio"]
        aspect_ratio_vt = float(inputs["data:geometry:vertical_tail:aspect_ratio"])
        rudder_chord_ratio = inputs["data:geometry:vertical_tail:rudder:chord_ratio"]
        rudder_max_deflection = inputs["data:geometry:vertical_tail:rudder:max_deflection"]

        cl_alpha_vt = inputs["data:aerodynamics:vertical_tail:low_speed:CL_alpha"]

        # Assumed that the rudder covers more or less all of the vertical tail while leaving a small gap at the bottom
        # and at the top
        eta_in = 0.05
        eta_out = 0.95
        kb = calculator._compute_kb_flaps(inputs, eta_in, eta_out, taper_ratio_vt)

        # Interpolation of the first graph of figure 8.53 of Roskam
        x_fit = [0.0, 0.02484, 0.05882, 0.09804, 0.14771, 0.19216, 0.2366, 0.32026, 0.38824,
                 0.45359, 0.52157, 0.58693, 0.68105, 0.79085, 1.0]
        y_fit = [0.0, 0.19746, 0.30424, 0.39322, 0.47966, 0.54576, 0.59915, 0.68559, 0.74915, 0.80254, 0.84322, 0.88136,
                 0.92203, 0.96017, 1.0]
        rudder_effectiveness_parameter_inter = interpolate.interp1d(x_fit, y_fit)
        rudder_effectiveness_parameter = rudder_effectiveness_parameter_inter(rudder_chord_ratio)

        k_prime = self.compute_k_prime(aspect_ratio_vt, rudder_effectiveness_parameter)

        rudder_effectiveness_2d = calculator._compute_alpha_flap(rudder_max_deflection, rudder_chord_ratio)

        cy_delta_r = cl_alpha_vt * kb * k_prime * rudder_effectiveness_2d

        outputs["data:aerodynamics:rudder:low_speed:Cy_delta_r"] = cy_delta_r

    @staticmethod
    def compute_k_prime(aspect_ratio, effectiveness_parameter):

        """ Numerisation of the curve given in figure 8.53 of Roskam part VI"""

        ar_array_0_1 = [1.34211, 1.57895, 1.92105, 2.36842, 2.84211, 3, 3.63158, 4, 4.71053, 5, 6, 6.5, 7, 8.02632, 9,
                        10]
        k_prime_array_0_1 = [2.00261, 1.90588, 1.80392, 1.70196, 1.60261, 1.57908, 1.49804, 1.46144, 1.40131, 1.38562,
                             1.3281, 1.30458, 1.28366, 1.24967, 1.22092, 1.20261]

        ar_array_0_2 = [0.52632, 0.63158, 0.81579, 0.97368, 1.05263, 1.42105, 1.9473, 2.02632, 2.71053, 3, 4, 5, 6, 7,
                        8, 9.02632, 10.02632]
        k_prime_array_0_2 = [2.00261, 1.90327, 1.80392, 1.73595, 1.70196, 1.60261, 1.50065, 1.48758, 1.39869, 1.37778,
                             1.29935, 1.24183, 1.20261, 1.17647, 1.15817, 1.1451, 1.13987]

        ar_array_0_3 = [0.18421, 0.21053, 0.31579, 0.5, 0.71053, 1, 1.10526, 1.60526, 2.02632, 2.5, 3, 3.97368, 5, 6,
                        7.02632, 8.02632, 9, 10]
        k_prime_array_0_3 = [2.00261, 1.90327, 1.80392, 1.70196, 1.6, 1.52418, 1.50065, 1.40392, 1.35163, 1.29935,
                             1.26536, 1.2, 1.16601, 1.14248, 1.12418, 1.11111, 1.09804, 1.09804]

        ar_array_0_4 = [0, 0.07895, 0.15789, 0.31579, 0.57895, 1.02632, 0.94737, 1.63158, 2.02632, 2.71053, 3.02632,
                        4.02632, 5.02632, 6.02632, 7.02632, 8, 9, 10]
        k_prime_array_0_4 = [1.87712, 1.80392, 1.69935, 1.59739, 1.50065, 1.38824, 1.40392, 1.29935, 1.25752, 1.19739,
                             1.18431, 1.14248, 1.11634, 1.09804, 1.08758, 1.07974, 1.07451, 1.06667]

        ar_array_0_5 = [0.0, 0.10526, 0.23684, 0.5, 0.89474, 1, 1.81579, 2, 3.02632, 4, 5, 6.02632, 7.02632, 8.02632, 9,
                        10.02632]
        k_prime_array_0_5 = [1.67843, 1.59739, 1.49804, 1.40131, 1.30196, 1.28366, 1.19739, 1.18431, 1.13464, 1.10327,
                             1.08497, 1.0719, 1.06144, 1.05621, 1.05098, 1.04575]

        ar_array_0_6 = [0.0, 0.18421, 0.44737, 0.97368, 1.13158, 2.02632, 2.73684, 3, 4.02632, 5, 6.02632, 7.02632, 8,
                        9, 10.02632]
        k_prime_array_0_6 = [1.48758, 1.39869, 1.30196, 1.20784, 1.2, 1.13464, 1.10065, 1.0902, 1.0719, 1.05882,
                             1.04575, 1.04052, 1.03791, 1.03529, 1.03268]

        ar_array_0_7 = [0, 0.10526, 0.47368, 1, 1.84211, 2.02632, 3, 4.02632, 5, 5.97368, 7, 8.02632, 9, 10.02632]
        k_prime_array_0_7 = [1.34902, 1.30196, 1.2, 1.14248, 1.09542, 1.0902, 1.06405, 1.05098, 1.04052, 1.03268,
                             1.02484, 1.02484, 1.02222, 1.02222]

        ar_array_0_8 = [0.02632, 0.81579, 1, 2.05263, 3, 4.02632, 5, 6, 7, 8, 9.02632, 9.97368]
        k_prime_array_0_8 = [1.20261, 1.10327, 1.0902, 1.05621, 1.04052, 1.03007, 1.01699, 1.01438, 1.01438, 1.01176,
                             1.00915, 1.01176]

        ar_array_0_9 = [0, 0.05263, 1.02632, 2.02632, 3, 4.02632, 5.02632, 6, 7, 8, 10.02632]
        k_prime_array_0_9 = [1.11634, 1.09804, 1.04575, 1.02745, 1.01438, 1.00392, 1.00131, 1.00131, 0.99869, 0.99869,
                             1.00131]

        ar_array_1_0 = [-0.0, 1.02632, 1.97368, 3.02632, 4.02632, 5.02632, 6.02632, 7.02632, 8.02632, 9.05263, 10]
        k_prime_array_1_0 = [0.99608, 0.99869, 0.99869, 0.99608, 0.99608, 0.99608, 0.99869, 0.99346, 0.99608, 0.99869,
                             0.99869]

        if effectiveness_parameter < 0.2:
            ar_array_prec = ar_array_0_1
            k_prime_array_prec = k_prime_array_0_1
            ar_array_next = ar_array_0_2
            k_prime_array_next = k_prime_array_0_2
        elif effectiveness_parameter < 0.3:
            ar_array_prec = ar_array_0_2
            k_prime_array_prec = k_prime_array_0_2
            ar_array_next = ar_array_0_3
            k_prime_array_next = k_prime_array_0_3
        elif effectiveness_parameter < 0.4:
            ar_array_prec = ar_array_0_3
            k_prime_array_prec = k_prime_array_0_3
            ar_array_next = ar_array_0_4
            k_prime_array_next = k_prime_array_0_4
        elif effectiveness_parameter < 0.5:
            ar_array_prec = ar_array_0_4
            k_prime_array_prec = k_prime_array_0_4
            ar_array_next = ar_array_0_5
            k_prime_array_next = k_prime_array_0_5
        elif effectiveness_parameter < 0.6:
            ar_array_prec = ar_array_0_5
            k_prime_array_prec = k_prime_array_0_5
            ar_array_next = ar_array_0_6
            k_prime_array_next = k_prime_array_0_6
        elif effectiveness_parameter < 0.7:
            ar_array_prec = ar_array_0_6
            k_prime_array_prec = k_prime_array_0_6
            ar_array_next = ar_array_0_7
            k_prime_array_next = k_prime_array_0_7
        elif effectiveness_parameter < 0.8:
            ar_array_prec = ar_array_0_7
            k_prime_array_prec = k_prime_array_0_7
            ar_array_next = ar_array_0_8
            k_prime_array_next = k_prime_array_0_8
        elif effectiveness_parameter < 0.9:
            ar_array_prec = ar_array_0_8
            k_prime_array_prec = k_prime_array_0_8
            ar_array_next = ar_array_0_9
            k_prime_array_next = k_prime_array_0_9
        elif effectiveness_parameter < 1.0:
            ar_array_prec = ar_array_0_9
            k_prime_array_prec = k_prime_array_0_9
            ar_array_next = ar_array_1_0
            k_prime_array_next = k_prime_array_1_0
        else:
            ar_array_prec = ar_array_1_0
            k_prime_array_prec = k_prime_array_1_0
            ar_array_next = ar_array_1_0
            k_prime_array_next = k_prime_array_1_0

        k_prime_prec_interp = interpolate.interp1d(ar_array_prec, k_prime_array_prec)
        k_prime_next_interp = interpolate.interp1d(ar_array_next, k_prime_array_next)
        k_prime_prec = k_prime_prec_interp(min(aspect_ratio, 10))
        k_prime_next = k_prime_next_interp(min(aspect_ratio, 10))

        effectiveness_parameter_prec = math.floor(10.*effectiveness_parameter)/10
        effectiveness_parameter_next = math.ceil(10.*effectiveness_parameter)/10

        k_prime_interp = interpolate.interp1d([effectiveness_parameter_prec, effectiveness_parameter_next],
                                              [k_prime_prec, k_prime_next])
        k_prime = k_prime_interp(effectiveness_parameter)

        return k_prime
