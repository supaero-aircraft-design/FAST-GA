"""
Computation of lift and drag increment due to high-lift devices
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
import math
import openmdao.api as om
from pandas import read_csv
from importlib.resources import open_text
from typing import Union, Tuple, Optional
from scipy import interpolate

from . import resources

LIFT_EFFECTIVENESS_FILENAME = "interpolation_of_lift_effectiveness.txt"
DELTA_CL_PLAIN_FLAP = "delta_lift_plain_flap.csv"
K_PLAIN_FLAP = "k_plain_flap.csv"
KB_FLAPS = "kb_flaps.csv"


class ComputeDeltaHighLift(om.ExplicitComponent):
    """
    Provides lift and drag increments due to high-lift devices
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.phase = None

    def setup(self):

        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:horizontal_tail:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:taper_ratio", val=np.nan)
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:thickness_ratio", val=np.nan)
        self.add_input("data:geometry:wing:sweep_25", val=np.nan, units="deg")
        self.add_input("data:geometry:flap:chord_ratio", val=0.2)
        self.add_input("data:geometry:flap:span_ratio", val=np.nan)
        self.add_input("data:geometry:flap_type", val=np.nan)
        self.add_input("data:geometry:horizontal_tail:elevator_chord_ratio", val=np.nan)
        self.add_input("data:geometry:horizontal_tail:thickness_ratio", val=np.nan)
        self.add_input("data:aerodynamics:wing:low_speed:CL_alpha", val=np.nan, units="rad**-1")
        self.add_input("data:aerodynamics:low_speed:mach", val=np.nan)
        self.add_input("data:mission:sizing:landing:flap_angle", val=30.0, units="deg")
        self.add_input("data:mission:sizing:takeoff:flap_angle", val=10.0, units="deg")

        self.add_output("data:aerodynamics:flaps:landing:CL")
        self.add_output("data:aerodynamics:flaps:landing:CL_max")
        self.add_output("data:aerodynamics:flaps:landing:CM")
        self.add_output("data:aerodynamics:flaps:landing:CD")
        self.add_output("data:aerodynamics:flaps:takeoff:CL")
        self.add_output("data:aerodynamics:flaps:takeoff:CL_max")
        self.add_output("data:aerodynamics:flaps:takeoff:CM")
        self.add_output("data:aerodynamics:flaps:takeoff:CD")
        self.add_output("data:aerodynamics:elevator:low_speed:CL_delta", units="rad**-1")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        mach_ls = inputs["data:aerodynamics:low_speed:mach"]

        # Computes flaps contribution during low speed operations (take-off/landing)
        for self.phase in ['landing', 'takeoff']:
            if self.phase == 'landing':
                flap_angle = float(inputs["data:mission:sizing:landing:flap_angle"])
                outputs["data:aerodynamics:flaps:landing:CL"], \
                outputs["data:aerodynamics:flaps:landing:CL_max"] = self._get_flaps_delta_cl(
                    inputs,
                    flap_angle,
                    mach_ls,
                )
                outputs["data:aerodynamics:flaps:landing:CM"] = self._get_flaps_delta_cm(
                    inputs,
                    flap_angle,
                    mach_ls,
                )
                outputs["data:aerodynamics:flaps:landing:CD"] = self._get_flaps_delta_cd(
                    inputs,
                    flap_angle,
                )
            else:
                flap_angle = float(inputs["data:mission:sizing:takeoff:flap_angle"])
                outputs["data:aerodynamics:flaps:takeoff:CL"], \
                outputs["data:aerodynamics:flaps:takeoff:CL_max"] = self._get_flaps_delta_cl(
                    inputs,
                    flap_angle,
                    mach_ls,
                )
                outputs["data:aerodynamics:flaps:takeoff:CM"] = self._get_flaps_delta_cm(
                    inputs,
                    flap_angle,
                    mach_ls,
                )
                outputs["data:aerodynamics:flaps:takeoff:CD"] = self._get_flaps_delta_cd(
                    inputs,
                    flap_angle,
                )

        # Computes elevator contribution during low speed operations (for different deflection angle)
        outputs["data:aerodynamics:elevator:low_speed:CL_delta"] = self._get_elevator_delta_cl(
            inputs,
            25.0,
        )  # get derivative for 25° angle assuming it is linear when <= to 25 degree,
        # derivative wrt to the wing

    def _get_elevator_delta_cl(self, inputs, elevator_angle: Union[float, np.array]) -> Union[float, np.array]:
        """
        Applies the plain flap lift variation function :meth:`_delta_lift_plainflap`.

        :param elevator_angle: elevator angle (in Degree)
        :return: lift coefficient derivative
        """

        ht_area = inputs["data:geometry:horizontal_tail:area"]
        wing_area = inputs["data:geometry:wing:area"]
        elevator_chord_ratio = inputs["data:geometry:horizontal_tail:elevator_chord_ratio"]
        htp_thickness_ratio = inputs["data:geometry:horizontal_tail:thickness_ratio"]

        # Elevator (plain flap). Default: maximum deflection (25deg)
        cl_delta_theory, k = self._delta_lift_plainflap(abs(elevator_angle), elevator_chord_ratio, htp_thickness_ratio)
        cl_alpha_elev = (cl_delta_theory * k) * ht_area / wing_area
        cl_alpha_elev *= 0.9  # Correction for the central fuselage part (no elevator there)

        return cl_alpha_elev

    def _get_flaps_delta_cl(self, inputs, flap_angle: float, mach: float) -> Tuple[float, float]:
        """
        Method based on...

        :param flap_angle: flap angle (in Degree)
        :param mach: air speed
        :return: increment of lift coefficient
        """

        cl_alpha_wing = inputs['data:aerodynamics:wing:low_speed:CL_alpha']
        span_wing = inputs['data:geometry:wing:span']
        y1_wing = inputs['data:geometry:fuselage:maximum_width'] / 2.0
        y2_wing = inputs['data:geometry:wing:root:y']
        flap_span_ratio = inputs['data:geometry:flap:span_ratio']

        # 2D flap lift coefficient
        delta_cl_airfoil = self._compute_delta_cl_airfoil_2D(inputs, flap_angle, mach)
        # Roskam 3D flap parameters
        eta_in = y1_wing / (span_wing / 2.0)
        eta_out = ((y2_wing - y1_wing) + flap_span_ratio * (span_wing / 2.0 - y2_wing)) / (span_wing / 2.0 - y2_wing)
        kb = self._compute_kb_flaps(inputs, eta_in, eta_out)
        effect = 1.04  # fig 8.53 (cf/c=0.25, small effect of AR)
        delta_cl0_flaps = kb * delta_cl_airfoil * (cl_alpha_wing / (2 * math.pi)) * effect
        delta_clmax_flaps = self._compute_delta_clmax_flaps(inputs, flap_angle)

        return delta_cl0_flaps, delta_clmax_flaps

    def _get_flaps_delta_cm(self, inputs, flap_angle: float, mach: float) -> float:
        """
        Method based on Roskam book

        :param flap_angle: flap angle (in Degree)
        :param mach: air speed
        :return: increment of moment coefficient
        """

        wing_taper_ratio = float(inputs['data:geometry:wing:taper_ratio'])

        # Method from Roskam (sweep=0, flaps 60%, simple slotted and not extensible, at 25% MAC, cf/c+0.25)
        k_p = interpolate.interp1d([0., 0.2, 0.33, 0.5, 1.], [0.65, 0.75, 0.7, 0.63, 0.5])
        # k_p: Figure 8.105, interpolated function of taper ratio (span ratio fixed)
        delta_cl_flap = self._get_flaps_delta_cl(inputs, flap_angle, mach)[0]
        delta_cm_flap = k_p(min(max(0.0, wing_taper_ratio), 1.0)) * (-0.27) * delta_cl_flap  # -0.27: Figure 8.106

        return delta_cm_flap

    def _get_flaps_delta_cd(self, inputs, flap_angle: float) -> float:
        """
        Method from Young (in Gudmunsson book; page 725)
        
        :param flap_angle: flap angle (in Degree)
        :return: increment of drag coefficient
        """

        flap_type = inputs['data:geometry:flap_type']
        flap_chord_ratio = inputs['data:geometry:flap:chord_ratio']
        flap_area_ratio = self._compute_flap_area_ratio(inputs)

        if flap_type == 1.0:  # slotted flap
            k1 = 179.32 * flap_chord_ratio ** 4 - \
                 111.6 * flap_chord_ratio ** 3 + \
                 28.929 * flap_chord_ratio ** 2 + \
                 2.3705 * flap_chord_ratio - 0.0089
            k2 = -3.9877E-12 * flap_angle ** 6 + \
                 1.1685e-9 * flap_angle ** 5 - \
                 1.2846e-7 * flap_angle ** 4 + \
                 6.1742e-6 * flap_angle ** 3 - \
                 9.89444e-5 * flap_angle ** 2 + \
                 6.8324e-4 * flap_angle - \
                 3.892e-4

        else:  # plain flap
            k1 = - 21.09 * flap_chord_ratio ** 3 + \
                 14.091 * flap_chord_ratio ** 2 + \
                 3.165 * flap_chord_ratio - \
                 0.00103
            k2 = -3.795E-7 * flap_angle ** 3 + \
                 5.387E-5 * flap_angle ** 2 + \
                 6.843E-4 * flap_angle - \
                 1.4729E-3
        delta_cd_flaps = k1 * k2 * flap_area_ratio

        return delta_cd_flaps

    def _compute_delta_cl_airfoil_2D(self, inputs, angle: float, mach: float) -> float:
        """
        Compute airfoil 2D lift contribution.

        :param angle: airfoil angle (in Degree)
        :param mach: air speed
        :return: increment of lift coefficient
        """

        flap_type = inputs['data:geometry:flap_type']
        flap_chord_ratio = float(inputs['data:geometry:flap:chord_ratio'])

        # 2D flap lift coefficient
        if flap_type == 1:  # Slotted flap
            alpha_flap = self._compute_alpha_flap(angle, flap_chord_ratio)
            delta_cl_airfoil = 2 * math.pi / math.sqrt(1 - mach ** 2) * alpha_flap * (angle * math.pi / 180)
        else:  # Plain flap
            cl_delta_theory, k = self._delta_lift_plainflap(angle, flap_chord_ratio)
            delta_cl_airfoil = cl_delta_theory * k * (angle * math.pi / 180)

        return delta_cl_airfoil

    def _compute_delta_clmax_flaps(self, inputs, flap_angle) -> float:
        """

        Method from Roskam vol.6.  Particularised for single slotted flaps in 
        airfoils with 12% thickness (which is the design case); with
        chord ratio of 0.25 and typical flap deflections (30deg landing, 10deg TO).
        Plain flap included (40 deg landing deflection here)
        """

        flap_type = inputs['data:geometry:flap_type']
        el_aero = inputs["data:geometry:wing:thickness_ratio"]
        flap_chord_ratio = inputs["data:geometry:flap:chord_ratio"]
        sweep_25 = inputs["data:geometry:wing:sweep_25"] * math.pi / 180.
        flap_area_ratio = self._compute_flap_area_ratio(inputs)

        # Numerisation of figure 8.31
        # For the plain flaps
        el_aero_array = np.array(
            [0.97561, 1.98374, 2.99187, 3.96748, 5.00813, 5.98374, 7.02439, 8, 9.00813, 10.01626, 10.99187,
             12, 13.04065, 14.01626, 15.02439, 16, 16.97561, 17.98374, 18.99187, 19.96748])
        base_increment_plain = np.array(
            [1.0102, 1.00626, 0.99287, 0.98265, 0.95664, 0.92437, 0.87001, 0.8283, 0.8149, 0.82356, 0.86059, 0.91963,
             0.99126, 1.08183, 1.18812, 1.31333, 1.4291, 1.51335, 1.56925, 1.59682])
        base_increment_plain_interp = interpolate.interp1d(el_aero_array, base_increment_plain)

        flap_chord_ratio_array = np.array(
            [0.0, 1.09225, 2.06642, 4.01476, 6.00738, 8, 9.94834, 11.94096, 13.97786, 15.9262, 17.91882, 19.91144,
             21.85978, 23.80812, 25.80074, 27.74908, 29.69742])
        k1_array_plain = np.array([0.0, 0.1, 0.18667, 0.33556, 0.46667, 0.56889, 0.65111, 0.72, 0.78, 0.83778, 0.88444,
                                   0.92444, 0.96222, 0.99111, 1.01778, 1.04222, 1.06222])
        k1_plain_interp = interpolate.interp1d(flap_chord_ratio_array, k1_array_plain)

        flap_deflection_array_plain = np.array(
            [0, 2.92929, 5.05051, 5.9596, 8.9899, 10, 12.62626, 15.15152, 16.86869, 20.10101, 21.11111, 24.94949,
             26.36364, 29.89899, 32.82828, 34.94949, 39.79798, 42.62626, 44.84848, 49.79798, 54.84848, 60.0
             ])
        k2_array_plain = np.array(
            [0, 0.09899, 0.16768, 0.19798, 0.29899, 0.32929, 0.39798, 0.46061, 0.49899, 0.57374, 0.59596, 0.67071,
             0.69697, 0.75758, 0.79394, 0.8202, 0.87071, 0.89697, 0.91111, 0.94141, 0.97172, 1.0
             ])
        k2_plain_interp = interpolate.interp1d(flap_deflection_array_plain, k2_array_plain)

        # For the slotted flaps
        base_increment_simple_slot = np.array(
            [1.00705, 1.00629, 1.00864, 1.00782, 1.01018, 1.02516, 1.04956, 1.08341, 1.12359, 1.17004, 1.22911, 1.30704,
             1.37868, 1.46925, 1.5724, 1.65667, 1.70944, 1.72125, 1.711, 1.66929
             ])
        base_increment_simple_slot_interp = interpolate.interp1d(el_aero_array, base_increment_simple_slot)

        k1_array_simple_slot = k1_array_plain
        k1_simple_slot_interp = interpolate.interp1d(flap_chord_ratio_array, k1_array_simple_slot)

        flap_deflection_array_simple_slot = np.array(
            [0, 0.80808, 4.14141, 5.05051, 7.37374, 10.20202, 11.51515, 15.05051, 15.55556, 20, 24.94949, 25.65657,
             30.10101, 32.12121, 34.94949, 39.69697, 45.0])
        k2_array_simple_slot = np.array(
            [0.1798, 0.2, 0.29495, 0.32323, 0.4, 0.46869, 0.49899, 0.5899, 0.59798, 0.69495, 0.78788, 0.79596, 0.86465,
             0.89495, 0.92525, 0.96768, 1.])
        k2_simple_slot_interp = interpolate.interp1d(flap_deflection_array_simple_slot, k2_array_simple_slot)
        flap_deflection_ratio_array = np.array(
            [0.07097, 0.10108, 0.15054, 0.2, 0.23226, 0.30323, 0.31398, 0.3957, 0.4043, 0.48602, 0.50108, 0.60215,
             0.70108, 0.8, 0.84516, 0.90108, 1.0])
        k3_array_simple_slot = np.array(
            [0.1, 0.1375, 0.2, 0.26458, 0.30208, 0.3875, 0.4, 0.50208, 0.51042, 0.6, 0.61458, 0.71042, 0.78958, 0.86667,
             0.89792, 0.93958, 1.0])
        k3_simple_slot_interp = interpolate.interp1d(flap_deflection_ratio_array, k3_array_simple_slot)

        if flap_type == 1.0:  # simple slotted
            base_increment = base_increment_simple_slot_interp(el_aero * 100.)
            k1 = k1_simple_slot_interp(flap_chord_ratio * 100.)
            k2 = k2_simple_slot_interp(flap_angle)
            reference_flap_deflection = 45.0
            k3 = k3_simple_slot_interp(flap_angle/reference_flap_deflection)

        else:  # plain flap
            base_increment = base_increment_plain_interp(el_aero * 100.)
            k1 = k1_plain_interp(flap_chord_ratio * 100.)
            k2 = k2_plain_interp(flap_angle)
            k3 = 1.0

        k_planform = (1. - 0.08 * math.cos(sweep_25) ** 2.) * math.cos(sweep_25) ** (3. / 4.)
        delta_clmax_flaps = base_increment * k1 * k2 * k3 * k_planform * flap_area_ratio

        return delta_clmax_flaps

    @staticmethod
    def _compute_flap_area_ratio(inputs) -> float:
        """
        Compute ratio of flap over wing (reference area).
        Takes into account the wing portion under the fuselage.
        """

        wing_span = inputs["data:geometry:wing:span"]
        wing_area = inputs["data:geometry:wing:area"]
        wing_taper_ratio = inputs['data:geometry:wing:taper_ratio']
        y1_wing = inputs['data:geometry:fuselage:maximum_width'] / 2.0
        y2_wing = inputs['data:geometry:wing:root:y']
        wing_root_chord = inputs['data:geometry:wing:root:chord']
        flap_span_ratio = inputs['data:geometry:flap:span_ratio']

        flap_area = (y2_wing - y1_wing) * wing_root_chord + \
                    flap_span_ratio * (wing_span / 2. - y2_wing) * \
                    (wing_root_chord * (2 - (1 - wing_taper_ratio) * flap_span_ratio)) * 0.5

        flap_area_ratio = 2 * flap_area / wing_area

        return flap_area_ratio

    @staticmethod
    def _compute_alpha_flap(flap_angle: float, chord_ratio: float) -> np.ndarray:
        """
        Roskam data to calculate the effectiveness of a simple slotted flap.

        :param flap_angle: flap angle (in Degree)
        :param chord_ratio: position of flap on wing chord
        :return: effectiveness ratio
        """

        temp_array = []
        with open_text(resources, LIFT_EFFECTIVENESS_FILENAME) as file:
            for line in file:
                temp_array.append([float(x) for x in line.split(",")])
        x1 = []
        y1 = []
        x2 = []
        y2 = []
        x3 = []
        y3 = []
        x4 = []
        y4 = []
        x5 = []
        y5 = []
        for arr in temp_array:
            x1.append(arr[0])
            y1.append(arr[1])
            x2.append(arr[2])
            y2.append(arr[3])
            x3.append(arr[4])
            y3.append(arr[5])
            x4.append(arr[6])
            y4.append(arr[7])
            x5.append(arr[8])
            y5.append(arr[9])
        tck1 = interpolate.splrep(x1, y1, s=0)
        tck2 = interpolate.splrep(x2, y2, s=0)
        tck3 = interpolate.splrep(x3, y3, s=0)
        tck4 = interpolate.splrep(x4, y4, s=0)
        tck5 = interpolate.splrep(x5, y5, s=0)
        ynew1 = interpolate.splev(min(max(flap_angle, min(x1)), max(x1)), tck1, der=0)
        ynew2 = interpolate.splev(min(max(flap_angle, min(x2)), max(x2)), tck2, der=0)
        ynew3 = interpolate.splev(min(max(flap_angle, min(x3)), max(x3)), tck3, der=0)
        ynew4 = interpolate.splev(min(max(flap_angle, min(x4)), max(x4)), tck4, der=0)
        ynew5 = interpolate.splev(min(max(flap_angle, min(x5)), max(x5)), tck5, der=0)
        zs = [0.15, 0.20, 0.25, 0.30, 0.40]
        y_final = [float(ynew1), float(ynew2), float(ynew3), float(ynew4), float(ynew5)]
        tck6 = interpolate.splrep(zs, y_final, s=0)
        effectiveness = interpolate.splev(min(max(chord_ratio, min(zs)), max(zs)), tck6, der=0)

        return effectiveness

    @staticmethod
    def _delta_lift_plainflap(
            flap_angle: Union[float, np.array],
            chord_ratio: float,
            thickness: Optional[float] = 0.12
    ) -> Tuple[np.array, np.array]:
        """
        Roskam data to estimate plain flap lift increment and correction factor K.

        :param flap_angle: flap angle (in Degree)
        :param chord_ratio: position of flap on wing chord
        :return: lift increment and correction factor
        """

        file = pth.join(resources.__path__[0], DELTA_CL_PLAIN_FLAP)
        db = read_csv(file)

        x_0 = db['X_0']
        y_0 = db['Y_0']
        errors = np.logical_or(np.isnan(x_0), np.isnan(y_0))
        x_0 = x_0[np.logical_not(errors)].tolist()
        y_0 = y_0[np.logical_not(errors)].tolist()
        x_04 = db['X_04']
        y_04 = db['Y_04']
        errors = np.logical_or(np.isnan(x_04), np.isnan(y_04))
        x_04 = x_04[np.logical_not(errors)].tolist()
        y_04 = y_04[np.logical_not(errors)].tolist()
        x_10 = db['X_10']
        y_10 = db['Y_10']
        errors = np.logical_or(np.isnan(x_10), np.isnan(y_10))
        x_10 = x_10[np.logical_not(errors)].tolist()
        y_10 = y_10[np.logical_not(errors)].tolist()
        x_15 = db['X_15']
        y_15 = db['Y_15']
        errors = np.logical_or(np.isnan(x_15), np.isnan(y_15))
        x_15 = x_15[np.logical_not(errors)].tolist()
        y_15 = y_15[np.logical_not(errors)].tolist()
        cld_thk0 = interpolate.interp1d(x_0, y_0)
        cld_thk04 = interpolate.interp1d(x_04, y_04)
        cld_thk10 = interpolate.interp1d(x_10, y_10)
        cld_thk15 = interpolate.interp1d(x_15, y_15)
        cld_t = [float(cld_thk0(min(max(chord_ratio, min(x_0)), max(x_0)))),
                 float(cld_thk04(min(max(chord_ratio, min(x_04)), max(x_04)))),
                 float(cld_thk10(min(max(chord_ratio, min(x_10)), max(x_10)))),
                 float(cld_thk15(min(max(chord_ratio, min(x_15)), max(x_15))))]
        cl_delta = interpolate.interp1d([0.0, 0.04, 0.1, 0.15], cld_t)(min(thickness, 0.15))

        file = pth.join(resources.__path__[0], K_PLAIN_FLAP)
        db = read_csv(file)

        x_10 = db['X_10']
        y_10 = db['Y_10']
        errors = np.logical_or(np.isnan(x_10), np.isnan(y_10))
        x_10 = x_10[np.logical_not(errors)].tolist()
        y_10 = y_10[np.logical_not(errors)].tolist()
        x_15 = db['X_15']
        y_15 = db['Y_15']
        errors = np.logical_or(np.isnan(x_15), np.isnan(y_15))
        x_15 = x_15[np.logical_not(errors)].tolist()
        y_15 = y_15[np.logical_not(errors)].tolist()
        x_25 = db['X_25']
        y_25 = db['Y_25']
        errors = np.logical_or(np.isnan(x_25), np.isnan(y_25))
        x_25 = x_25[np.logical_not(errors)].tolist()
        y_25 = y_25[np.logical_not(errors)].tolist()
        x_30 = db['X_30']
        y_30 = db['Y_30']
        errors = np.logical_or(np.isnan(x_30), np.isnan(y_30))
        x_30 = x_30[np.logical_not(errors)].tolist()
        y_30 = y_30[np.logical_not(errors)].tolist()
        x_40 = db['X_40']
        y_40 = db['Y_40']
        errors = np.logical_or(np.isnan(x_40), np.isnan(y_40))
        x_40 = x_40[np.logical_not(errors)].tolist()
        y_40 = y_40[np.logical_not(errors)].tolist()
        x_50 = db['X_50']
        y_50 = db['Y_50']
        errors = np.logical_or(np.isnan(x_50), np.isnan(y_50))
        x_50 = x_50[np.logical_not(errors)].tolist()
        y_50 = y_50[np.logical_not(errors)].tolist()
        k_chord10 = interpolate.interp1d(x_10, y_10)
        k_chord15 = interpolate.interp1d(x_15, y_15)
        k_chord25 = interpolate.interp1d(x_25, y_25)
        k_chord30 = interpolate.interp1d(x_30, y_30)
        k_chord40 = interpolate.interp1d(x_40, y_40)
        k_chord50 = interpolate.interp1d(x_50, y_50)
        k = []
        if type(flap_angle) == float:
            flap_angle = [flap_angle]
        else:
            flap_angle = list(flap_angle)
        for angle in flap_angle:
            k_chord = [float(k_chord10(max(min(angle, max(x_10)), min(x_10)))),
                       float(k_chord15(max(min(angle, max(x_15)), min(x_15)))),
                       float(k_chord25(max(min(angle, max(x_25)), min(x_25)))),
                       float(k_chord30(max(min(angle, max(x_30)), min(x_30)))),
                       float(k_chord40(max(min(angle, max(x_40)), min(x_40)))),
                       float(k_chord50(max(min(angle, max(x_50)), min(x_50))))]
            k.append(float(interpolate.interp1d([0.1, 0.15, 0.25, 0.3, 0.4, 0.5], k_chord)
                           (min(max(chord_ratio, 0.1), 0.5))))
        k = np.array(k)

        return cl_delta, k

    @staticmethod
    def _compute_kb_flaps(inputs, eta_in: float, eta_out: float) -> float:
        """
        Use Roskam graph (Figure 8.52) to interpolate kb factor.
        This factor accounts for a finite flap contribution to the 3D lift increase, depending on its position and size
        and the taper ratio of the wing.

        :param eta_in: ????
        :param eta_out: ????
        :return: kb factor contribution to 3D lift
        """

        eta_in = float(eta_in)
        eta_out = float(eta_out)
        wing_taper_ratio = max(min(float(inputs['data:geometry:wing:taper_ratio']), 1.0), 0.0)
        file = pth.join(resources.__path__[0], KB_FLAPS)
        db = read_csv(file)

        x_0 = db['X_0']
        y_0 = db['Y_0']
        errors = np.logical_or(np.isnan(x_0), np.isnan(y_0))
        x_0 = x_0[np.logical_not(errors)].tolist()
        y_0 = y_0[np.logical_not(errors)].tolist()
        x_05 = db['X_0.5']
        y_05 = db['Y_0.5']
        errors = np.logical_or(np.isnan(x_05), np.isnan(y_05))
        x_05 = x_05[np.logical_not(errors)].tolist()
        y_05 = y_05[np.logical_not(errors)].tolist()
        x_1 = db['X_1']
        y_1 = db['Y_1']
        errors = np.logical_or(np.isnan(x_1), np.isnan(y_1))
        x_1 = x_1[np.logical_not(errors)].tolist()
        y_1 = y_1[np.logical_not(errors)].tolist()
        k_taper0 = interpolate.interp1d(x_0, y_0)
        k_taper05 = interpolate.interp1d(x_05, y_05)
        k_taper1 = interpolate.interp1d(x_1, y_1)
        k_eta = [float(k_taper0(min(max(eta_in, min(x_0)), max(x_0)))),
                 float(k_taper05(min(max(eta_in, min(x_05)), max(x_05)))),
                 float(k_taper1(min(max(eta_in, min(x_1)), max(x_1))))]
        kb_in = interpolate.interp1d([0.0, 0.5, 1.0], k_eta)(wing_taper_ratio)
        k_eta = [float(k_taper0(min(max(eta_out, min(x_0)), max(x_0)))),
                 float(k_taper05(min(max(eta_out, min(x_05)), max(x_05)))),
                 float(k_taper1(min(max(eta_out, min(x_1)), max(x_1))))]
        kb_out = interpolate.interp1d([0.0, 0.5, 1.0], k_eta)(wing_taper_ratio)

        return float(kb_out - kb_in)
