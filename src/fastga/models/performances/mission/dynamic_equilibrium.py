"""
    FAST - Copyright (c) 2016 ONERA ISAE.
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

import os
import math
import logging
import numpy as np
import openmdao.api as om
from scipy.constants import g
from scipy.optimize import fsolve
import pandas as pd

from stdatm import Atmosphere

CSV_DATA_LABELS = [
    "time",
    "altitude",
    "ground_distance",
    "mass",
    "true_airspeed",
    "equivalent_airspeed",
    "mach",
    "density",
    "gamma",
    "alpha",
    "cl_wing",
    "cl_htp",
    "thrust (N)",
    "thrust_rate",
    "tsfc (kg/s/N)",
    "name",
]

_LOGGER = logging.getLogger(__name__)


class DynamicEquilibrium(om.ExplicitComponent):
    """
    Compute the derivatives and associated lift-drag-thrust decomposition depending if DP model
    is included or not.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cl_wing_sol = 0.0
        self.cl_tail_sol = 0.0
        self.error_on_pitch_equilibrium = False
        self.delta_e_sol = 0.0

    def initialize(self):
        self.options.declare("out_file", default="", types=str)

    def setup(self):
        self.add_input("data:geometry:wing:MAC:leading_edge:x:local", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:virtual_chord", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input(
            "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="m"
        )
        self.add_input("data:aerodynamics:wing:cruise:CL_alpha", val=np.nan, units="rad**-1")
        self.add_input("data:aerodynamics:wing:cruise:CL0_clean", val=np.nan)
        self.add_input("data:aerodynamics:wing:cruise:CM0_clean", val=np.nan)
        self.add_input("data:aerodynamics:wing:low_speed:CL_alpha", val=np.nan, units="rad**-1")
        self.add_input("data:aerodynamics:wing:low_speed:CL0_clean", val=np.nan)
        self.add_input("data:aerodynamics:wing:low_speed:CM0_clean", val=np.nan)
        self.add_input("data:aerodynamics:wing:low_speed:CL_max_clean", val=np.nan)
        self.add_input("data:aerodynamics:fuselage:cm_alpha", val=np.nan, units="rad**-1")
        self.add_input("data:aerodynamics:horizontal_tail:cruise:CL0", val=np.nan)
        self.add_input(
            "data:aerodynamics:horizontal_tail:cruise:CL_alpha", val=np.nan, units="rad**-1"
        )
        self.add_input(
            "data:aerodynamics:horizontal_tail:cruise:CL_alpha_isolated",
            val=np.nan,
            units="rad**-1",
        )
        self.add_input("data:aerodynamics:horizontal_tail:low_speed:CL0", val=np.nan)
        self.add_input(
            "data:aerodynamics:horizontal_tail:low_speed:CL_alpha", val=np.nan, units="rad**-1"
        )
        self.add_input(
            "data:aerodynamics:horizontal_tail:low_speed:CL_alpha_isolated",
            val=np.nan,
            units="rad**-1",
        )
        self.add_input("data:aerodynamics:horizontal_tail:low_speed:CL_max_clean", val=np.nan)
        self.add_input("data:aerodynamics:horizontal_tail:low_speed:CL_min_clean", val=np.nan)
        self.add_input("data:aerodynamics:elevator:low_speed:CL_delta", val=np.nan, units="rad**-1")
        self.add_input("data:aerodynamics:elevator:low_speed:CD_delta", val=np.nan, units="rad**-2")
        self.add_input("data:weight:aircraft:CG:aft:x", val=np.nan, units="m")
        self.add_input(
            "data:weight:aircraft:in_flight_variation:fixed_mass_comp:equivalent_moment",
            val=np.nan,
            units="kg*m",
        )
        self.add_input("data:weight:propulsion:tank:CG:x", val=np.nan, units="m")
        self.add_input(
            "data:weight:aircraft:in_flight_variation:fixed_mass_comp:mass", val=np.nan, units="kg"
        )
        self.add_input("data:weight:aircraft_empty:CG:z", val=np.nan, units="m")
        self.add_input("data:weight:propulsion:engine:CG:z", val=np.nan, units="m")
        self.add_input("data:mission:sizing:main_route:cruise:altitude", val=np.nan, units="m")

    def dynamic_equilibrium(
        self,
        inputs,
        gamma: float,
        q: float,
        dvx_dt: float,
        dvz_dt: float,
        mass: float,
        flap_condition: str,
        previous_step: tuple,
        low_speed: bool = False,
        x_cg=None,
    ):
        """
        Method that finds the regulated thrust and aircraft to air angle to obtain dynamic
        equilibrium

        :param inputs: inputs derived from aero and mass models
        :param gamma: path angle (in rad.) equal to climb rate c=dh/dt over air speed V,
        sin(gamma)=c/V
        :param q: dynamic pressure q=1/2*rho*V²
        :param dvx_dt: acceleration linear to air speed
        :param dvz_dt: acceleration perpendicular to air speed
        :param mass: current mass of the flying aircraft (taking into account propulsion consumption
        if needed)
        :param flap_condition: can refer either to "takeoff" or "landing" if high-lift contribution
        should be considered
        :param previous_step: give previous step equilibrium if known to accelerate the calculation
        :param low_speed: define which aerodynamic models should be used (either low speed or high
        speed)
        :param x_cg: x position of the center of gravity of the aircraft, if not given, computed
        based on fuel in tank
        """

        cl_max_clean_htp = inputs["data:aerodynamics:horizontal_tail:low_speed:CL_max_clean"]
        cl_min_clean_htp = inputs["data:aerodynamics:horizontal_tail:low_speed:CL_min_clean"]

        if len(previous_step) == 2:
            result = fsolve(
                self.equation_outer,
                np.array([previous_step[0] * 180.0 / math.pi, previous_step[1] / 1000.0]),
                args=(inputs, gamma, q, dvx_dt, dvz_dt, mass, flap_condition, low_speed, x_cg),
                xtol=1.0e-3,
            )
        else:
            result = fsolve(
                self.equation_outer,
                np.array([0.0, 1.0]),
                args=(inputs, gamma, q, dvx_dt, dvz_dt, mass, flap_condition, low_speed, x_cg),
                xtol=1.0e-3,
            )
        alpha_equilibrium = result[0] * math.pi / 180.0
        # noinspection PyTypeChecker
        thrust_equilibrium = result[1] * 1000.0

        cl_wing_local = self.cl_wing_sol
        cl_htp_local = self.cl_tail_sol
        delta_elevator = self.delta_e_sol

        error_on_htp = bool((cl_htp_local > cl_max_clean_htp) or (cl_htp_local < cl_min_clean_htp))

        error_on_wing = self.error_on_pitch_equilibrium

        error = error_on_htp or error_on_wing

        return (
            alpha_equilibrium,
            thrust_equilibrium,
            cl_wing_local,
            cl_htp_local,
            delta_elevator,
            error,
        )

    @staticmethod
    def found_cl_repartition(
        inputs,
        load_factor: float,
        mass: float,
        q: float,
        delta_cm: float,
        low_speed: bool = False,
        x_cg: float = None,
    ):
        """
        Method that founds the lift equilibrium with regard to the global moment

        :param inputs: inputs derived from aero and mass models
        :param load_factor: load factor applied to the
        aircraft expressed as a ratio of g
        :param mass: current aircraft mass
        :param q: dynamic pressure q=1/2*rho*V²
        :param delta_cm: DP induced cm to be added to the moment equilibrium
        :param low_speed: define which aerodynamic models should be used (either low speed or high
        speed)
        :param x_cg: x_cg position of the aircraft, can be specified. If not specified, computed
        based on current fuel
        in the aircraft
        """

        l0_wing = inputs["data:geometry:wing:MAC:length"]
        x_wing = inputs["data:geometry:wing:MAC:at25percent:x"]
        wing_area = inputs["data:geometry:wing:area"]
        x_htp = x_wing + inputs["data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"]
        cm_alpha_fus = inputs["data:aerodynamics:fuselage:cm_alpha"]
        if low_speed:
            cl_alpha_wing = inputs["data:aerodynamics:wing:low_speed:CL_alpha"]
            cl0_wing = inputs["data:aerodynamics:wing:low_speed:CL0_clean"]
            cm0_wing = inputs["data:aerodynamics:wing:low_speed:CM0_clean"]
        else:
            cl_alpha_wing = inputs["data:aerodynamics:wing:cruise:CL_alpha"]
            cl0_wing = inputs["data:aerodynamics:wing:cruise:CL0_clean"]
            cm0_wing = inputs["data:aerodynamics:wing:cruise:CM0_clean"]
        cl_max_clean = inputs["data:aerodynamics:wing:low_speed:CL_max_clean"]

        if x_cg is None:
            c1 = inputs[
                "data:weight:aircraft:in_flight_variation:fixed_mass_comp:equivalent_moment"
            ]
            cg_tank = inputs["data:weight:propulsion:tank:CG:x"]
            c3 = inputs["data:weight:aircraft:in_flight_variation:fixed_mass_comp:mass"]
            fuel_mass = mass - c3
            x_cg = (c1 + cg_tank * fuel_mass) / (c3 + fuel_mass)

        # Define matrix equilibrium (applying load and moment equilibrium)
        a11 = 1
        a12 = 1
        b1 = mass * g * load_factor / (q * wing_area)
        a21 = (x_wing - x_cg) - (cm_alpha_fus / cl_alpha_wing) * l0_wing
        a22 = x_htp - x_cg
        b2 = (cm0_wing + delta_cm + (cm_alpha_fus / cl_alpha_wing) * cl0_wing) * l0_wing

        a = np.array([[a11, a12], [float(a21), float(a22)]])
        b = np.array([b1, b2])
        inv_a = np.linalg.inv(a)
        cl_array = np.dot(inv_a, b)

        # Return equilibrated lift coefficients if low speed maximum clean Cl not exceeded
        # otherwise only cl_wing, 3rd term is an error flag returned by the function
        if cl_array[0] < cl_max_clean:
            cl_wing_return = float(cl_array[0])
            cl_htp_return = float(cl_array[1])
            error = False
        else:
            cl_wing_return = float(mass * g * load_factor / (q * wing_area))
            cl_htp_return = 0.0
            error = True

        return cl_wing_return, cl_htp_return, error

    def save_csv(
        self,
        dataframe_to_add: pd.DataFrame,
    ):
        """
        Method to save mission point to .csv file for further post-processing

        :param dataframe_to_add: Dataframe to add to the csv
        """

        if not os.path.exists(self.options["out_file"]):
            dataframe_to_add.index = range(len(dataframe_to_add))
            dataframe_to_add.to_csv(self.options["out_file"])
        else:
            dataframe_existing = pd.read_csv(self.options["out_file"])
            if "Unnamed: 0" in dataframe_existing.columns:
                dataframe_existing = dataframe_existing.drop("Unnamed: 0", axis=1)
            dataframe_to_add.index = range(
                max(dataframe_existing.index), max(dataframe_existing.index) + len(dataframe_to_add)
            )
            dataframe_existing = pd.concat([dataframe_existing, dataframe_to_add])
            dataframe_existing.to_csv(self.options["out_file"])

    def equation_outer(
        self,
        x,
        inputs,
        gamma: float,
        q: float,
        dvx_dt: float,
        dvz_dt: float,
        mass: float,
        flap_condition: str,
        low_speed: bool = False,
        x_cg=None,
    ):

        # Define the system of equations to be solved: load equilibrium along the air x/z axis
        # and moment equilibrium performed with found_cl_repartition sub-function. The moment
        # generated by (x_cg_aircraft - x_cg_engine) * T * sin(alpha - alpha_eng) is neglected!

        if low_speed:
            coeff_k_wing = inputs["data:aerodynamics:wing:low_speed:induced_drag_coefficient"]
            coeff_k_htp = inputs[
                "data:aerodynamics:horizontal_tail:low_speed:induced_drag_coefficient"
            ]
            cl_alpha_wing = inputs["data:aerodynamics:wing:low_speed:CL_alpha"]
            cl0_wing = inputs["data:aerodynamics:wing:low_speed:CL0_clean"]
            cd0 = inputs["data:aerodynamics:aircraft:low_speed:CD0"]
            cl0_htp = inputs["data:aerodynamics:horizontal_tail:low_speed:CL0"]
            cl_alpha_htp = inputs["data:aerodynamics:horizontal_tail:low_speed:CL_alpha"]
        else:
            coeff_k_wing = inputs["data:aerodynamics:wing:cruise:induced_drag_coefficient"]
            coeff_k_htp = inputs[
                "data:aerodynamics:horizontal_tail:cruise:induced_drag_coefficient"
            ]
            cl_alpha_wing = inputs["data:aerodynamics:wing:cruise:CL_alpha"]
            cl0_wing = inputs["data:aerodynamics:wing:cruise:CL0_clean"]
            cd0 = inputs["data:aerodynamics:aircraft:cruise:CD0"]
            cl0_htp = inputs["data:aerodynamics:horizontal_tail:cruise:CL0"]
            cl_alpha_htp = inputs["data:aerodynamics:horizontal_tail:cruise:CL_alpha"]

        cl_elevator_delta = inputs["data:aerodynamics:elevator:low_speed:CL_delta"]
        cd_elevator_delta = inputs["data:aerodynamics:elevator:low_speed:CD_delta"]
        cl_max_clean = inputs["data:aerodynamics:wing:low_speed:CL_max_clean"]
        z_cg_aircraft = inputs["data:weight:aircraft_empty:CG:z"]
        z_cg_engine = inputs["data:weight:propulsion:engine:CG:z"]
        wing_mac = inputs["data:geometry:wing:MAC:length"]
        wing_area = inputs["data:geometry:wing:area"]
        z_eng = z_cg_aircraft - z_cg_engine
        alpha_eng = 0.0  # fixme: angle between propulsion and wing not defined

        alpha = x[0] * math.pi / 180.0  # defined in degree to be homogenous on x-tolerance
        thrust = x[1] * 1000.0  # defined in kN to be homogenous on x-tolerance

        load_factor = (
            -dvz_dt + g * math.cos(gamma) - thrust / mass * math.sin(alpha - alpha_eng)
        ) / g
        # Additional aerodynamics
        delta_cl = 0.0
        delta_cm = z_eng * thrust * math.cos(alpha - alpha_eng) / (wing_mac * q * wing_area)
        cl_wing_blown, cl_htp, error_tag = self.found_cl_repartition(
            inputs, load_factor, mass, q, delta_cm, low_speed, x_cg
        )

        self.error_on_pitch_equilibrium = error_tag

        if low_speed:
            if flap_condition == "takeoff":
                cl_wing = inputs["data:aerodynamics:flaps:takeoff:CL"]
                cd0_flaps = inputs["data:aerodynamics:flaps:takeoff:CD"]
            elif flap_condition == "landing":
                cl_wing = inputs["data:aerodynamics:flaps:landing:CL"]
                cd0_flaps = inputs["data:aerodynamics:flaps:landing:CD"]
            else:
                cl_wing = 0.0
                cd0_flaps = 0.0
        else:
            cl_wing = 0.0
            cd0_flaps = 0.0
        cl_wing += cl0_wing + cl_alpha_wing * alpha
        # TODO : Change formula for trimmable htp
        # Calculate the elevator command if htp not trimmed
        delta_e = (cl_htp - (alpha * cl_alpha_htp + cl0_htp)) / cl_elevator_delta
        cd = (
            cd0
            + cd0_flaps
            + coeff_k_wing * cl_wing ** 2
            + coeff_k_htp * cl_htp ** 2
            + (cd_elevator_delta * delta_e ** 2.0)
        )
        drag = q * cd * wing_area
        # Divide the results by characteristic number to have homogeneous responses
        f1 = float(
            thrust * math.cos(alpha - alpha_eng) - mass * g * math.sin(gamma) - drag - mass * dvx_dt
        ) / (float(mass) / 10.0)
        f2 = float(cl_wing_blown - (cl_wing + delta_cl)) / float(cl_max_clean)

        self.cl_wing_sol = cl_wing_blown
        self.cl_tail_sol = cl_htp
        self.delta_e_sol = delta_e

        return np.array([f1, f2])


def save_df(
    time,
    altitude,
    distance,
    mass,
    v_tas,
    v_cas,
    rho,
    gamma,
    equilibrium_result,
    thrust_rate,
    sfc,
    name: str,
    existing_dataframe: pd.DataFrame = None,
):
    """
    Method to save mission point to a pandas dataframe file for further post-processing

    :param time: mission time in seconds
    :param altitude: flight altitude in meters
    :param distance: flight distance in meters
    :param mass: aircraft current mass
    :param v_tas: true air speed in m/s
    :param v_cas: calibrated air speed in m/s
    :param rho: air density in kg/m3
    :param gamma: slope angle in degree
    :param equilibrium_result: result vector of dynamic equilibrium
    :param thrust_rate: thrust rate at flight point
    :param sfc: sfc at flight point
    :param name: phase name
    :param existing_dataframe: Dataframe
    """

    alpha = float(equilibrium_result[0]) * 180.0 / math.pi
    thrust = float(equilibrium_result[1])
    cl_wing = float(equilibrium_result[2])
    cl_htp = float(equilibrium_result[3])
    atm = Atmosphere(altitude, altitude_in_feet=False)
    mach = v_tas / atm.speed_of_sound
    if existing_dataframe is None:
        existing_dataframe = pd.DataFrame(columns=CSV_DATA_LABELS)
        existing_dataframe.loc[0] = [
            float(time),
            float(altitude),
            float(distance),
            float(mass),
            float(v_tas),
            float(v_cas),
            float(mach),
            float(rho),
            float(gamma),
            alpha,
            cl_wing,
            cl_htp,
            thrust,
            float(thrust_rate),
            float(sfc),
            name,
        ]
    else:
        data = [
            float(time),
            float(altitude),
            float(distance),
            float(mass),
            float(v_tas),
            float(v_cas),
            float(mach),
            float(rho),
            float(gamma),
            alpha,
            cl_wing,
            cl_htp,
            thrust,
            float(thrust_rate),
            float(sfc),
            name,
        ]
        row = pd.DataFrame(data, index=CSV_DATA_LABELS).transpose()
        existing_dataframe = pd.concat([existing_dataframe, row])

    return existing_dataframe
