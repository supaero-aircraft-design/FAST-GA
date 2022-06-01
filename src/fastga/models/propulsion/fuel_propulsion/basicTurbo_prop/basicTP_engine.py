"""Parametric turboprop engine."""
# -*- coding: utf-8 -*-
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

import os.path as pth
import logging
import math
from typing import Union, Sequence, Tuple, Optional
from scipy.interpolate import interp2d, interp1d
from scipy.optimize import fsolve
from pandas import read_csv
import numpy as np

import fastoad.api as oad
from fastoad.constants import EngineSetting
from fastoad.exceptions import FastUnknownEngineSettingError
from stdatm import Atmosphere

from fastga.models.propulsion.fuel_propulsion.basicTurbo_prop import resources
from fastga.models.propulsion.fuel_propulsion.basicTurbo_prop.exceptions import (
    FastBasicICEngineInconsistentInputParametersError,
    FastBasicTPEngineImpossibleTurbopropGeometry,
    FastBasicTPEngineUnknownLimit,
)
from fastga.models.propulsion.fuel_propulsion.base import AbstractFuelPropulsion
from fastga.models.propulsion.dict import DynamicAttributeDict, AddKeyAttributes

# Logger for this module
_LOGGER = logging.getLogger(__name__)

# Set of dictionary keys that are mapped to instance attributes.
ENGINE_LABELS = {
    "power_SL": dict(doc="power at sea level in watts."),
    "mass": dict(doc="Mass in kilograms."),
    "length": dict(doc="Length in meters."),
    "height": dict(doc="Height in meters."),
    "width": dict(doc="Width in meters."),
}
# Set of dictionary keys that are mapped to instance attributes.
NACELLE_LABELS = {
    "wet_area": dict(doc="Wet area in meters²."),
    "length": dict(doc="Length in meters."),
    "height": dict(doc="Height in meters."),
    "width": dict(doc="Width in meters."),
}


class BasicTPEngine(AbstractFuelPropulsion):
    def __init__(
        self,
        power_design: float,  # In kW
        t_41t_design: float,
        opr_design: float,
        cruise_altitude_propeller: float,
        design_altitude: float,
        design_mach: float,
        prop_layout: float,
        bleed_control: float,
        itt_limit: float,
        power_limit: float,
        opr_limit: float,
        speed_SL,
        thrust_SL,
        thrust_limit_SL,
        efficiency_SL,
        speed_CL,
        thrust_CL,
        thrust_limit_CL,
        efficiency_CL,
        effective_J,
        effective_efficiency_ls,
        effective_efficiency_cruise,
        eta_225=0.85,
        eta_253=0.86,
        eta_445=0.86,
        eta_455=0.86,
        eta_q=43.260e6 * 0.95,
        eta_axe=0.98,
        pi_02=0.8,
        pi_cc=0.95,
        cooling_ratio=0.05,
        hp_shaft_power_out=50 * 745.7,
        gearbox_efficiency=0.98,
        inter_compressor_bleed=0.04,
        exhaust_mach_design=0.4,
        pr_1_ratio_design=0.25,
    ):
        """
        Parametric turboprop engine.

        It computes engine characteristics using a simplified thermodynamic model, according to
        the model established byt Aitor Busteros Ramos.

        :param power_design: thermodynamic power at the design point, in kW
        :param t_41t_design: turbine entry temperature at the design point, in K
        :param opr_design: overall pressure ratio at the design point
        :param cruise_altitude_propeller: cruise altitude, in m
        :param design_altitude: altitude at the design point, in m
        :param design_mach: mach number at the design point
        :param prop_layout: location of the turboprop
        :param bleed_control: usage of the bleed in off-design point, "low" or "high"
        :param itt_limit: temperature limit between the turbines, in K
        :param power_limit: power limit on the gearbox, in kW
        :param opr_limit: opr limit in the compressor
        :param speed_SL: array with the speed at which the sea level performance of the propeller
        were computed
        :param thrust_SL: array with the required thrust at which the sea level performance of the
        propeller were
        computed
        :param thrust_limit_SL: array with the limit thrust available at the speed in speed_SL
        :param efficiency_SL: array containing the sea level efficiency computed at speed_SL and
        thrust_SL
        :param speed_CL: array with the speed at which the cruise level performance of the propeller
        were computed
        :param thrust_CL: array with the required thrust at which the cruise level performance of
        the propeller were
        computed
        :param thrust_limit_CL: array with the limit thrust available at the speed in speed_CL
        :param efficiency_CL: array containing the cruise level efficiency computed at speed_CL and
        thrust_CL
        :param eta_225: first compressor stage polytropic efficiency
        :param eta_253: second compressor stage polytropic efficiency
        :param eta_445: high pressure turbine  polytropic efficiency
        :param eta_455: power turbine  polytropic efficiency
        :param eta_q: combustion efficiency, in J/kg
        :param eta_axe: high pressure axe mechanical efficiency
        :param pi_02: inlet total pressure loss
        :param pi_cc: combustion chamber pressure loss
        :param cooling_ratio: percentage of the total aspirated airflow used for turbine cooling
        :param hp_shaft_power_out: power used for electrical generation obtained from the HP shaft,
        in W
        :param gearbox_efficiency: power shaft mechanical efficiency
        :param inter_compressor_bleed: total compressor airflow extracted after the first
        compression stage (in station 25)
        :param exhaust_mach_design: mach number at the exhaust in the design point
        :param pr_1_ratio_design: ratio of the first stage pressure ration to the OPR at the design
        point.
        """

        # Load the value of the air properties graph
        cv_c, cp_c = self.air_coefficients_reader()

        self.cv_c = cv_c
        self.cp_c = cp_c

        # Load the value of the cabin pressure
        self.cabin_pressure = self.air_renewal_coefficients()

        # Definition of the Turboprop design parameters
        self.eta_225 = eta_225  # First compressor stage polytropic efficiency
        self.eta_253 = eta_253  # Second compressor stage polytropic efficiency
        self.eta_445 = eta_445  # High pressure turbine  polytropic efficiency
        self.eta_455 = eta_455  # power turbine  polytropic efficiency
        self.eta_q = eta_q  # Combustion efficiency [J/kg]
        self.eta_axe = eta_axe  # HP axe mechanical efficiency
        self.pi_02 = pi_02  # Inlet pressure loss
        self.pi_cc = pi_cc  # Combustion chamber pressure loss
        self.cooling_ratio = (
            cooling_ratio  # Percentage of the total aspirated airflow used for turbine cooling
        )
        self.hp_shaft_power_out = (
            hp_shaft_power_out  # power used for electrical generation obtained from the HP shaft
        )
        # (in Watts)
        self.gearbox_efficiency = gearbox_efficiency  # power shaft mechanical efficiency
        self.inter_compressor_bleed = (
            inter_compressor_bleed  # Total compressor airflow extracted after the first
        )
        # compression stage (in station 25)
        self.exhaust_mach_design = (
            exhaust_mach_design  # Mach of the exhaust gases in the design point
        )
        self.opr_1_design = (
            pr_1_ratio_design * np.array(opr_design).item()
        )  # Compression ratio of the first stage in the design
        # point
        self.bleed_control_design = (
            "high"  # Switch between "high" or "low" for the bleed in the design point
        )
        self.bleed_control = np.array(bleed_control).item()

        # Definition of the Turboprop design parameters
        self.design_point_power = np.array(power_design).item()
        self.t_41t_d = np.array(t_41t_design).item()
        self.opr_d = np.array(opr_design).item()

        # Definition of the Turboprop limits
        self.itt_limit = itt_limit  # In Kelvin
        self.max_power_avail = (
            power_limit  # Max power the engine can give, with al the active constraints in the
        )
        # design point
        self.opr_limit = opr_limit  # To avoid compressor instabilities

        self.prop_layout = prop_layout
        self.cruise_altitude_propeller = np.array(cruise_altitude_propeller).item()
        self.design_point_altitude = np.array(design_altitude).item()
        self.design_point_mach = np.array(design_mach).item()
        self.design_mach = np.array(design_mach).item()
        self.fuel_type = 3.0  # Turboprops only use JetFuel
        self.idle_thrust_rate = 0.01
        self.speed_SL = speed_SL
        self.thrust_SL = thrust_SL
        self.thrust_limit_SL = thrust_limit_SL
        self.efficiency_SL = efficiency_SL
        self.speed_CL = speed_CL
        self.thrust_CL = thrust_CL
        self.thrust_limit_CL = thrust_limit_CL
        self.efficiency_CL = efficiency_CL
        self.effective_J = float(effective_J)
        self.effective_efficiency_ls = float(effective_efficiency_ls)
        self.effective_efficiency_cruise = float(effective_efficiency_cruise)
        self.specific_shape = None

        # Declare sub-components attribute
        self.engine = Engine(power_SL=power_design)
        self.engine.mass = None
        self.engine.length = None
        self.engine.width = None
        self.engine.height = None

        self.nacelle = Nacelle()
        self.nacelle.wet_area = None

        self.propeller = None

        # This dictionary is expected to have a Mixture coefficient for all EngineSetting values
        self.mixture_values = {
            EngineSetting.TAKEOFF: 1.15,
            EngineSetting.CLIMB: 1.15,
            EngineSetting.CRUISE: 1.0,
            EngineSetting.IDLE: 1.0,
        }

        # ... so check that all EngineSetting values are in dict
        unknown_keys = [key for key in EngineSetting if key not in self.mixture_values.keys()]
        if unknown_keys:
            raise FastUnknownEngineSettingError("Unknown flight phases: %s", unknown_keys)

        # Computation of the turboprop geometry based on the design point performance
        (
            alfa,
            alfa_p,
            a_41,
            a_45,
            a_8,
            eta_compress,
            m_c,
            t_4t,
            _,
            t_45t,
            opr_2_opr_1,
        ) = self.turboprop_geometry_calculation()

        # Storing the propeller geometry and constant parameter
        self.alfa = alfa
        self.alfa_p = alfa_p
        self.a_41 = a_41
        self.a_45 = a_45
        self.a_8 = a_8
        self.eta_compress_design = eta_compress
        self.m_c_dp = m_c
        self.t_4t_dp = t_4t
        self.t_45t_dp = t_45t
        self.opr_2_opr_1_dp = (
            opr_2_opr_1  # Compression ratio relationship between the second and first stages
        )

        # Here some internal attributes are defined.

        self.t_4t_int = 0.0
        self.t_41t_int = 0.0
        self.t_45t_int = 0.0
        self.p_41t_int = 0.0
        self.opr_int = 0.0
        self.mach_8_int = 0.0
        self.g_int = 0.0
        self.f_fuel_ratio_int = 0.0
        self.m_int = 0.0

        # Here is where we we store the results of our computations, they are overwritten after
        # each computation
        self.t_45t_sol = 0.0
        self.opr_sol = 0.0
        self.power_sol = [0.0]
        self.thrust_sol = [0.0]

    @staticmethod
    def air_coefficients_reader():

        """
        This function reads  table with a et of temperatures, Cv and Cp values. It creates two
        polynomial interpolation functions, whose coefficients are returned [one for Cv = f(T)
        and another for Cp = f(T)].
        """

        file = pth.join(resources.__path__[0], "T_Cv_Cp.csv")
        database = read_csv(file)

        temp = database["T"]
        cv_n = database["CV"]
        cp_n = database["CP"]

        cv_t_coefficients = np.polyfit(temp, cv_n, 15)
        cp_t_coefficients = np.polyfit(temp, cp_n, 15)

        return cv_t_coefficients, cp_t_coefficients

    def compute_cp_cv_gamma(self, temperature):
        """
        Obtains the Cv and Cp values for a given Temperature

        It evaluates the polynomial interpolation functions for Cp and Cv, shortening code:

        :param temperature: the actual Temperature in Kelvin [K]

        :return cp_out: The actual Cp value for the given Temperature
        :return cv_out: The actual Cv value for the given Temperature
        :return gamma: The actual gamma value for the given Temperature
        """

        cp_coefficient = self.cp_c
        cv_coefficient = self.cv_c

        cp_out = np.polyval(cp_coefficient, temperature)
        cv_out = np.polyval(cv_coefficient, temperature)
        gamma = cp_out / cv_out
        return cp_out, cv_out, gamma

    @staticmethod
    def compute_gamma_functions(gamma):
        """
        Computes the three gamma functions for each gamma value

        :param gamma: heat capacity ratio

        :return f1: equal to gamma / (gamma - 1)
        :return f2: equal to (gamma - 1) / gamma
        :return f_gamma: equal to
        np.sqrt(gamma) * (2 / (gamma + 1)) ** ((gamma + 1) / (2 * (gamma - 1)))
        """
        f_1 = gamma / (gamma - 1)
        f_2 = 1 / f_1
        f_gamma = np.sqrt(gamma) * (2 / (gamma + 1)) ** ((gamma + 1) / (2 * (gamma - 1)))

        return f_1, f_2, f_gamma

    @staticmethod
    def air_renewal_coefficients():
        """
        This function reads table with a set of flight altitude and cabin altitude. It creates an
        interpolation function.

        :return cabin_pressure: interp1d function linking flight altitude to cabin altitude.
        """

        file = pth.join(resources.__path__[0], "cabin_pressurisation.csv")
        database = read_csv(file)

        h_flight = database["FLIGHT_ALTITUDE"]
        h_cab = database["CABIN_ALTITUDE"]

        cabin_pressure = interp1d(h_flight, h_cab)

        return cabin_pressure

    def air_renewal(self, altitude, bleed_control):
        """
        Computes the airflow used for cabin air renewal

        :param altitude: The flight altitude in meters [m]
        :param bleed_control: The air packs setting "high" or "low"

        :return m_air: The cabin airflow in [kg/s]
        """
        cabin_volume = 5.0  # in m3

        if bleed_control == "low":
            control = 0.3
        else:
            control = 1.0

        renovation_time = 2.0  # in minutes

        h_cab = self.cabin_pressure(altitude)

        t_cab = 20.0 + 273.0
        atmosphere = Atmosphere(h_cab, altitude_in_feet=False)
        p_cab = atmosphere.pressure

        rho_cab = p_cab / 287.0 / t_cab
        m_air = cabin_volume * rho_cab / (renovation_time * 60.0) * control
        return m_air

    def point_design_solver(
        self,
        var_to_solve,
        t_41t,
        p_0,
        t_2t,
        t_25t,
        t_3t,
        p_3t,
        power,
        exhaust_mach,
        bleed_control,
        h_0,
    ):
        """
        Solver for the design point. Finds the engine properties for which the thermodynamic
        equations leads to the design point target performances

        :param var_to_solve: an array containing the values used to solve the system of
        thermodynamic equation, contains :
        m_c : the mass flow of fuel,
        t_4t : the turbine entry temperature,
        t_45t : the inter-turbine temperature,
        p_45t : the inter-turbine pressure,
        p_5t : the pressure after the turbines,
        m0 : the airflow in the usual units.
        :param t_41t: the temperature after the mixing with the cold air, in K
        :param p_0: the atmospheric pressure, in Pa
        :param t_2t: the total temperature before the compressors, in K
        :param t_25t: the total temperature between the compressors, in K
        :param t_3t: the total temperature after the compressors, in K
        :param p_3t: the atmospheric pressure, in Pa
        :param power: the thermodynamic power at the design point, in kW
        :param exhaust_mach: mach number at the exhaust
        :param bleed_control: setting of the bleed at the design point, either "high" or "low"
        :param h_0: the design point altitude, in m

        :return f: an array containing the application of the thermodynamic equation written as
        differences to set to 0
        """
        m_c = var_to_solve[0]
        t_4t = var_to_solve[1]
        t_45t = var_to_solve[2]
        t_5t = var_to_solve[3]
        p_45t = var_to_solve[4]
        p_5t = var_to_solve[5]
        m_0 = var_to_solve[6]

        g_r = self.air_renewal(h_0, bleed_control) / m_0

        p4t = p_3t * self.pi_cc

        cp_2, _, _ = self.compute_cp_cv_gamma(t_2t)
        cp_25, _, _ = self.compute_cp_cv_gamma(t_25t)
        cp_3, _, _ = self.compute_cp_cv_gamma(t_3t)
        cp_4, _, _ = self.compute_cp_cv_gamma(t_4t)
        cp_41, _, gamma41 = self.compute_cp_cv_gamma(t_41t)
        f1_41, _, _ = self.compute_gamma_functions(gamma41)
        cp_45, _, gamma45 = self.compute_cp_cv_gamma(t_45t)
        _, f2_45, _ = self.compute_gamma_functions(gamma45)
        cp_5, _, gamma5 = self.compute_cp_cv_gamma(t_5t)
        f1_5, _, _ = self.compute_gamma_functions(gamma5)

        fuel_air_ratio = m_c / m_0
        icb = self.inter_compressor_bleed / m_0

        return_array = np.zeros(7)
        # Temperature change after through the combustion chamber
        return_array[0] = (cp_4 * t_4t - cp_3 * t_3t) * (
            1 + fuel_air_ratio - g_r - self.cooling_ratio - icb
        ) - self.eta_q * fuel_air_ratio
        # Mixing of hot air from the compressor with the hot gases from the combustion chamber
        return_array[1] = t_41t - (
            (
                t_4t * (1 + fuel_air_ratio - g_r - self.cooling_ratio - icb)
                + t_3t * self.cooling_ratio
            )
            / (1 + fuel_air_ratio - g_r - icb)
        )
        # Mechanic equilibrium on the high pressure axis
        return_array[2] = (
            (1 + fuel_air_ratio - g_r - icb) * (cp_41 * t_41t - cp_45 * t_45t) * self.eta_axe
            - self.hp_shaft_power_out / m_0
            - (cp_3 * t_3t - cp_25 * t_25t) * (1 - icb)
            - (cp_25 * t_25t - cp_2 * t_2t)
        )
        # Expansion in the high pressure turbine
        return_array[3] = p_45t - (p4t * (t_45t / t_41t) ** (f1_41 / self.eta_445))
        # Power given to the propeller
        return_array[4] = (
            m_0 * 1000.0
            - (
                ((power * 1000.0 / self.gearbox_efficiency) / (cp_45 * t_45t - cp_5 * t_5t))
                / (1 - g_r + fuel_air_ratio - icb)
            )
            * 1000.0
        )
        # Expansion in the power turbine
        return_array[5] = t_5t - t_45t * ((p_5t / p_45t) ** (f2_45 * self.eta_455))
        return_array[6] = p_5t - (p_0 * (1 + (gamma5 - 1) / 2 * exhaust_mach ** 2) ** f1_5)

        return return_array

    def turboprop_geometry_calculation(self):

        """
        This method is the core of the Turboprop Class constructor. It obtains the geometry of
        the turboprop: the turbine and exhaust sections A41, A45 and A8 as well as the alfa
        parameter. These values are returned to the constructor and stored as attributes as these
        four values WILL BE CONSTANT FOR ALL THE TURBOPROP OPERATION REGIMES (for further info,
        read the documentation of the turboprop model). Other, non-constant values,
        such us temperatures or compression ratios are also obtained.
        """
        r_g = 287.0

        design_point_mach = self.design_point_mach
        h_0 = self.design_point_altitude
        power = self.design_point_power
        global_opr = self.opr_d
        t_41t = self.t_41t_d
        exhaust_mach = self.exhaust_mach_design
        bleed_control = self.bleed_control_design
        cab_bleed = self.air_renewal(h_0, bleed_control)
        opr_1 = self.opr_1_design
        opr_2 = global_opr / self.opr_1_design

        # Computing air properties at the entry of the turboprop
        atmosphere_0 = Atmosphere(h_0, altitude_in_feet=False)
        p_0 = atmosphere_0.pressure
        t_0 = atmosphere_0.temperature

        p_0t = p_0 * (1 + (1.4 - 1) / 2 * design_point_mach ** 2) ** 3.5
        t_0t = t_0 * (1 + (1.4 - 1) / 2 * design_point_mach ** 2)

        # Entry of the compressor
        p_2t = p_0t * self.pi_02
        t_2t = t_0t

        cp_2, _, gamma2 = self.compute_cp_cv_gamma(t_2t)
        f1_2, f2_2, _ = self.compute_gamma_functions(gamma2)

        # Inter compressor stage
        t_25t = t_2t * opr_1 ** (f2_2 / self.eta_225)
        p_25t = p_2t * opr_1
        p_3t = p_25t * opr_2

        cp_25, _, gamma25 = self.compute_cp_cv_gamma(t_25t)
        _, f2_25, _ = self.compute_gamma_functions(gamma25)

        # After the compressor
        t_3t = t_25t * opr_2 ** (f2_25 / self.eta_253)
        eta_compress = math.log(global_opr) / math.log(t_3t / t_2t) * f2_2

        # Solving the gas generator equations
        p4t = p_3t * self.pi_cc
        p_41t = p4t

        initial_values = np.array([0.06, 1350.0, 1000.0, 800.0, 400000.0, 110000.0, 3.5])
        # z = np.zeros(len(X0))
        [solution_vector, _, ier, _] = fsolve(
            self.point_design_solver,
            initial_values,
            (
                t_41t,
                p_0,
                t_2t,
                t_25t,
                t_3t,
                p_3t,
                power,
                exhaust_mach,
                bleed_control,
                h_0,
            ),
            xtol=1e-4,
            full_output=True,
        )

        if ier != 1:
            raise FastBasicTPEngineImpossibleTurbopropGeometry(
                "Solver returned wrong results while constructing the Turboprop geometry, "
                "check the input parameters "
            )

        m_c = solution_vector[0]
        t_4t = solution_vector[1]
        t_45t = solution_vector[2]
        t_5t = solution_vector[3]
        p_45t = solution_vector[4]
        p_5t = solution_vector[5]
        airflow_design = solution_vector[6]

        fuel_air_ratio = m_c / airflow_design
        g_r = cab_bleed / airflow_design
        icb = self.inter_compressor_bleed / airflow_design

        cp_3, _, _ = self.compute_cp_cv_gamma(t_3t)
        cp_41, _, gamma41 = self.compute_cp_cv_gamma(t_41t)
        _, _, f_gamma_41 = self.compute_gamma_functions(gamma41)
        cp_45, _, gamma45 = self.compute_cp_cv_gamma(t_45t)
        _, _, f_gamma_45 = self.compute_gamma_functions(gamma45)
        cp_5, _, gamma5 = self.compute_cp_cv_gamma(t_5t)

        alfa = t_45t / t_41t
        alfa_p = p_45t / p_41t

        # Computing the turboprop sections
        a_41 = (
            airflow_design
            * (1 + fuel_air_ratio - g_r - icb)
            * np.sqrt(t_41t * r_g)
            / p4t
            / f_gamma_41
        )
        a_45 = (
            airflow_design
            * (1 + fuel_air_ratio - g_r - icb)
            * np.sqrt(t_45t * r_g)
            / p_45t
            / f_gamma_45
        )
        a_8_1 = airflow_design * (1 + fuel_air_ratio - g_r - icb) * np.sqrt(t_5t * r_g) / p_5t
        a_8_2 = (
            np.sqrt(gamma5)
            * exhaust_mach
            * (1 + (gamma5 - 1) / 2 * exhaust_mach ** 2) ** ((gamma5 + 1) / (2 * (1 - gamma5)))
        )
        a_8 = a_8_1 / a_8_2

        # Verification that the solution found by the solver give plausible results on the whole
        # turboprop
        opr_check = (
            cp_2 / cp_3 / (1 - icb)
            + self.eta_axe
            * (1 + fuel_air_ratio - g_r - icb)
            / (1 - icb)
            * (cp_41 - cp_45 * alfa)
            / cp_3
            * t_41t
            / t_2t
            - self.hp_shaft_power_out / (cp_3 * airflow_design * (1 - icb) * t_2t)
            - t_25t / t_2t * cp_25 / cp_3 * (1 / (1 - icb) - 1)
        ) ** (f1_2 * eta_compress)

        power_check = (
            (cp_45 * t_45t - cp_5 * t_5t)
            * airflow_design
            / 1000.0
            * (1 - g_r + fuel_air_ratio - icb)
            * self.gearbox_efficiency
        )

        if abs(opr_check - global_opr) / global_opr > 1e-3:
            raise FastBasicTPEngineImpossibleTurbopropGeometry(
                "Overall Pressure Ratio check failed while constructing the Turboprop geometry, "
                "check the input parameters "
            )
        if abs(power_check - power) / power > 1e-3:
            raise FastBasicTPEngineImpossibleTurbopropGeometry(
                "Power check failed while constructing the Turboprop geometry, check the input "
                "parameters "
            )

        return alfa, alfa_p, a_41, a_45, a_8, eta_compress, m_c, t_4t, t_41t, t_45t, opr_2 / opr_1

    def turboshaft_performance_solver_real_gas(self, var_to_solve, m_c, p_2t, t_2t, m_air):

        """

        Solver for an off-design point. Finds the engine properties for which the thermodynamic
        equations leads to the required performances

        :param var_to_solve: an array containing the values used to solve the system of
        thermodynamic equation, contains:
        t_25t : the temperature between the axial and radial compressor, in K,
        t_3t : the temperature after the radial compressor, in K,
        t_41t : the temperature after the mixing of cold air, in K,
        m : the total airflow, in kg/s,
        p_3t : the total pressure after the radial compressor in Pa
        :param m_c: the fuel mass flow, in kg/s
        :param p_2t: the total pressure before the compressor stage, in Pa
        :param t_2t: the total temperature before the compressor stage, in K
        :param m_air: the bleed air mass flow, in kg/s

        :return f: an array containing the application of the thermodynamic equation written as
        differences to set to 0

        """

        t_25t = var_to_solve[0]
        t_3t = var_to_solve[1]
        t_41t = var_to_solve[2]
        air_mass_flow = var_to_solve[3]
        p_3t = var_to_solve[4]
        t_45t = t_41t * self.alfa

        self.t_41t_int = t_41t
        self.m_int = air_mass_flow
        self.t_45t_int = t_45t

        r_g = 287.0
        g_r = m_air / air_mass_flow

        self.g_int = g_r

        f_fuel_ratio = m_c / air_mass_flow
        icb = self.inter_compressor_bleed / air_mass_flow

        self.f_fuel_ratio_int = f_fuel_ratio

        cp_2, _, gamma2 = self.compute_cp_cv_gamma(t_2t)
        f1_2, _, _ = self.compute_gamma_functions(gamma2)
        cp_25, _, gamma25 = self.compute_cp_cv_gamma(t_25t)
        f1_25, _, _ = self.compute_gamma_functions(gamma25)
        cp_3, _, _ = self.compute_cp_cv_gamma(t_3t)
        cp_41, _, gamma41 = self.compute_cp_cv_gamma(t_41t)
        _, _, f_gamma_41 = self.compute_gamma_functions(gamma41)
        cp_45, _, _ = self.compute_cp_cv_gamma(t_45t)

        p_25t = p_2t * (t_25t / t_2t) ** (f1_2 * self.eta_225)
        opr_2 = p_3t / p_25t
        opr_1 = p_25t / p_2t
        opr = opr_1 * opr_2

        t_4t = (t_41t * (1 + f_fuel_ratio - g_r - icb) - t_3t * self.cooling_ratio) / (
            1 + f_fuel_ratio - g_r - self.cooling_ratio - icb
        )

        cp_4, _, _ = self.compute_cp_cv_gamma(t_4t)

        p_41t = p_3t * self.pi_cc

        return_array = np.zeros(5)
        # Temperature change through the combustion chamber
        return_array[0] = 1.0 - air_mass_flow * (
            1 + f_fuel_ratio - g_r - self.cooling_ratio - icb
        ) * (cp_4 * t_4t - cp_3 * t_3t) / (m_c * self.eta_q)
        return_array[1] = (
            1.0
            - air_mass_flow
            * (1 + f_fuel_ratio - g_r - icb)
            * np.sqrt(t_41t * r_g)
            / p_41t
            / f_gamma_41
            / self.a_41
        )
        # Pressure change through the compressors
        return_array[2] = 1.0 - p_25t / p_3t * (t_3t / t_25t) ** (f1_25 * self.eta_253)
        return_array[3] = 1.0 - self.opr_2_opr_1_dp / (opr_2 / opr_1)
        # Temperature change through the compressor
        return_array[4] = (
            1.0
            - (
                t_2t
                * (
                    cp_2 / cp_3 / (1 - icb)
                    + self.eta_axe
                    * (1 + f_fuel_ratio - g_r - icb)
                    / (1 - icb)
                    * (cp_41 - cp_45 * self.alfa)
                    / cp_3
                    * t_41t
                    / t_2t
                    - self.hp_shaft_power_out / (cp_3 * air_mass_flow * (1 - icb) * t_2t)
                    - t_25t / t_2t * cp_25 / cp_3 * (1 / (1 - icb) - 1)
                )
            )
            / t_3t
        )

        self.t_4t_int = t_4t
        self.t_41t_int = t_41t
        self.t_45t_int = t_45t
        self.p_41t_int = p_41t
        self.opr_int = opr

        return return_array

    def exhaust_mach_solver_real_gas(self, t_5t, t_45t, p_45t, p_0):

        """

        Solver for the off-design point exhaust thrust. Finds the total temperature after the
        power turbine that gives an adapted nozzle

        :param t_5t: the total temperature after the power turbine, in K
        :param t_45t: the total temperature between the turbines, in K
        :param p_45t: the total pressure between the turbines, in Pa
        :param p_0: the atmospheric pressure, in Pa

        :return function_minimize: the expansion equation of the power turbine written as a
        difference to equate to 0

        """

        _, _, gamma5 = self.compute_cp_cv_gamma(t_5t)
        f1_5, _, f_gamma_5 = self.compute_gamma_functions(gamma5)
        _, _, gamma45 = self.compute_cp_cv_gamma(t_45t)
        _, f2_45, _ = self.compute_gamma_functions(gamma45)

        mach_8 = (
            f_gamma_5
            * self.a_45
            / np.sqrt(gamma5)
            / self.a_8
            * (p_45t / p_0) ** ((gamma5 + 1) / 2 / gamma5)
        )

        p_5t = p_0 * (1 + (gamma5 - 1) / 2 * mach_8 ** 2) ** f1_5

        # Temperature change in the power turbine
        function_minimize = 1.0 - t_45t / t_5t * (p_5t / p_45t) ** (f2_45 * self.eta_455)

        self.mach_8_int = mach_8

        return function_minimize

    def turboshaft_performance_real_gas(self, altitude, flight_mach, m_c):

        """
        Computes the characteristics of the engine when a certain fuel flow is injected into the
        turboprop in certain flight conditions.

        :param altitude: the flight altitude, in m
        :param flight_mach: the flight mach
        :param m_c: the fuel flow injected in the turboprop, in kg/s

        :return t_4t: the turbine entry temperature, in K
        :return m: the air mass flow, in kg/s
        :return power: the power output of the engine, in kW
        :return f_fuel_ratio: the ratio between the fuel mass flow and the air mass flow
        :return p_41t: the pressure after the mixing of cold air, in Pa
        :return opr: the overall pressure ratio
        :return t_41t: the temperature after the mixing of cold air, in K
        :return t_45t: the temperature between the turbines, in K
        :return m: the air flow, in m/s.
        """

        performance_atmosphere = Atmosphere(altitude, altitude_in_feet=False)

        p_0 = performance_atmosphere.pressure
        t_0 = performance_atmosphere.temperature
        if self.bleed_control == 1.0:
            bleed_control = "high"
        else:
            bleed_control = "low"
        m_air = self.air_renewal(altitude, bleed_control)
        r_g = 287.0

        # Computing atmospheric conditions
        p_0t = p_0 * (1 + (1.4 - 1) / 2 * flight_mach ** 2) ** 3.5
        t_0t = t_0 * (1 + (1.4 - 1) / 2 * flight_mach ** 2)

        # Entry of the compressor
        p_2t = p_0t * self.pi_02
        t_2t = t_0t

        # Solving the gas generator equation
        initial_values = np.array([300, 500, 1200, 3, 400000])
        fsolve(
            self.turboshaft_performance_solver_real_gas,
            initial_values,
            (m_c, p_2t, t_2t, m_air),
            xtol=1e-8,
        )

        t_4t = self.t_4t_int
        t_41t = self.t_41t_int
        t_45t = self.t_45t_int
        p_41t = self.p_41t_int
        opr = self.opr_int
        g_r = self.g_int
        f_fuel_ratio = self.f_fuel_ratio_int
        air_mass_flow = self.m_int

        p_45t = p_41t * self.alfa_p
        icb = self.inter_compressor_bleed / air_mass_flow

        # Solving the exhaust equation
        t_5t_0 = np.array([[900]])
        z_minimize = np.array(
            fsolve(self.exhaust_mach_solver_real_gas, t_5t_0, (t_45t, p_45t, p_0))
        )
        t_5t = z_minimize

        mach_8 = self.mach_8_int

        cp_45, _, _ = self.compute_cp_cv_gamma(t_45t)
        cp_5, _, gamma5 = self.compute_cp_cv_gamma(t_5t)

        # Computing the shaft power output
        power = (
            air_mass_flow
            * (1 - g_r + f_fuel_ratio - icb)
            * (cp_45 * t_45t - cp_5 * t_5t)
            * self.gearbox_efficiency
        )

        t_8 = t_5t / (1 + (gamma5 - 1) / 2 * mach_8 ** 2)
        v_8 = mach_8 * np.sqrt(gamma5 * r_g * t_8)

        # Computing the exhaust thrust
        thrust_exhaust = (
            air_mass_flow
            * (1 + f_fuel_ratio - icb - g_r)
            * (v_8 - flight_mach * np.sqrt(t_0 * 287.0 * 1.4))
        )

        power_sol = power / 1000.0

        self.t_45t_sol = t_45t
        self.opr_sol = opr
        self.power_sol = power_sol
        self.thrust_sol = thrust_exhaust

        return t_4t, air_mass_flow, power_sol, f_fuel_ratio, p_41t, opr, t_41t, t_45t

    def turboshaft_performance_envelope_solver_real_gas(
        self, fuel_flow, limit_name, limit_value, altitude, mach_vol
    ):
        """
        Function that computes the turboprop performance when one of the engine limits is reached

        :param fuel_flow: the fuel flow, in kg/s
        :param limit_name: the name of the limit for which we want to find the performance,
        can be "opr", "t_45t" or "power"
        :param limit_value: the value of the limit, no unit for the opr, K for the t_45t and kW for
        the power
        :param altitude: the flight altitude, in m
        :param mach_vol: the flight mach number

        :return f_value: for the given fuel flow, the difference between the limit and the
        computed value
        """
        m_c = fuel_flow[0]

        if limit_name == "opr":
            (
                _,
                _,
                power,
                _,
                _,
                opr,
                _,
                t_45t,
            ) = self.turboshaft_performance_real_gas(altitude, mach_vol, m_c)
            var_2_return = opr

        elif limit_name == "t_45t":
            (
                _,
                _,
                power,
                _,
                _,
                opr,
                _,
                t_45t,
            ) = self.turboshaft_performance_real_gas(altitude, mach_vol, m_c)
            var_2_return = t_45t

        elif limit_name == "power":
            (
                _,
                _,
                power,
                _,
                _,
                opr,
                _,
                t_45t,
            ) = self.turboshaft_performance_real_gas(altitude, mach_vol, m_c)
            var_2_return = power

        else:
            raise FastBasicTPEngineUnknownLimit(
                "Unknown limit provided, should be opr, t_45t or power"
            )

        f_value = var_2_return - limit_value

        return f_value

    def turboshaft_performance_envelope_limits_real_gas(
        self, limit_name, limit_value, altitude, mach_vol
    ):

        """
        Function that finds the fuel flow which leads to the desired engine limit

        :param limit_name: the name of the limit for which we want to find the performance, can be
         "opr", "t_45t" or "power"
        :param limit_value: the value of the limit, no unit for the opr, K for the t_45t and kW for
        the power
        :param altitude: the flight altitude, in m
        :param mach_vol: the flight mach number

        :return fuel_flow: the fuel flow that constrains the engine to the desired limit
        """

        fuel_flow_0 = np.array([0.046 * (1 - altitude / 29000)])

        fuel_flow = np.array(
            fsolve(
                self.turboshaft_performance_envelope_solver_real_gas,
                fuel_flow_0,
                (limit_name, limit_value, altitude, mach_vol),
            )
        )[0]

        return fuel_flow

    def turboshaft_compute_within_limits(self, target_power, altitude, mach_vol):

        """
        Computes the fuel flow necessary to achieve the target power and checks if it is within
        the capability of the turboprop. If it leads to constraints greater than the one defined
        in the XML, gives the fuel flow corresponding to the highest power achievable within the
        limits.

        :param target_power: required power, in kW.
        :param altitude: the flight altitude, in m.
        :param mach_vol: the flight mach number.

        :return fuel: the fuel flow giving the required power or highest achievable power, in kg/s.
        :return power_sol: the required power or highest achievable power, in kW.
        :return thrust_sol: the exhaust thrust, in N.
        """

        # Check if we can get to the target power
        fuel = self.turboshaft_performance_envelope_limits_real_gas(
            "power", target_power, altitude, mach_vol
        )

        t_45t_sol = self.t_45t_sol
        opr_sol = self.opr_sol
        power_sol = self.power_sol
        thrust_sol = self.thrust_sol

        # If target can't be reached we see which limit we have attained and get the performances
        # corresponding to that limit
        if t_45t_sol > self.itt_limit:
            fuel = self.turboshaft_performance_envelope_limits_real_gas(
                "t_45t", self.itt_limit, altitude, mach_vol
            )
            # print("t_45t limit processing", t_45t_sol)

            opr_sol = self.opr_sol
            power_sol = self.power_sol
            thrust_sol = self.thrust_sol

        if opr_sol > self.opr_limit:
            fuel = self.turboshaft_performance_envelope_limits_real_gas(
                "opr", self.opr_limit, altitude, mach_vol
            )
            # print("opr limit processing", opr_sol)

            power_sol = self.power_sol
            thrust_sol = self.thrust_sol

        return fuel, power_sol[0], thrust_sol[0]

    def compute_flight_points(self, flight_points: oad.FlightPoint):
        # pylint: disable=too-many-arguments
        # they define the trajectory
        self.specific_shape = np.shape(flight_points.mach)
        if isinstance(flight_points.mach, float):
            sfc, thrust_rate, thrust = self._compute_flight_points(
                flight_points.mach,
                flight_points.altitude,
                flight_points.thrust_is_regulated,
                flight_points.thrust_rate,
                flight_points.thrust,
            )
            flight_points.sfc = sfc
            flight_points.thrust_rate = thrust_rate
            flight_points.thrust = thrust
        else:
            mach = np.asarray(flight_points.mach)
            altitude = np.asarray(flight_points.altitude).flatten()
            if flight_points.thrust_is_regulated is None:
                thrust_is_regulated = None
            else:
                thrust_is_regulated = np.asarray(flight_points.thrust_is_regulated).flatten()
            if flight_points.thrust_rate is None:
                thrust_rate = None
            else:
                thrust_rate = np.asarray(flight_points.thrust_rate).flatten()
            if flight_points.thrust is None:
                thrust = None
            else:
                thrust = np.asarray(flight_points.thrust).flatten()
            self.specific_shape = np.shape(mach)
            sfc, thrust_rate, thrust = self._compute_flight_points(
                mach.flatten(),
                altitude,
                thrust_is_regulated,
                thrust_rate,
                thrust,
            )
            if len(self.specific_shape) != 1:  # reshape data that is not array form
                # noinspection PyUnresolvedReferences
                flight_points.sfc = sfc.reshape(self.specific_shape)
                # noinspection PyUnresolvedReferences
                flight_points.thrust_rate = thrust_rate.reshape(self.specific_shape)
                # noinspection PyUnresolvedReferences
                flight_points.thrust = thrust.reshape(self.specific_shape)
            else:
                flight_points.sfc = sfc
                flight_points.thrust_rate = thrust_rate
                flight_points.thrust = thrust

    def _compute_flight_points(
        self,
        mach: Union[float, Sequence],
        altitude: Union[float, Sequence],
        thrust_is_regulated: Optional[Union[bool, Sequence]] = None,
        thrust_rate: Optional[Union[float, Sequence]] = None,
        thrust: Optional[Union[float, Sequence]] = None,
    ) -> Tuple[Union[float, Sequence], Union[float, Sequence], Union[float, Sequence]]:
        """
        Same as :meth:`compute_flight_points`.

        :param mach: Mach number
        :param altitude: (unit=m) altitude w.r.t. to sea level
        :param thrust_is_regulated: tells if thrust_rate or thrust should be used (works
        element-wise)
        :param thrust_rate: thrust rate (unit=none)
        :param thrust: required thrust (unit=N)
        :return: SFC (in kg/s/N), thrust rate, thrust (in N)

        Computes the Specific Fuel Consumption based on aircraft trajectory conditions.
        """

        # Treat inputs (with check on thrust rate <=1.0)
        if thrust_is_regulated is not None:
            thrust_is_regulated = np.asarray(np.round(thrust_is_regulated, 0), dtype=bool)
        thrust_is_regulated, thrust_rate, thrust = self._check_thrust_inputs(
            thrust_is_regulated, thrust_rate, thrust
        )
        thrust_is_regulated = np.asarray(np.round(thrust_is_regulated, 0), dtype=bool)
        thrust_rate = np.asarray(thrust_rate)
        thrust = np.asarray(thrust)

        # Get maximum thrust @ given altitude & mach
        atmosphere = Atmosphere(np.asarray(altitude), altitude_in_feet=False)
        mach = np.asarray(mach) + (np.asarray(mach) == 0) * 1e-12
        atmosphere.mach = mach
        max_thrust = self.max_thrust(atmosphere)

        # We compute thrust values from thrust rates when needed
        idx = np.logical_not(thrust_is_regulated)
        if np.size(max_thrust) == 1:
            maximum_thrust = max_thrust
            out_thrust_rate = thrust_rate
            out_thrust = thrust
        else:
            out_thrust_rate = (
                np.full(np.shape(max_thrust), thrust_rate.item())
                if np.size(thrust_rate) == 1
                else thrust_rate
            )
            out_thrust = (
                np.full(np.shape(max_thrust), thrust.item()) if np.size(thrust) == 1 else thrust
            )
            maximum_thrust = max_thrust[idx]
        if np.any(idx):
            out_thrust[idx] = out_thrust_rate[idx] * maximum_thrust
        if np.any(thrust_is_regulated):
            out_thrust[thrust_is_regulated] = np.minimum(
                out_thrust[thrust_is_regulated], max_thrust[thrust_is_regulated]
            )

        # thrust_rate is obtained from entire thrust vector (could be optimized if needed,
        # as some thrust rates that are computed may have been provided as input)
        out_thrust_rate = out_thrust / max_thrust

        # Now SFC (g/kwh) can be computed and converted to sfc_thrust (kg/N) to match computation
        # from turboshaft
        sfc, mechanical_power = self.sfc(out_thrust, atmosphere)
        sfc_time = mechanical_power * sfc  # sfc in kg/s
        sfc_thrust = sfc_time / np.maximum(out_thrust, 1e-6)  # avoid 0 division

        return sfc_thrust, out_thrust_rate, out_thrust

    @staticmethod
    def _check_thrust_inputs(
        thrust_is_regulated: Optional[Union[float, Sequence]],
        thrust_rate: Optional[Union[float, Sequence]],
        thrust: Optional[Union[float, Sequence]],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Checks that inputs are consistent and return them in proper shape.
        Some of the inputs can be None, but outputs will be proper numpy arrays.
        :param thrust_is_regulated:
        :param thrust_rate:
        :param thrust:
        :return: the inputs, but transformed in numpy arrays.
        """
        # Ensure they are numpy array
        if thrust_is_regulated is not None:
            # As OpenMDAO may provide floats that could be slightly different
            # from 0. or 1., a rounding operation is needed before converting
            # to booleans
            thrust_is_regulated = np.asarray(np.round(thrust_is_regulated, 0), dtype=bool)
        if thrust_rate is not None:
            thrust_rate = np.asarray(thrust_rate)
        if thrust is not None:
            thrust = np.asarray(thrust)

        # Check inputs: if use_thrust_rate is None, we will use the provided input between
        # thrust_rate and thrust
        if thrust_is_regulated is None:
            if thrust_rate is not None:
                thrust_is_regulated = False
                thrust = np.empty_like(thrust_rate)
            elif thrust is not None:
                thrust_is_regulated = True
                thrust_rate = np.empty_like(thrust)
            else:
                raise FastBasicICEngineInconsistentInputParametersError(
                    "When use_thrust_rate is None, either thrust_rate or thrust should be provided."
                )

        elif np.size(thrust_is_regulated) == 1:
            # Check inputs: if use_thrust_rate is a scalar, the matching input(thrust_rate or
            # thrust) must be provided.
            if thrust_is_regulated:
                if thrust is None:
                    raise FastBasicICEngineInconsistentInputParametersError(
                        "When thrust_is_regulated is True, thrust should be provided."
                    )
                thrust_rate = np.empty_like(thrust)
            else:
                if thrust_rate is None:
                    raise FastBasicICEngineInconsistentInputParametersError(
                        "When thrust_is_regulated is False, thrust_rate should be provided."
                    )
                thrust = np.empty_like(thrust_rate)

        else:
            # Check inputs: if use_thrust_rate is not a scalar, both thrust_rate and thrust must be
            # provided and have the same shape as use_thrust_rate
            if thrust_rate is None or thrust is None:
                raise FastBasicICEngineInconsistentInputParametersError(
                    "When thrust_is_regulated is a sequence, both thrust_rate and thrust should be "
                    "provided."
                )
            if np.shape(thrust_rate) != np.shape(thrust_is_regulated) or np.shape(
                thrust
            ) != np.shape(thrust_is_regulated):
                raise FastBasicICEngineInconsistentInputParametersError(
                    "When use_thrust_rate is a sequence, both thrust_rate and thrust should have "
                    "same shape as use_thrust_rate"
                )

        return thrust_is_regulated, thrust_rate, thrust

    def propeller_efficiency(
        self, thrust: Union[float, Sequence[float]], atmosphere: Atmosphere
    ) -> Union[float, Sequence]:
        """
        Compute the propeller efficiency.

        :param thrust: Thrust (in N)
        :param atmosphere: Atmosphere instance at intended altitude
        :return: efficiency
        """
        # Include advance ratio loss in here, we will assume that since we work at constant RPM
        # the change in advance ration is equal to a change in velocity
        installed_airspeed = atmosphere.true_airspeed * self.effective_J

        propeller_efficiency_SL = interp2d(
            self.thrust_SL,
            self.speed_SL,
            self.efficiency_SL * self.effective_efficiency_ls,  # Include the efficiency loss
            # in here
            kind="cubic",
        )
        propeller_efficiency_CL = interp2d(
            self.thrust_CL,
            self.speed_CL,
            self.efficiency_CL * self.effective_efficiency_cruise,  # Include the efficiency loss
            # in here
            kind="cubic",
        )
        if isinstance(atmosphere.true_airspeed, float):
            thrust_interp_SL = np.minimum(
                np.maximum(np.min(self.thrust_SL), thrust),
                np.interp(installed_airspeed, self.speed_SL, self.thrust_limit_SL),
            )
            thrust_interp_CL = np.minimum(
                np.maximum(np.min(self.thrust_CL), thrust),
                np.interp(installed_airspeed, self.speed_CL, self.thrust_limit_CL),
            )
        else:
            thrust_interp_SL = np.minimum(
                np.maximum(np.min(self.thrust_SL), thrust),
                np.interp(list(installed_airspeed), self.speed_SL, self.thrust_limit_SL),
            )
            thrust_interp_CL = np.minimum(
                np.maximum(np.min(self.thrust_CL), thrust),
                np.interp(list(installed_airspeed), self.speed_CL, self.thrust_limit_CL),
            )
        if isinstance(thrust, float):  # calculate for float
            lower_bound = float(propeller_efficiency_SL(thrust_interp_SL, installed_airspeed))
            upper_bound = float(propeller_efficiency_CL(thrust_interp_CL, installed_airspeed))
            altitude = atmosphere.get_altitude(altitude_in_feet=False)
            propeller_efficiency = np.interp(
                altitude, [0, self.cruise_altitude_propeller], [lower_bound, upper_bound]
            )
        else:  # calculate for array
            propeller_efficiency = np.zeros(np.size(thrust))
            for idx in range(np.size(thrust)):
                lower_bound = propeller_efficiency_SL(
                    thrust_interp_SL[idx], installed_airspeed[idx]
                )
                upper_bound = propeller_efficiency_CL(
                    thrust_interp_CL[idx], installed_airspeed[idx]
                )
                altitude = atmosphere.get_altitude(altitude_in_feet=False)[idx]
                propeller_efficiency[idx] = (
                    lower_bound
                    + (upper_bound - lower_bound)
                    * np.minimum(altitude, self.cruise_altitude_propeller)
                    / self.cruise_altitude_propeller
                )

        return propeller_efficiency

    def compute_max_power(self, flight_points: oad.FlightPoint) -> Union[float, Sequence]:
        """
        Compute the turboprop maximum power @ given flight-point.

        :param flight_points: current flight point, with altitude in meters as always !
        :return: maximum power in kW
        """

        h_vol = flight_points.altitude
        mach_vol = flight_points.mach
        _, power_out, _ = self.turboshaft_compute_within_limits(
            self.max_power_avail, h_vol, mach_vol
        )

        return power_out

    def sfc(
        self,
        thrust: Union[float, Sequence[float]],
        atmosphere: Atmosphere,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computation of the SFC.

        :param thrust: Thrust (in N)
        :param atmosphere: Atmosphere instance at intended altitude
        :return: SFC (in kg/s/W) and power (in W)
        """

        # Compute sfc
        power_shaft = np.zeros(np.size(thrust))
        # torque = np.zeros(np.size(thrust))
        sfc = np.zeros(np.size(thrust))
        if np.size(thrust) == 1:
            thrust_propeller = thrust
            while True:
                power_shaft = (
                    thrust_propeller
                    * atmosphere.true_airspeed
                    / self.propeller_efficiency(thrust_propeller, atmosphere)
                )
                h_vol_point = atmosphere.get_altitude(altitude_in_feet=False)
                mach_vol_point = atmosphere.mach

                power_in_kw = power_shaft / 1000.0

                fuel_point, power_out, thrust_exhaust = self.turboshaft_compute_within_limits(
                    power_in_kw, h_vol_point, mach_vol_point
                )
                power_out_watts = power_out * 1000.0

                sfc = fuel_point / power_out_watts
                power_shaft = power_out_watts
                if abs(thrust - thrust_propeller - thrust_exhaust) / thrust < 1e-3:
                    break
                else:
                    thrust_propeller = thrust - thrust_exhaust
        else:
            for idx in range(np.size(thrust)):
                thrust_propeller = thrust[idx]
                local_atmosphere = Atmosphere(
                    atmosphere.get_altitude(altitude_in_feet=False)[idx], altitude_in_feet=False
                )
                local_atmosphere.mach = atmosphere.mach[idx]
                while True:
                    power_shaft[idx] = (
                        thrust_propeller
                        * local_atmosphere.true_airspeed
                        / self.propeller_efficiency(thrust_propeller, local_atmosphere)
                    )
                    h_vol = local_atmosphere.get_altitude(altitude_in_feet=False)
                    mach_vol = local_atmosphere.mach
                    power_in_kw = power_shaft[idx] / 1000.0
                    fuel, power_out, thrust_exhaust = self.turboshaft_compute_within_limits(
                        power_in_kw, h_vol, mach_vol
                    )
                    power_out_watts = power_out * 1000.0
                    sfc[idx] = fuel / power_out_watts
                    power_shaft[idx] = power_out_watts
                    if abs(thrust[idx] - thrust_propeller - thrust_exhaust) / thrust[idx] < 1e-3:
                        break
                    else:
                        thrust_propeller = thrust[idx] - thrust_exhaust

        return sfc, power_shaft

    def max_thrust(
        self,
        atmosphere: Atmosphere,
    ) -> np.ndarray:
        """
        Computation of maximum thrust either due to propeller thrust limit or turboprop max power.

        :param atmosphere: Atmosphere instance at intended altitude (should be <=20km)
        :return: maximum thrust (in N)
        """

        # Calculate maximum propeller thrust @ given altitude and speed
        if isinstance(atmosphere.true_airspeed, float):
            lower_bound = np.interp(atmosphere.true_airspeed, self.speed_SL, self.thrust_limit_SL)
            upper_bound = np.interp(atmosphere.true_airspeed, self.speed_CL, self.thrust_limit_CL)
        else:
            lower_bound = np.interp(
                list(atmosphere.true_airspeed), self.speed_SL, self.thrust_limit_SL
            )
            upper_bound = np.interp(
                list(atmosphere.true_airspeed), self.speed_CL, self.thrust_limit_CL
            )
        altitude = atmosphere.get_altitude(altitude_in_feet=False)
        thrust_max_propeller = (
            lower_bound
            + (upper_bound - lower_bound)
            * np.minimum(altitude, self.cruise_altitude_propeller)
            / self.cruise_altitude_propeller
        )

        altitudes_to_evaluate = atmosphere.get_altitude(altitude_in_feet=False)
        if np.size(altitudes_to_evaluate) == 1:
            h_vol = altitudes_to_evaluate
            mach_vol = atmosphere.mach
            _, power_out, thrust_exhaust = self.turboshaft_compute_within_limits(
                self.max_power_avail, h_vol, mach_vol
            )
            power_out_watts = power_out * 1000.0
            max_power = power_out_watts
            exhaust_thrust_at_max_power = thrust_exhaust
        else:
            max_power = np.zeros(len(altitudes_to_evaluate))
            exhaust_thrust_at_max_power = np.zeros(len(altitudes_to_evaluate))
            for idx, h_vol in enumerate(altitudes_to_evaluate):
                mach_vol = atmosphere.mach[idx]
                _, power_out, thrust_exhaust = self.turboshaft_compute_within_limits(
                    self.max_power_avail, h_vol, mach_vol
                )
                power_out_watts = power_out * 1000.0
                # max_power = power_out_watts
                max_power[idx] = power_out_watts
                exhaust_thrust_at_max_power[idx] = thrust_exhaust

        # Max power --> Array containing the maximum available power at the given flight points
        # thrust_max_propeller --> Array containing the maximum available thrust at the given
        # flight points for the propeller

        # Found thrust relative to turboprop maximum power @ given altitude and speed: calculates
        # first thrust interpolation vector (between min and max of propeller table) and
        # associated efficiency, then calculates power and found thrust (interpolation limits to
        # max propeller thrust)
        thrust_interp = np.linspace(
            np.min(self.thrust_SL) * np.ones(np.size(thrust_max_propeller)),
            thrust_max_propeller,
            10,
        ).transpose()
        if np.size(altitude) == 1:  # Calculate for float
            thrust_max_global = 0.0
            local_atmosphere = Atmosphere(
                altitude * np.ones(np.size(thrust_interp)), altitude_in_feet=False
            )
            local_atmosphere.mach = atmosphere.mach * np.ones(np.size(thrust_interp))
            propeller_efficiency = self.propeller_efficiency(thrust_interp[0], local_atmosphere)
            mechanical_power = thrust_interp[0] * atmosphere.true_airspeed / propeller_efficiency
            if np.min(mechanical_power) > max_power:
                efficiency_relative_error = 1
                propeller_efficiency = propeller_efficiency[0]
                while efficiency_relative_error > 1e-2:
                    thrust_max_global = max_power * propeller_efficiency / atmosphere.true_airspeed
                    propeller_efficiency_new = self.propeller_efficiency(
                        thrust_max_global, atmosphere
                    )
                    efficiency_relative_error = np.abs(
                        (propeller_efficiency_new - propeller_efficiency)
                        / efficiency_relative_error
                    )
                    propeller_efficiency = propeller_efficiency_new
                thrust_max_global += exhaust_thrust_at_max_power
            # TODO : is there a need to take into account the fact when the turboprop is
            #  over-sized and the limiting  factor becomes the propeller ? Is it physically
            #  relevant ? Here it looks like in any case, the  engine is always the  limiting
            #  factor
            else:
                thrust_max_global = (
                    np.interp(max_power, mechanical_power, thrust_interp[0])
                    + exhaust_thrust_at_max_power
                )
            if isinstance(thrust_max_global, float):
                thrust_max_global = np.array([thrust_max_global])
        else:  # Calculate for array
            thrust_max_global = np.zeros(np.size(altitude))
            for idx in range(np.size(altitude)):
                local_atmosphere = Atmosphere(
                    altitude[idx] * np.ones(np.size(thrust_interp[idx])), altitude_in_feet=False
                )
                local_atmosphere.mach = atmosphere.mach[idx] * np.ones(np.size(thrust_interp[idx]))
                propeller_efficiency = self.propeller_efficiency(
                    thrust_interp[idx], local_atmosphere
                )
                mechanical_power = (
                    thrust_interp[idx] * atmosphere.true_airspeed[idx] / propeller_efficiency
                )
                # mechanical power contain the shaft power required to obtain the thrust from
                # thrust_interp, where thrust_interp[idx] is the thrust interpolation array
                # corresponding to the flight conditions [idx] Mechanical power is already
                # limited by the maximum thrust that the propeller can produce so we are sure
                # that we will never go above our propeller's capacity

                # If the limiting factor is the turboprop, thrust_max_global is computed based on
                # max_power[idx]
                if (
                    np.min(mechanical_power) > max_power[idx]
                ):  # take the lower bound efficiency for calculation
                    efficiency_relative_error = 1
                    local_atmosphere = Atmosphere(altitude[idx], altitude_in_feet=False)
                    local_atmosphere.mach = atmosphere.mach[idx]
                    propeller_efficiency = propeller_efficiency[0]
                    while efficiency_relative_error > 1e-2:
                        thrust_max_global[idx] = (
                            max_power[idx] * propeller_efficiency / atmosphere.true_airspeed[idx]
                        )
                        propeller_efficiency_new = self.propeller_efficiency(
                            thrust_max_global[idx], local_atmosphere
                        )
                        efficiency_relative_error = np.abs(
                            (propeller_efficiency_new - propeller_efficiency)
                            / efficiency_relative_error
                        )
                        propeller_efficiency = propeller_efficiency_new
                    thrust_max_global[idx] += exhaust_thrust_at_max_power[idx]
                else:
                    thrust_max_global[idx] = (
                        np.interp(max_power[idx], mechanical_power, thrust_interp[idx])
                        + exhaust_thrust_at_max_power[idx]
                    )

        return thrust_max_global

    def compute_weight(self) -> float:
        """
        Computes weight of uninstalled propulsion depending on maximum power. Uses a regression
        based on the PT6A family which data have been taken according to :
        https://www.easa.europa.eu/downloads/7787/en.
        """

        # Design point power in kW, uninstalled weight in lbf
        uninstalled_weight = (1.53833774e2 + 8.61372333e-2 * self.max_power_avail) * 2.2046
        self.engine.mass = uninstalled_weight

        return uninstalled_weight

    def compute_dimensions(self) -> (float, float, float, float):
        """
        Computes propulsion dimensions (engine/nacelle) from maximum power.
        Model from a regression on the PT6 family

        """

        max_thermal_power_in_kw = self.design_point_power * 1.34102

        # Compute engine dimensions
        if max_thermal_power_in_kw < 850.0:
            self.engine.height = (
                interp1d([500.0, 850.0], [21.0, 25.0])(max_thermal_power_in_kw) * 0.0254
            )
            self.engine.width = 21.5 * 0.0254
        else:
            self.engine.height = 22 * 0.0254
            self.engine.width = 19.5 * 0.0254

        self.engine.length = (1241.0 + 61.0 * self.opr_d) / 1000.0

        if self.prop_layout == 3.0:
            nacelle_length = 1.30 * self.engine.length
            # Based on the length between nose and firewall for TB20 and SR22
        else:
            nacelle_length = 1.95 * self.engine.length
            # Based on the length of the nacelle on the Beech 1900D

        # Compute nacelle dimensions
        self.nacelle = Nacelle(
            height=self.engine.height * 1.1,
            width=self.engine.width * 1.1,
            length=nacelle_length,
        )
        self.nacelle.wet_area = 2 * (self.nacelle.height + self.nacelle.width) * self.nacelle.length

        return (
            self.nacelle["height"],
            self.nacelle["width"],
            self.nacelle["length"],
            self.nacelle["wet_area"],
        )

    def compute_drag(self, mach, unit_reynolds, wing_mac):
        """
        Compute nacelle drag coefficient cd0.

        """

        # Compute dimensions
        _, _, _, _ = self.compute_dimensions()
        # Local Reynolds:
        reynolds = unit_reynolds * self.nacelle.length
        # Roskam method for wing-nacelle interaction factor (vol 6 page 3.62)
        cf_nac = 0.455 / (
            (1 + 0.144 * mach ** 2) ** 0.65 * (math.log10(reynolds)) ** 2.58
        )  # 100% turbulent
        fineness = self.nacelle.length / math.sqrt(
            4 * self.nacelle.height * self.nacelle.width / math.pi
        )
        ff_nac = 1 + 0.35 / fineness  # Raymer (seen in Gudmunsson)
        if_nac = 1.2  # Jenkinson (seen in Gudmundsson)
        drag_force = cf_nac * ff_nac * self.nacelle.wet_area * if_nac

        return drag_force


@AddKeyAttributes(ENGINE_LABELS)
class Engine(DynamicAttributeDict):
    """
    Class for storing data for engine.

    An instance is a simple dict, but for convenience, each item can be accessed
    as an attribute (inspired by pandas DataFrames). Hence, one can write::

        >>> engine = Engine(power_SL=10000.)
        >>> engine["power_SL"]
        10000.0
        >>> engine["mass"] = 70000.
        >>> engine.mass
        70000.0
        >>> engine.mass = 50000.
        >>> engine["mass"]
        50000.0

    Note: constructor will forbid usage of unknown keys as keyword argument, but
    other methods will allow them, while not making the matching between dict
    keys and attributes, hence::

        >>> engine["foo"] = 42  # Ok
        >>> bar = engine.foo  # raises exception !!!!
        >>> engine.foo = 50  # allowed by Python
        >>> # But inner dict is not affected:
        >>> engine.foo
        50
        >>> engine["foo"]
        42

    This class is especially useful for generating pandas DataFrame: a pandas
    DataFrame can be generated from a list of dict... or a list of oad.FlightPoint
    instances.

    The set of dictionary keys that are mapped to instance attributes is given by
    the :meth:`get_attribute_keys`.
    """


@AddKeyAttributes(NACELLE_LABELS)
class Nacelle(DynamicAttributeDict):
    """
    Class for storing data for nacelle.

    Similar to :class:`Engine`.
    """
