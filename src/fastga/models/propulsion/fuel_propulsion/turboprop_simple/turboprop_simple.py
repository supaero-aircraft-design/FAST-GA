"""
Module for calculating the parameters of a PT6A-66D Gas Turbine
"""

from fastoad.model_base import Atmosphere
import math


class GasTurbine(object):
    """ Class of a Gas Turbine Pratt & Whitney Canada PT6A-66D """

    def __init__(self, aircraft, generator_usage=False):
        self.aircraft = aircraft
        self.generator_usage = (
            generator_usage  # True if the GT is used as a generator, False if used as an engine
        )

    def compute_power_max(self, altitude=0.0, V_TAS=0.0):
        """
        Compute the power out of the gas turbine, assumed constant with V_TAS and altitude

        :param V_TAS: TAS (m/s)
        :param alt: altitude (ft)
        :return: Gas Turbine Power (W)
        """

        # We will implement the model from Mattingly et al. presented in Gudmunsson with an estimate
        # of the static thrust based on ADT with the inclusion of efficiency

        prop_dia = self.aircraft.vars_geometry["propeller_dia"]
        p_to = self.aircraft.vars_sizing_mission["p_takeoff"] * 1000.0
        prop_eff = self.aircraft.vars_propeller["propeller_eff_max"]
        gamma = 1.4
        TR = 1.0  # Throttle ratio, value used here correspond to a design point in static conditions at SL

        atm_sl = Atmosphere(0)
        temperature_sl = atm_sl.temperature
        density_sl = atm_sl.density
        pressure_sl = atm_sl.pressure
        viscosity_sl = atm_sl.kinematic_viscosity
        sos_sl = atm_sl.speed_of_sound

        atm = Atmosphere(altitude)
        temperature = atm.temperature
        density = atm.density
        pressure = atm.pressure
        viscosity = atm.kinematic_viscosity
        sos = atm.speed_of_sound

        prop_swept_area = math.pi * (prop_dia / 2.0) ** (2.0)

        T_SL = p_to ** (2.0 / 3.0) * (2.0 * density_SL * prop_swept_area) ** (1.0 / 3.0) * prop_eff

        Mach = V_TAS / sos

        theta_0 = temperature / temperature_SL * (1.0 + (gamma - 1.0) / 2.0 * Mach ** (2.0))
        delta_0 = (
            pression
            / pression_SL
            * (1.0 + (gamma - 1.0) / 2.0 * Mach ** (2.0)) ** (gamma / (gamma - 1.0))
        )

        if Mach < 0.1:  # For low flight speed we take this simple approach which reminds of ICE
            thrust = T_SL * delta_0
            P_max = p_to * delta_0
        elif theta_0 <= TR:
            thrust = T_SL * delta_0 * (1.0 - 0.96 * (Mach - 0.1) ** 0.25)
            P_max = thrust * V_TAS / prop_eff
        else:
            thrust = (
                T_SL
                * delta_0
                * (1.0 - 0.96 * (Mach - 0.1) ** 0.25 - 3.0 * (theta_0 - TR) / (8.13 * (Mach - 0.1)))
            )
            P_max = thrust * V_TAS / prop_eff

        print("P_max"), P_max

        # Assumption: GT power is assumed constant with V_TAS and altitude
        if self.generator_usage:
            return self.aircraft.vars_propulsion_hybrid["generator_max_power"]
        else:
            return self.aircraft.vars_propulsion_gas_turbine["gas_turbine_max_power"]

    def compute_fuel_consumption(self, alt, power, throttle=1.0):
        """
        Compute the fuel flow and the Power Specific Fuel Consumption (PSFC) out of the gas turbine for a given altitude
        and power

        :param alt: altitude (ft)
        :param power: power (W)
        :param throttle: (unused) --> /!\ not to be removed for genericity
        :return: Fuel flow (kg/s), PSFC (kg/s/kW)
        """
        # we compute the Power Specific Fuel Consumption (PSFC, in (kg/s)/kW) as a function of altitude
        PSFC = (
            3.447e-14 * alt ** 2 - 2.359e-9 * alt + 1.298e-4
        )  # regression from TBM 900 Pilot Instruction Manual
        # PSFC = 3.056e-14 * alt ** 2 - 2.049e-9 * alt + 1.129e-4  # 40 drag points penalty
        # PSFC = 2.892e-14 * alt ** 2 - 1.922e-9 * alt + 1.059e-4  # 60 drag points penalty

        # we multiply this PSFC to the required power to get the fuel flow
        ff = PSFC * power / 1000
        return ff, PSFC

    def compute_weight(self, engine_number):
        """
        Compute gas turbine weight based on the power density

        :param engine_number: (unused) --> /!\ not to be removed for genericity
        :return: weight (kg)
        """
        if self.generator_usage:
            max_power = self.aircraft.vars_propulsion_hybrid["generator_max_power"]
            # max_power = self.aircraft.vars_propulsion_hybrid_dep['generator_max_power'] * 1.35962  # kW to SHP
            power_mass_ratio = self.aircraft.vars_propulsion_hybrid["generator_power_mass"]
        else:
            max_power = self.aircraft.vars_propulsion_gas_turbine["gas_turbine_max_power"]
            # max_power = self.aircraft.vars_propulsion_gas_turbine['gas_turbine_max_power'] * 1.35962  # kW to SHP
            power_mass_ratio = self.aircraft.vars_propulsion_gas_turbine["gas_turbine_power_mass"]

        w_gas_turbine = max_power / power_mass_ratio

        # w_gas_turbine = (max_power - 110.7) / 2.631

        return w_gas_turbine

    def change_power(self, power):
        """
        Change the gas turbine power to a new value

        :param power: power (W)
        """
        if self.generator_usage:
            self.aircraft.vars_propulsion_hybrid["generator_max_power"] = power / 1000.0  # in kW
        else:
            self.aircraft.vars_propulsion_gas_turbine["gas-turbine_max_power"] = (
                power / 1000.0
            )  # in kW
