"""Parametric turboprop engine."""
# -*- coding: utf-8 -*-
#  This file is part of FAST-OAD_CS23 : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2024  ONERA & ISAE-SUPAERO
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

import logging
from collections import OrderedDict
from typing import Union, Sequence, Tuple, Optional
from scipy.interpolate import interp1d, RectBivariateSpline
import numpy as np

import openmdao.api as om

import fastoad.api as oad
from fastoad.constants import EngineSetting
from fastoad.exceptions import FastUnknownEngineSettingError
from stdatm import Atmosphere

from fastga.models.propulsion.fuel_propulsion.basicTurbo_prop.exceptions import (
    FastBasicICEngineInconsistentInputParametersError,
)
from fastga.models.propulsion.fuel_propulsion.base import AbstractFuelPropulsion
from fastga.models.propulsion.dict import DynamicAttributeDict, AddKeyAttributes

from .turboprop_components.turboshaft_geometry_computation import DesignPointCalculation
from .turboprop_components.turboshaft_off_design_max_power import (
    TurboshaftMaxThrustPowerLimit,
    TurboshaftMaxThrustOPRLimit,
    TurboshaftMaxThrustITTLimit,
    TurboshaftMaxThrustPropellerThrustLimit,
)
from .turboprop_components.turboshaft_off_design_fuel import Turboshaft

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

CACHE_MAX_SIZE = 128
MAX_ITER_NO_LS_PROBLEM = 10


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
        self.pr_1_ratio_design = pr_1_ratio_design
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

        # This new version of the turboprop model will use nested OpenMDAO problem to compute
        # fuel consumption, max thrust and geometry, but because the setup might take some time,
        # we won't do it automatically, only when needed i.e. when, at some point we need to call
        # the max thrust function (the compute_flight_point method calls it). Also we will make
        # it so that we only have to set it up once by introducing a setup checker and attribute
        # getter on the problems and on the turboprop geometry parameter that are required to
        # compute max thrust and fuel consumption (Areas, alpha ratios and OPR ratios).

        self._turboprop_sizing_problem = None
        self._turboprop_sizing_problem_setup = False

        self._turboprop_max_thrust_power_limit_problem = None
        self._turboprop_max_thrust_power_limit_problem_setup = False

        self._turboprop_max_thrust_opr_limit_problem = None
        self._turboprop_max_thrust_opr_limit_problem_setup = False

        self._turboprop_max_thrust_itt_limit_problem = None
        self._turboprop_max_thrust_itt_limit_problem_setup = False

        self._turboprop_max_thrust_propeller_thrust_limit_problem = None
        self._turboprop_max_thrust_propeller_thrust_limit_problem_setup = False

        self._turboprop_fuel_problem = None
        self._turboprop_fuel_problem_setup = False

        self._turboprop_fuel_problem_ls = None
        self._turboprop_fuel_problem_ls_setup = False

        self._alpha = None
        self._alpha_p = None
        self._a_41 = None
        self._a_45 = None
        self._a_8 = None
        self._opr_2_opr_1 = None

        self._cache_max_thrust = OrderedDict()

    @property
    def turboprop_sizing_problem(self) -> om.Problem:
        """OpenMDAO problem to compute the sizing point"""

        if not self._turboprop_sizing_problem_setup:

            ivc = om.IndepVarComp()
            ivc.add_output(
                "compressor_bleed_mass_flow", val=self.inter_compressor_bleed, units="kg/s"
            )
            ivc.add_output("cooling_bleed_ratio", val=self.cooling_ratio)
            # Some parameters were hard-coded in previous version of the code, we'll leave them
            # like that for now
            ivc.add_output("cabin_air_renewal_time", val=2.0, units="min")
            ivc.add_output("data:geometry:cabin:volume", val=5.0, units="m**3")
            ivc.add_output("bleed_control", val=1.0)  # Hard-coded at 1.0

            ivc.add_output("eta_225", val=self.eta_225)
            ivc.add_output("eta_253", val=self.eta_253)
            ivc.add_output("eta_445", val=self.eta_445)
            ivc.add_output("eta_455", val=self.eta_455)
            ivc.add_output("total_pressure_loss_02", val=self.pi_02)
            ivc.add_output("pressure_loss_34", val=self.pi_cc)
            ivc.add_output("combustion_energy", val=self.eta_q, units="J/kg")

            ivc.add_output("electric_power", val=self.hp_shaft_power_out / 745.7, units="hp")

            ivc.add_output(
                "settings:propulsion:turboprop:design_point:first_stage_pressure_ratio",
                val=self.pr_1_ratio_design,
            )
            ivc.add_output(
                "settings:propulsion:turboprop:efficiency:high_pressure_axe",
                val=self.eta_axe,
            )
            ivc.add_output(
                "settings:propulsion:turboprop:efficiency:gearbox",
                val=self.gearbox_efficiency,
            )
            ivc.add_output(
                "settings:propulsion:turboprop:design_point:mach_exhaust",
                val=self.exhaust_mach_design,
            )

            ivc.add_output(
                "data:propulsion:turboprop:design_point:altitude",
                val=self.design_point_altitude,
                units="m",
            )
            ivc.add_output(
                "data:propulsion:turboprop:design_point:mach", val=self.design_point_mach
            )
            ivc.add_output(
                "data:propulsion:turboprop:design_point:power",
                val=self.design_point_power,
                units="kW",
            )
            ivc.add_output(
                "data:propulsion:turboprop:design_point:turbine_entry_temperature",
                val=self.t_41t_d,
                units="degK",
            )
            ivc.add_output(
                "data:propulsion:turboprop:design_point:OPR",
                val=self.opr_d,
            )

            prob = om.Problem()
            prob.model.add_subsystem("ivc", ivc, promotes=["*"])
            prob.model.add_subsystem(
                "turboshaft_sizing",
                DesignPointCalculation(number_of_points=1),
                promotes=["*"],
            )

            prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
            prob.model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()
            prob.model.nonlinear_solver.options["iprint"] = 0
            prob.model.nonlinear_solver.options["maxiter"] = 100
            prob.model.nonlinear_solver.options["rtol"] = 1e-5
            prob.model.nonlinear_solver.options["atol"] = 5e-5
            prob.model.linear_solver = om.DirectSolver()

            prob.setup()

            self._turboprop_sizing_problem_setup = True

            self._turboprop_sizing_problem = prob

        return self._turboprop_sizing_problem

    @turboprop_sizing_problem.setter
    def turboprop_sizing_problem(self, value: om.Problem):
        """OpenMDAO problem to compute the sizing point"""

        self._turboprop_sizing_problem = value

    def get_ivc_max_thrust_problem(self) -> om.IndepVarComp:
        """
        The 4 problems that compute the maximum will have all their inputs in common, so we
        centralize the creation of the IndepVarComp that will supply them.
        """

        ivc = om.IndepVarComp()

        ivc.add_output(
            "data:aerodynamics:propeller:cruise_level:altitude",
            val=self.cruise_altitude_propeller,
            units="m",
        )
        ivc.add_output(
            "data:aerodynamics:propeller:sea_level:speed", val=self.speed_SL, units="m/s"
        )
        ivc.add_output(
            "data:aerodynamics:propeller:cruise_level:speed", val=self.speed_CL, units="m/s"
        )
        ivc.add_output(
            "data:aerodynamics:propeller:sea_level:thrust", val=self.thrust_SL, units="N"
        )
        ivc.add_output(
            "data:aerodynamics:propeller:cruise_level:thrust", val=self.thrust_CL, units="N"
        )
        ivc.add_output(
            "data:aerodynamics:propeller:sea_level:thrust_limit",
            val=self.thrust_limit_SL,
            units="N",
        )
        ivc.add_output(
            "data:aerodynamics:propeller:cruise_level:thrust_limit",
            val=self.thrust_limit_CL,
            units="N",
        )
        ivc.add_output("data:aerodynamics:propeller:sea_level:efficiency", val=self.efficiency_SL)
        ivc.add_output(
            "data:aerodynamics:propeller:cruise_level:efficiency", val=self.efficiency_CL
        )

        ivc.add_output("eta_225", val=self.eta_225)
        ivc.add_output("eta_253", val=self.eta_253)
        ivc.add_output("eta_455", val=self.eta_455)
        ivc.add_output("total_pressure_loss_02", val=self.pi_02)
        ivc.add_output("pressure_loss_34", val=self.pi_cc)
        ivc.add_output("combustion_energy", val=self.eta_q, units="J/kg")

        ivc.add_output("electric_power", val=self.hp_shaft_power_out / 745.7, units="hp")

        ivc.add_output("cooling_bleed_ratio", val=self.cooling_ratio)
        ivc.add_output("compressor_bleed_mass_flow", val=self.inter_compressor_bleed, units="kg/s")
        # Some parameters were hard-coded in previous version of the code, we'll leave them
        # like that for now
        ivc.add_output("cabin_air_renewal_time", val=2.0, units="min")
        ivc.add_output("data:geometry:cabin:volume", val=5.0, units="m**3")
        ivc.add_output("bleed_control", val=self.bleed_control)

        ivc.add_output(
            "settings:propulsion:turboprop:efficiency:high_pressure_axe",
            val=self.eta_axe,
        )
        ivc.add_output(
            "settings:propulsion:turboprop:efficiency:gearbox",
            val=self.gearbox_efficiency,
        )

        ivc.add_output("data:propulsion:turboprop:section:41", val=self.a_41, units="m**2")
        ivc.add_output("data:propulsion:turboprop:section:45", val=self.a_45, units="m**2")
        ivc.add_output("data:propulsion:turboprop:section:8", val=self.a_8, units="m**2")
        ivc.add_output("data:propulsion:turboprop:design_point:alpha", val=self.alpha)
        ivc.add_output("data:propulsion:turboprop:design_point:alpha_p", val=self.alpha_p)
        ivc.add_output("data:propulsion:turboprop:design_point:opr_2_opr_1", val=self.opr_2_opr_1)

        ivc.add_output(
            "data:aerodynamics:propeller:installation_effect:effective_efficiency:low_speed",
            val=self.effective_efficiency_ls,
        )
        ivc.add_output(
            "data:aerodynamics:propeller:installation_effect:effective_efficiency:cruise",
            val=self.effective_efficiency_cruise,
        )
        ivc.add_output(
            "data:aerodynamics:propeller:installation_effect:effective_advance_ratio",
            val=self.effective_J,
        )

        ivc.add_output("itt_limit", val=self.itt_limit, units="degK")
        ivc.add_output("shaft_power_limit", val=self.max_power_avail, units="kW")
        ivc.add_output("opr_limit", val=self.opr_limit)

        return ivc

    @property
    def turboprop_max_thrust_power_limit_problem(self) -> om.Problem:
        """OpenMDAO problem to compute the max thrust if the power is limiting"""

        if not self._turboprop_max_thrust_power_limit_problem_setup:

            ivc = self.get_ivc_max_thrust_problem()

            prob = om.Problem()
            prob.model.add_subsystem("ivc", ivc, promotes=["*"])
            prob.model.add_subsystem(
                "turboshaft_off_design_max_power",
                TurboshaftMaxThrustPowerLimit(number_of_points=1),
                promotes=["*"],
            )

            prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
            prob.model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()
            prob.model.nonlinear_solver.linesearch.options["maxiter"] = 5
            prob.model.nonlinear_solver.linesearch.options["alpha"] = 1.7
            prob.model.nonlinear_solver.linesearch.options["c"] = 2e-1
            prob.model.nonlinear_solver.options["iprint"] = 0
            prob.model.nonlinear_solver.options["maxiter"] = 100
            prob.model.nonlinear_solver.options["rtol"] = 1e-5
            prob.model.nonlinear_solver.options["atol"] = 5e-5
            prob.model.linear_solver = om.DirectSolver()

            prob.setup()

            self._turboprop_max_thrust_power_limit_problem_setup = True
            self._turboprop_max_thrust_power_limit_problem = prob

        return self._turboprop_max_thrust_power_limit_problem

    @turboprop_max_thrust_power_limit_problem.setter
    def turboprop_max_thrust_power_limit_problem(self, value: om.Problem):
        """OpenMDAO problem to compute the max thrust if the power is limiting"""

        self._turboprop_max_thrust_power_limit_problem = value

    @property
    def turboprop_max_thrust_opr_limit_problem(self) -> om.Problem:
        """OpenMDAO problem to compute the max thrust if the opr is limiting"""

        if not self._turboprop_max_thrust_opr_limit_problem_setup:
            ivc = self.get_ivc_max_thrust_problem()

            prob = om.Problem()
            prob.model.add_subsystem("ivc", ivc, promotes=["*"])
            prob.model.add_subsystem(
                "turboshaft_off_design_max_power",
                TurboshaftMaxThrustOPRLimit(number_of_points=1),
                promotes=["*"],
            )

            prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
            prob.model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()
            prob.model.nonlinear_solver.linesearch.options["maxiter"] = 5
            prob.model.nonlinear_solver.linesearch.options["alpha"] = 1.7
            prob.model.nonlinear_solver.linesearch.options["c"] = 2e-1
            prob.model.nonlinear_solver.options["iprint"] = 0
            prob.model.nonlinear_solver.options["maxiter"] = 100
            prob.model.nonlinear_solver.options["rtol"] = 1e-5
            prob.model.nonlinear_solver.options["atol"] = 5e-5
            prob.model.linear_solver = om.DirectSolver()

            prob.setup()

            self._turboprop_max_thrust_opr_limit_problem_setup = True
            self._turboprop_max_thrust_opr_limit_problem = prob

        return self._turboprop_max_thrust_opr_limit_problem

    @turboprop_max_thrust_opr_limit_problem.setter
    def turboprop_max_thrust_opr_limit_problem(self, value: om.Problem):
        """OpenMDAO problem to compute the max thrust if the opr is limiting"""

        self._turboprop_max_thrust_opr_limit_problem = value

    @property
    def turboprop_max_thrust_itt_limit_problem(self) -> om.Problem:
        """OpenMDAO problem to compute the max thrust if the ITT is limiting"""

        if not self._turboprop_max_thrust_itt_limit_problem_setup:
            ivc = self.get_ivc_max_thrust_problem()

            prob = om.Problem()
            prob.model.add_subsystem("ivc", ivc, promotes=["*"])
            prob.model.add_subsystem(
                "turboshaft_off_design_max_power",
                TurboshaftMaxThrustITTLimit(number_of_points=1),
                promotes=["*"],
            )

            prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
            prob.model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()
            prob.model.nonlinear_solver.linesearch.options["maxiter"] = 5
            prob.model.nonlinear_solver.linesearch.options["alpha"] = 1.7
            prob.model.nonlinear_solver.linesearch.options["c"] = 2e-1
            prob.model.nonlinear_solver.options["iprint"] = 0
            prob.model.nonlinear_solver.options["maxiter"] = 100
            prob.model.nonlinear_solver.options["rtol"] = 1e-5
            prob.model.nonlinear_solver.options["atol"] = 5e-5
            prob.model.linear_solver = om.DirectSolver()

            prob.setup()

            self._turboprop_max_thrust_itt_limit_problem_setup = True
            self._turboprop_max_thrust_itt_limit_problem = prob

        return self._turboprop_max_thrust_itt_limit_problem

    @turboprop_max_thrust_itt_limit_problem.setter
    def turboprop_max_thrust_itt_limit_problem(self, value: om.Problem):
        """OpenMDAO problem to compute the max thrust if the itt is limiting"""

        self._turboprop_max_thrust_itt_limit_problem = value

    @property
    def turboprop_max_thrust_propeller_thrust_limit_problem(self) -> om.Problem:
        """OpenMDAO problem to compute the max thrust if the propeller thrust is limiting"""

        if not self._turboprop_max_thrust_propeller_thrust_limit_problem_setup:
            ivc = self.get_ivc_max_thrust_problem()

            prob = om.Problem()
            prob.model.add_subsystem("ivc", ivc, promotes=["*"])
            prob.model.add_subsystem(
                "turboshaft_off_design_max_power",
                TurboshaftMaxThrustPropellerThrustLimit(number_of_points=1),
                promotes=["*"],
            )

            prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
            prob.model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()
            prob.model.nonlinear_solver.linesearch.options["maxiter"] = 5
            prob.model.nonlinear_solver.linesearch.options["alpha"] = 1.7
            prob.model.nonlinear_solver.linesearch.options["c"] = 2e-1
            prob.model.nonlinear_solver.options["iprint"] = 0
            prob.model.nonlinear_solver.options["maxiter"] = 100
            prob.model.nonlinear_solver.options["rtol"] = 1e-5
            prob.model.nonlinear_solver.options["atol"] = 5e-5
            prob.model.linear_solver = om.DirectSolver()

            prob.setup()

            self._turboprop_max_thrust_propeller_thrust_limit_problem_setup = True
            self._turboprop_max_thrust_propeller_thrust_limit_problem = prob

        return self._turboprop_max_thrust_propeller_thrust_limit_problem

    @turboprop_max_thrust_propeller_thrust_limit_problem.setter
    def turboprop_max_thrust_propeller_thrust_limit_problem(self, value: om.Problem):
        """OpenMDAO problem to compute the max thrust if the itt is limiting"""

        self._turboprop_max_thrust_propeller_thrust_limit_problem = value

    def _compute_geometry(self):
        """
        Runs the turboprop sizing problem and assigns the value for the geometric parameter
        """

        self.turboprop_sizing_problem.run_model()

        self._alpha = self.turboprop_sizing_problem.get_val(
            "data:propulsion:turboprop:design_point:alpha"
        )[0]
        self._alpha_p = self.turboprop_sizing_problem.get_val(
            "data:propulsion:turboprop:design_point:alpha_p"
        )[0]
        self._a_41 = self.turboprop_sizing_problem.get_val(
            "data:propulsion:turboprop:section:41", units="m**2"
        )[0]
        self._a_45 = self.turboprop_sizing_problem.get_val(
            "data:propulsion:turboprop:section:45", units="m**2"
        )[0]
        self._a_8 = self.turboprop_sizing_problem.get_val(
            "data:propulsion:turboprop:section:8", units="m**2"
        )[0]
        self._opr_2_opr_1 = (
            self.turboprop_sizing_problem.get_val("opr_2")[0]
            / self.turboprop_sizing_problem.get_val("opr_1")[0]
        )

    @property
    def alpha(self) -> float:
        """
        Return the temperature ratio between turbines at the design point, assumed to be
        constant.
        """

        if not self._alpha:
            self._compute_geometry()

        return self._alpha

    @alpha.setter
    def alpha(self, value: float):
        """
        Set the temperature ratio between turbines at the design point, assumed to be constant.
        """

        self._alpha = value

    @property
    def alpha_p(self) -> float:
        """
        Return the pressure ratio between turbines at the design point, assumed to be
        constant.
        """

        if not self._alpha_p:
            self._compute_geometry()

        return self._alpha_p

    @alpha_p.setter
    def alpha_p(self, value: float):
        """Set the pressure ratio between turbines at the design point, assumed to be constant."""

        self._alpha_p = value

    @property
    def a_41(self) -> float:
        """Return the combustion chamber cross-flow area."""

        if not self._a_41:
            self._compute_geometry()

        return self._a_41

    @a_41.setter
    def a_41(self, value: float):
        """Set the combustion chamber cross-flow area."""

        self._a_41 = value

    @property
    def a_45(self) -> float:
        """Return the turbine cross-flow area."""

        if not self._a_45:
            self._compute_geometry()

        return self._a_45

    @a_45.setter
    def a_45(self, value: float):
        """Set the turbine cross-flow area."""

        self._a_45 = value

    @property
    def a_8(self) -> float:
        """Return the exhaust cross-flow area."""

        if not self._a_8:
            self._compute_geometry()

        return self._a_8

    @a_8.setter
    def a_8(self, value: float):
        """Set the turbine cross-flow area."""

        self._a_8 = value

    @property
    def opr_2_opr_1(self) -> float:
        """Return the ratio between OPR 2 and OPR 1."""

        if not self._opr_2_opr_1:
            self._compute_geometry()

        return self._opr_2_opr_1

    @opr_2_opr_1.setter
    def opr_2_opr_1(self, value: float):
        """Set the ratio between OPR 2 and OPR 1."""

        self._opr_2_opr_1 = value

    @property
    def turboprop_fuel_problem(self) -> om.Problem:
        """
        OpenMDAO problem to compute the fuel consumption without linesearch algorithm. Quicker but
        can sometimes fail so it will be the favored problem to use when computing sfc.
        """
        if not self._turboprop_fuel_problem_setup:

            # Contains everything that is need and a bit more, but I don't think this more is
            # worth the new lines that would have been necessary to implement a new method
            ivc = self.get_ivc_max_thrust_problem()

            prob = om.Problem()
            prob.model.add_subsystem("ivc", ivc, promotes=["*"])
            prob.model.add_subsystem(
                "turboshaft_off_design_fuel",
                Turboshaft(number_of_points=1),
                promotes=["*"],
            )

            prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
            prob.model.nonlinear_solver.options["iprint"] = 0
            prob.model.nonlinear_solver.options["maxiter"] = MAX_ITER_NO_LS_PROBLEM
            prob.model.nonlinear_solver.options["rtol"] = 1e-5
            prob.model.nonlinear_solver.options["atol"] = 1e-5
            prob.model.linear_solver = om.DirectSolver()

            prob.setup()

            self._turboprop_fuel_problem_setup = True
            self._turboprop_fuel_problem = prob

        return self._turboprop_fuel_problem

    @turboprop_fuel_problem.setter
    def turboprop_fuel_problem(self, value: om.Problem):
        """
        OpenMDAO problem to compute the fuel consumption without linesearch algorithm. Quicker but
        can sometimes fail so it will be the favored problem to use when computing sfc.
        """

        self._turboprop_fuel_problem = value

    @property
    def turboprop_fuel_problem_ls(self) -> om.Problem:
        """
        OpenMDAO problem to compute the fuel consumption with linesearch algorithm. Longer but more
        likely to converge
        """
        if not self._turboprop_fuel_problem_ls_setup:
            # Contains everything that is need and a bit more, but I don't think this more is
            # worth the new lines that would have been necessary to implement a new method
            ivc = self.get_ivc_max_thrust_problem()

            prob = om.Problem()
            prob.model.add_subsystem("ivc", ivc, promotes=["*"])
            prob.model.add_subsystem(
                "turboshaft_off_design_fuel",
                Turboshaft(number_of_points=1),
                promotes=["*"],
            )

            prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
            prob.model.nonlinear_solver.options["iprint"] = 0
            prob.model.nonlinear_solver.options["maxiter"] = 100
            prob.model.nonlinear_solver.options["rtol"] = 1e-5
            prob.model.nonlinear_solver.options["atol"] = 1e-5
            prob.model.linear_solver = om.DirectSolver()

            prob.setup()

            self._turboprop_fuel_problem_ls_setup = True
            self._turboprop_fuel_problem_ls = prob

        return self._turboprop_fuel_problem_ls

    @turboprop_fuel_problem_ls.setter
    def turboprop_fuel_problem_ls(self, value: om.Problem):
        """
        OpenMDAO problem to compute the fuel consumption with linesearch algorithm. More likely to
        converge
        """

        self._turboprop_fuel_problem_ls = value

    def reset_problems(self):
        """
        Resets all the problem to force them to be recreated, except for the sizing problem which
        should not be necessary to reset
        """

        self._turboprop_max_thrust_power_limit_problem = None
        self._turboprop_max_thrust_power_limit_problem_setup = False

        self._turboprop_max_thrust_opr_limit_problem = None
        self._turboprop_max_thrust_opr_limit_problem_setup = False

        self._turboprop_max_thrust_itt_limit_problem = None
        self._turboprop_max_thrust_itt_limit_problem_setup = False

        self._turboprop_max_thrust_propeller_thrust_limit_problem = None
        self._turboprop_max_thrust_propeller_thrust_limit_problem_setup = False

        self._turboprop_fuel_problem = None
        self._turboprop_fuel_problem_setup = False

        self._turboprop_fuel_problem_ls = None
        self._turboprop_fuel_problem_ls_setup = False

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

    def _max_power(self, altitude: float, mach: float):
        """
        Computation of maximum power either due to propeller thrust limit or turboprop max
        power. Assumes the limits are reached in this order: Power limit, OPR limit, ITT limit,
        propeller thrust_limit. No cache will be taken for that particular function as it won't be
        used that often. May be changed later

        :param altitude: altitude in ft
        :param mach: Mach number

        :return max_power: in kW
        """

        prob_max_thrust_power_limit = self.turboprop_max_thrust_power_limit_problem

        prob_max_thrust_power_limit.set_val("altitude", val=altitude, units="ft")
        prob_max_thrust_power_limit.set_val("mach_0", val=mach)

        prob_max_thrust_power_limit.run_model()

        # Check if a bound is violated, if not we simply end the process.
        if prob_max_thrust_power_limit.get_val("opr") < prob_max_thrust_power_limit.get_val(
            "opr_limit"
        ):
            max_power = prob_max_thrust_power_limit.get_val("shaft_power", units="kW")
            return max_power

        prob_max_thrust_opr_limit = self.turboprop_max_thrust_opr_limit_problem

        prob_max_thrust_opr_limit.set_val("altitude", val=altitude, units="ft")
        prob_max_thrust_opr_limit.set_val("mach_0", val=mach)

        prob_max_thrust_opr_limit.run_model()

        # Check if a bound is violated, if not we simply end the process. Rather we will simply
        # check that the expected bound is reached
        if prob_max_thrust_opr_limit.get_val(
            "total_temperature_45", units="degK"
        ) < prob_max_thrust_opr_limit.get_val("itt_limit", units="degK"):

            max_power = prob_max_thrust_opr_limit.get_val("shaft_power", units="kW")
            return max_power

        prob_max_thrust_itt_limit = self.turboprop_max_thrust_itt_limit_problem

        prob_max_thrust_itt_limit.set_val("altitude", val=altitude, units="ft")
        prob_max_thrust_itt_limit.set_val("mach_0", val=mach)

        prob_max_thrust_itt_limit.run_model()

        # Check if a bound is violated, if not we simply end the process. Rather we will simply
        # check that the expected bound is reached
        if prob_max_thrust_itt_limit.get_val(
            "propeller_thrust", units="N"
        ) < prob_max_thrust_itt_limit.get_val("propeller_max_thrust", units="N"):

            max_power = prob_max_thrust_itt_limit.get_val("shaft_power", units="kW")
            return max_power

        prob_max_thrust_propeller_thrust_limit = (
            self.turboprop_max_thrust_propeller_thrust_limit_problem
        )

        prob_max_thrust_propeller_thrust_limit.set_val("altitude", val=altitude, units="ft")
        prob_max_thrust_propeller_thrust_limit.set_val("mach_0", val=mach)

        prob_max_thrust_propeller_thrust_limit.run_model()

        max_power = prob_max_thrust_propeller_thrust_limit.get_val("shaft_power", units="kW")
        return max_power

    def compute_max_power(self, flight_points: oad.FlightPoint) -> Union[float, Sequence]:
        """
        Compute the turboprop maximum power @ given flight-point. We'll assume the max power
        happens when the max_thrust does so we'll use the same OpenMDAO problem.

        :param flight_points: current flight point, with altitude in meters as always !
        :return: maximum power in kW
        """

        altitude_feet = flight_points.altitude / 0.3048
        mach = flight_points.mach

        power_out = self._max_power(altitude_feet, mach)

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

        altitude = atmosphere.get_altitude(altitude_in_feet=True)
        mach = atmosphere.mach

        if isinstance(altitude, float):
            fuel_consumed, power_shaft = self._fuel_consumed(altitude, mach, thrust)
            sfc = np.array(fuel_consumed / power_shaft)
            power_shaft = np.array(power_shaft)

        else:
            sfc = np.zeros_like(altitude)
            power_shaft = np.zeros_like(altitude)
            for idx in range(np.size(altitude)):
                fuel_consumed, power_shaft_loc = self._fuel_consumed(
                    altitude[idx], mach[idx], thrust[idx]
                )
                sfc[idx] = fuel_consumed / power_shaft_loc
                power_shaft[idx] = power_shaft_loc

        return sfc, power_shaft

    def _add_to_max_thrust_cache(self, key: str, value: float):
        """
        Adds a new value to the cache but first check that the size has not been exceeded. If it
        has, pop the first element that has been computed.

        :param key: key to add to the cache
        :param value: value to add to the cache
        """

        if len(self._cache_max_thrust) >= CACHE_MAX_SIZE:
            self._cache_max_thrust.popitem(last=False)

        self._cache_max_thrust[key] = value

    def _max_thrust(self, altitude: float, mach: float):
        """
        Computation of maximum thrust either due to propeller thrust limit or turboprop max
        power. Assumes the limits are reached in this order: Power limit, OPR limit, ITT limit,
        propeller thrust_limit.

        :param altitude: altitude in ft
        :param mach: Mach number
        """

        # Check if a result with approximately the same altitude and approximately the same mach
        # exist in cached results. If it does, take existing results, otherwise compute and add
        # to cache.
        cache_key = "alt" + str(round(altitude)) + "ft" + str(round(mach, 3))

        if cache_key in self._cache_max_thrust:
            return self._cache_max_thrust[cache_key]

        prob_max_thrust_power_limit = self.turboprop_max_thrust_power_limit_problem

        prob_max_thrust_power_limit.set_val("altitude", val=altitude, units="ft")
        prob_max_thrust_power_limit.set_val("mach_0", val=mach)

        prob_max_thrust_power_limit.run_model()

        # Check if a bound is violated, if not we simply end the process.
        if prob_max_thrust_power_limit.get_val("opr") < prob_max_thrust_power_limit.get_val(
            "opr_limit"
        ):
            max_thrust = prob_max_thrust_power_limit.get_val("required_thrust", units="N")
            self._add_to_max_thrust_cache(cache_key, max_thrust[0])
            return max_thrust

        prob_max_thrust_opr_limit = self.turboprop_max_thrust_opr_limit_problem

        prob_max_thrust_opr_limit.set_val("altitude", val=altitude, units="ft")
        prob_max_thrust_opr_limit.set_val("mach_0", val=mach)

        prob_max_thrust_opr_limit.run_model()

        # Check if a bound is violated, if not we simply end the process. Rather we will simply
        # check that the expected bound is reached
        if prob_max_thrust_opr_limit.get_val(
            "total_temperature_45", units="degK"
        ) < prob_max_thrust_opr_limit.get_val("itt_limit", units="degK"):

            max_thrust = prob_max_thrust_opr_limit.get_val("required_thrust", units="N")
            self._add_to_max_thrust_cache(cache_key, max_thrust[0])
            return max_thrust

        prob_max_thrust_itt_limit = self.turboprop_max_thrust_itt_limit_problem

        prob_max_thrust_itt_limit.set_val("altitude", val=altitude, units="ft")
        prob_max_thrust_itt_limit.set_val("mach_0", val=mach)

        prob_max_thrust_itt_limit.run_model()

        # Check if a bound is violated, if not we simply end the process. Rather we will simply
        # check that the expected bound is reached
        if prob_max_thrust_itt_limit.get_val(
            "propeller_thrust", units="N"
        ) < prob_max_thrust_itt_limit.get_val("propeller_max_thrust", units="N"):

            max_thrust = prob_max_thrust_itt_limit.get_val("required_thrust", units="N")
            self._add_to_max_thrust_cache(cache_key, max_thrust[0])
            return max_thrust

        prob_max_thrust_propeller_thrust_limit = (
            self.turboprop_max_thrust_propeller_thrust_limit_problem
        )

        prob_max_thrust_propeller_thrust_limit.set_val("altitude", val=altitude, units="ft")
        prob_max_thrust_propeller_thrust_limit.set_val("mach_0", val=mach)

        prob_max_thrust_propeller_thrust_limit.run_model()

        max_thrust = prob_max_thrust_propeller_thrust_limit.get_val("required_thrust", units="N")
        self._add_to_max_thrust_cache(cache_key, max_thrust[0])
        return max_thrust

    def max_thrust(
        self,
        atmosphere: Atmosphere,
    ) -> np.ndarray:
        """
        Computation of maximum thrust either due to propeller thrust limit or turboprop max power.

        :param atmosphere: Atmosphere instance at intended altitude (should be <=20km)
        :return: maximum thrust (in N)
        """
        altitude = atmosphere.get_altitude(altitude_in_feet=True)

        if isinstance(altitude, float):  # Calculate for float
            thrust_max_global = self._max_thrust(altitude, atmosphere.mach)
            if isinstance(thrust_max_global, float):
                thrust_max_global = np.array([thrust_max_global])

        else:  # Calculate for array
            thrust_max_global = np.zeros_like(altitude)
            for idx in range(np.size(altitude)):
                thrust_max_global[idx] = self._max_thrust(altitude[idx], atmosphere.mach[idx])

        return thrust_max_global

    def _fuel_consumed(
        self, altitude: float, mach: float, thrust_required: float
    ) -> Tuple[float, float]:
        """
        Computes the fuel consumed at the current flight point. Will first attempt to use the no
        ls problem because it is quicker, if it doesn't converge, compute using the ls problem.
        This function can't really be cached as it depends on the thrust which tends to never be
        equal from one point to another unlike speed and altitude

        :param altitude: altitude in ft
        :param mach: Mach number
        :param thrust_required: thrust required in N

        :return fuel_consumed: fuel consumed in kg/s
        :return shaft_power: shaft power in W
        """

        prob_fuel_consumed = self.turboprop_fuel_problem

        prob_fuel_consumed.set_val("altitude", val=altitude, units="ft")
        prob_fuel_consumed.set_val("mach_0", val=mach)
        prob_fuel_consumed.set_val("required_thrust", val=thrust_required, units="N")

        prob_fuel_consumed.run_model()

        if prob_fuel_consumed.model.nonlinear_solver._iter_count < MAX_ITER_NO_LS_PROBLEM:
            return (
                prob_fuel_consumed.get_val("fuel_mass_flow", units="kg/s")[0],
                prob_fuel_consumed.get_val("shaft_power", units="W")[0],
            )

        # From my tests it is rarely used but better safe than sorry
        prob_fuel_consumed_ls = self.turboprop_fuel_problem_ls

        prob_fuel_consumed_ls.set_val("altitude", val=altitude, units="ft")
        prob_fuel_consumed_ls.set_val("mach_0", val=mach)
        prob_fuel_consumed_ls.set_val("required_thrust", val=thrust_required, units="N")

        prob_fuel_consumed_ls.run_model()

        return (
            prob_fuel_consumed_ls.get_val("fuel_mass_flow", units="kg/s")[0],
            prob_fuel_consumed_ls.get_val("shaft_power", units="W")[0],
        )

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

        propeller_efficiency_SL = RectBivariateSpline(
            self.thrust_SL,
            self.speed_SL,
            self.efficiency_SL.T * self.effective_efficiency_ls,  # Include the efficiency loss
            # in here
        )
        propeller_efficiency_CL = RectBivariateSpline(
            self.thrust_CL,
            self.speed_CL,
            self.efficiency_CL.T * self.effective_efficiency_cruise,  # Include the efficiency loss
            # in here
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
        if np.size(thrust) == 1:  # calculate for float
            lower_bound = float(propeller_efficiency_SL(thrust_interp_SL, installed_airspeed))
            upper_bound = float(propeller_efficiency_CL(thrust_interp_CL, installed_airspeed))
            altitude = atmosphere.get_altitude(altitude_in_feet=False)
            propeller_efficiency = np.interp(
                altitude, [0.0, self.cruise_altitude_propeller], [lower_bound, upper_bound]
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
            (1 + 0.144 * mach ** 2) ** 0.65 * (np.log10(reynolds)) ** 2.58
        )  # 100% turbulent
        fineness = self.nacelle.length / np.sqrt(
            4 * self.nacelle.height * self.nacelle.width / np.pi
        )
        ff_nac = 1.0 + 0.35 / fineness  # Raymer (seen in Gudmundsson)
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
