"""Estimation of speed/load factors for aircraft design."""
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

import logging
import math
import warnings

import numpy as np
import openmdao.api as om
import scipy.optimize as optimize
from scipy import interpolate
from scipy.constants import g, knot, foot, lbf

# noinspection PyProtectedMember
from fastoad.module_management._bundle_loader import BundleLoader
from fastoad.constants import EngineSetting
import fastoad.api as oad

from stdatm import Atmosphere

from ..constants import SUBMODEL_VH

DOMAIN_PTS_NB = 19  # number of (V,n) calculated for the flight domain

_LOGGER = logging.getLogger(__name__)

oad.RegisterSubmodel.active_models[
    SUBMODEL_VH
] = "fastga.submodel.aerodynamics.aircraft.max_level_speed.legacy"


class ComputeVNAndVH(om.Group):
    """Group containing the computation of the V_h and the V-n diagram"""

    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):
        propulsion_option = {"propulsion_id": self.options["propulsion_id"]}
        self.add_subsystem(
            "compute_vh",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_VH, options=propulsion_option),
            promotes=["*"],
        )
        self.add_subsystem(
            "compute_vn_diagram",
            ComputeVN(),
            promotes=["*"],
        )


@oad.RegisterSubmodel(SUBMODEL_VH, "fastga.submodel.aerodynamics.aircraft.max_level_speed.legacy")
class ComputeVh(om.ExplicitComponent):
    """
    Computes the maximum level velocity of the aircraft at sea level

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._engine_wrapper = None

    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):
        self._engine_wrapper = BundleLoader().instantiate_component(self.options["propulsion_id"])
        self._engine_wrapper.setup(self)

        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="kg")
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:aerodynamics:aircraft:cruise:CD0", val=np.nan)
        self.add_input("data:aerodynamics:wing:cruise:induced_drag_coefficient", val=np.nan)

        self.add_output("data:TLAR:v_max_sl", units="m/s")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        # The maximum Sea Level flight velocity is computed using a method which finds for which
        # speed the thrust required for flight (drag) is equal to the thrust available

        _LOGGER.info("Entering load factors computation")

        design_mass = inputs["data:weight:aircraft:MTOW"]
        v_h = self.max_speed(inputs, 0.0, design_mass)

        outputs["data:TLAR:v_max_sl"] = v_h

    def max_speed(self, inputs, altitude, mass):
        # noinspection PyTypeChecker
        roots = optimize.fsolve(self.delta_axial_load, 300.0, args=(inputs, altitude, mass))[0]

        return np.max(roots[roots > 0.0])

    def delta_axial_load(self, air_speed, inputs, altitude, mass):
        propulsion_model = self._engine_wrapper.get_model(inputs)
        wing_area = inputs["data:geometry:wing:area"]
        cd0 = inputs["data:aerodynamics:aircraft:cruise:CD0"]
        coeff_k = inputs["data:aerodynamics:wing:cruise:induced_drag_coefficient"]

        # Get the available thrust from propulsion system
        atm = Atmosphere(altitude, altitude_in_feet=False)
        flight_point = oad.FlightPoint(
            mach=air_speed / atm.speed_of_sound,
            altitude=altitude,
            engine_setting=EngineSetting.TAKEOFF,
            thrust_rate=1.0,
        )
        propulsion_model.compute_flight_points(flight_point)
        thrust = float(flight_point.thrust)

        # TODO: Change to use the Equilibrium computation
        # Get the necessary thrust to overcome
        c_l = (mass * g) / (0.5 * atm.density * wing_area * air_speed ** 2.0)
        c_d = cd0 + coeff_k * c_l ** 2.0
        drag = 0.5 * atm.density * wing_area * c_d * air_speed ** 2.0

        return thrust - drag


class ComputeVN(om.ExplicitComponent):
    """
    Computes the load diagram of the aircraft.

    Based on the methodology presented in :cite:`roskampart5:1985` adapted with the
    certifications of :cite:`EASA:cs23` and :cite:`ASTM:F3116`, available at :
    - https://www.easa.europa.eu/sites/default/files/dfu/CS-23%20Amendment%204.pdf
    - https://www.astm.org/Standards/F3116.htm
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kts_to_ms = knot  # Converting from knots to meters per seconds
        self.ft_to_m = foot  # Converting from feet to meters
        self.lbf_to_N = lbf  # Converting from pound force to Newtons

    def setup(self):

        self.add_input("data:TLAR:category", val=3.0)
        self.add_input("data:TLAR:level", val=2.0)
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:chord", val=np.nan, units="m")
        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="kg")
        self.add_input("data:weight:aircraft:MZFW", val=np.nan, units="kg")
        self.add_input("data:TLAR:v_max_sl", val=np.nan, units="m/s")
        self.add_input("data:aerodynamics:aircraft:landing:CL_max", val=np.nan)
        self.add_input("data:aerodynamics:wing:low_speed:CL_max_clean", val=np.nan)
        self.add_input("data:aerodynamics:wing:low_speed:CL_min_clean", val=np.nan)
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
        self.add_input("data:TLAR:v_cruise", val=np.nan, units="m/s")
        self.add_input("data:mission:sizing:main_route:cruise:altitude", val=np.nan, units="m")

        self.add_output(
            "data:mission:sizing:cs23:flight_domain:mtow:velocity", units="m/s", shape=DOMAIN_PTS_NB
        )
        self.add_output(
            "data:mission:sizing:cs23:flight_domain:mtow:load_factor", shape=DOMAIN_PTS_NB
        )

        self.add_output(
            "data:mission:sizing:cs23:flight_domain:mzfw:velocity", units="m/s", shape=DOMAIN_PTS_NB
        )
        self.add_output(
            "data:mission:sizing:cs23:flight_domain:mzfw:load_factor", shape=DOMAIN_PTS_NB
        )

        self.declare_partials("*", "*", method="fd")

    def check_config(self, logger):
        # let void to avoid logger error on "The command cannot be empty"
        pass

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        v_tas = inputs["data:TLAR:v_cruise"]
        cruise_altitude = inputs["data:mission:sizing:main_route:cruise:altitude"]
        mtow = inputs["data:weight:aircraft:MTOW"]
        mzfw = inputs["data:weight:aircraft:MTOW"]

        atm = Atmosphere(cruise_altitude, altitude_in_feet=False)
        atm.true_airspeed = v_tas
        design_vc = atm.equivalent_airspeed

        velocity_array_mtow, load_factor_array_mtow, _ = self.flight_domain(
            inputs,
            mtow,
            cruise_altitude,
            design_vc,
            design_n_ps=0.0,
            design_n_ng=0.0,
        )

        if DOMAIN_PTS_NB < len(velocity_array_mtow):
            velocity_array_mtow = velocity_array_mtow[0 : DOMAIN_PTS_NB - 1]
            load_factor_array_mtow = load_factor_array_mtow[0 : DOMAIN_PTS_NB - 1]
            warnings.warn(
                "Defined maximum stored domain points at the beginning of the file exceeded!"
            )
        else:
            additional_zeros = list(np.zeros(DOMAIN_PTS_NB - len(velocity_array_mtow)))
            velocity_array_mtow.extend(additional_zeros)
            load_factor_array_mtow.extend(additional_zeros)

        outputs["data:mission:sizing:cs23:flight_domain:mtow:velocity"] = np.array(
            velocity_array_mtow
        )
        outputs["data:mission:sizing:cs23:flight_domain:mtow:load_factor"] = np.array(
            load_factor_array_mtow
        )

        velocity_array_mzfw, load_factor_array_mzfw, _ = self.flight_domain(
            inputs,
            mzfw,
            cruise_altitude,
            design_vc,
            design_n_ps=0.0,
            design_n_ng=0.0,
        )

        if DOMAIN_PTS_NB < len(velocity_array_mzfw):
            velocity_array_mzfw = velocity_array_mzfw[0 : DOMAIN_PTS_NB - 1]
            load_factor_array_mzfw = load_factor_array_mzfw[0 : DOMAIN_PTS_NB - 1]
            warnings.warn(
                "Defined maximum stored domain points at the beginning of the file exceeded!"
            )
        else:
            additional_zeros = list(np.zeros(DOMAIN_PTS_NB - len(velocity_array_mzfw)))
            velocity_array_mzfw.extend(additional_zeros)
            load_factor_array_mzfw.extend(additional_zeros)

        outputs["data:mission:sizing:cs23:flight_domain:mzfw:velocity"] = np.array(
            velocity_array_mzfw
        )
        outputs["data:mission:sizing:cs23:flight_domain:mzfw:load_factor"] = np.array(
            load_factor_array_mzfw
        )

    # noinspection PyUnusedLocal
    def flight_domain(self, inputs, mass, altitude, design_vc, design_n_ps=0.0, design_n_ng=0.0):
        """
        Function that computes the flight domain of the aircraft represented in the inputs for a
        given mass, altitude, cruise equivalent airspeed and design load factors

        @param inputs: a dictionary containing the properties of the aircraft
        @param mass: the mass for which we want to compute the flight domain
        @param altitude: the altitude at which we want to compute the flight domain
        @param design_vc: the cruise equivalent airspeed
        @param design_n_ps: the positive design load factor, will replace the maneuver load factor
        if higher than it
        @param design_n_ng: the negative design load factor, will replace the maneuver load factor
        if lower than it
        @return velocity_array: an array containing the characteristic speeds necessary to draw
        the flight domain stored as [Vs_1g_ps, Vs_1g_ng, V_a_ps (maneuver diagram), V_a_ng (maneuver
        diagram), V_a_ps (gust, 0 if same as maneuver), V_a_ng (gust, 0 if same as maneuver), V_c,
        V_c, V_c, V_d, V_d, V_d, V_d, V_ne, V_no, V_mg (for commuter), Vs_1g_fe, V_a_fe, V_fe]
        @return load_factor_array: an array containing the load factors necessary to draw the flight
        domain stored as [1.0, -1.0, n_lim_ps (maneuver diagram), n_lim_ng (maneuver diagram),
        n_a_ps (gust, 0 if same as maneuver), n_a_ng (gust, 0 if same as maneuver), n_lim_ng,
        n_c_ps (maneuver or gust, whichever is greatest), n_c_ng ( maneuver or gust, whichever is
        greatest), n_lim_ps, 0.0, n_d_ps (maneuver or gust, whichever is greatest), n_d_ng
        (maneuver or gust, whichever is greatest), 0.0, 0.0, n_v_mg, 1.0, n_fe, n_fe]
        @return conditions: an array containing the conditions at which the diagram was computed
        """

        # Get necessary inputs
        wing_area = inputs["data:geometry:wing:area"]
        mtow = inputs["data:weight:aircraft:MTOW"]
        category = inputs[
            "data:TLAR:category"
        ]  # Aerobatic = 1.0, Utility = 2.0, Normal = 3.0, Commuter = 4.0
        level = inputs["data:TLAR:level"]
        vh = inputs["data:TLAR:v_max_sl"]
        root_chord = inputs["data:geometry:wing:root:chord"]
        tip_chord = inputs["data:geometry:wing:tip:chord"]
        cl_max_flaps = inputs["data:aerodynamics:aircraft:landing:CL_max"]
        cl_max = inputs["data:aerodynamics:wing:low_speed:CL_max_clean"]
        cl_min = inputs["data:aerodynamics:wing:low_speed:CL_min_clean"]
        mean_chord = (root_chord + tip_chord) / 2.0
        atm_0 = Atmosphere(0.0)
        atm = Atmosphere(altitude, altitude_in_feet=False)

        # Initialise the lists in which we will store the data
        velocity_array = []
        load_factor_array = []

        # For some of the correlation presented in the regulation, we need to convert the data
        # of the airplane to imperial units
        weight_lbf = (mass * g) / self.lbf_to_N
        mtow_lbf = (mtow * g) / self.lbf_to_N
        wing_area_sft = wing_area / (self.ft_to_m ** 2.0)
        mtow_loading_psf = mtow_lbf / wing_area_sft  # [lbf/ft**2]

        # We can now start computing the values of the different air-speeds given in the regulation
        # as well as the load factors. We will here make the choice to stick with the limits given
        # in the certifications even though they sometimes allow to choose design speeds and loads
        # over the values written in the documents.

        # Lets start by computing the 1g/-1g stall speeds using the usual formulations
        vs_1g_ps = math.sqrt((2.0 * mass * g) / (atm_0.density * wing_area * cl_max))  # [m/s]
        vs_1g_ng = math.sqrt((2.0 * mass * g) / (atm_0.density * wing_area * abs(cl_min)))  # [m/s]
        velocity_array.append(float(vs_1g_ps))
        load_factor_array.append(1.0)
        velocity_array.append(float(vs_1g_ng))
        load_factor_array.append(-1.0)

        # As we will consider all the calculated speed to be Vs_1g_ps < V < 1.4*Vh, we will
        # compute cl_alpha for N points equally spaced on log scale (to take into account the
        # high non-linearity effect). If the option is not selected, we will only consider
        # low_speed and cruise cl_alpha points and consider a square regression between both.

        mach_interp = inputs["data:aerodynamics:aircraft:mach_interpolation:mach_vector"]
        v_interp = []
        for mach in mach_interp:
            v_interp.append(float(mach * atm.speed_of_sound))
        cl_alpha_interp = inputs["data:aerodynamics:aircraft:mach_interpolation:CL_alpha_vector"]
        cl_alpha_fct = interpolate.interp1d(
            v_interp, cl_alpha_interp, fill_value="extrapolate", kind="quadratic"
        )

        # We will now establish the minimum limit maneuvering load factors outside of gust load
        # factors. Th designer can take higher load factor if he so wish. As will later be done
        # for the the cruising speed, we will simply ensure that the designer choice agrees with
        # certifications The limit load factor can be found in section CS 23.337 (a) and (b)
        # https://www.easa.europa.eu/sites/default/files/dfu/CS-23%20Amendment%204.pdf
        # https://www.astm.org/Standards/F3116.htm

        if category == 1.0:
            n_lim_1 = 6.0  # For aerobatic GA aircraft
        else:
            n_lim_1 = 3.80  # For non aerobatic GA aircraft
        n_lim_2 = 2.1 + 24000.0 / (mtow_lbf + 10000.0)  # CS 23.337 (a)
        n_lim_ps_min = min(n_lim_1, n_lim_2)  # CS 23.337 (a)
        n_lim_ps = max(n_lim_ps_min, design_n_ps)

        if category == 1.0:
            n_lim_ng_max = -0.5 * n_lim_ps  # CS 23.337 (b)
        else:
            n_lim_ng_max = -0.4 * n_lim_ps  # CS 23.337 (b)
        n_lim_ng = min(n_lim_ng_max, design_n_ng)

        load_factor_array.append(float(n_lim_ps))
        load_factor_array.append(float(n_lim_ng))

        # Starting from there, we need to compute the gust lines as it can have an impact on the
        # choice of the maneuvering speed. We will also compute the maximum intensity gust line
        # for later use but keep in mind that this is specific for commuter or level 4 aircraft
        # The values used to compute the gust lines can be found in CS 23.341
        # https://www.easa.europa.eu/sites/default/files/dfu/CS-23%20Amendment%204.pdf

        # We first compute the gust velocities as presented in CS 23.333 (c), for now, we don't
        # take into account the case of the commuter nor do we implement the reduction of gust
        # intensity with the location of the gust center

        if altitude <= 20000.0:
            u_de_vc = 50.0  # [ft/s]
            u_de_vd = 25.0  # [ft/s]
            u_de_vmg = 66.0  # [ft/s]
        elif 20000.0 < altitude < 50000.0:
            u_de_vc = 66.7 - 0.000833 * altitude  # [ft/s]
            u_de_vd = 33.4 - 0.000417 * altitude  # [ft/s]
            u_de_vmg = 84.7 - 0.000933 * altitude  # [ft/s]
        else:
            u_de_vc = 25.0  # [ft/s]
            u_de_vd = 12.5  # [ft/s]
            u_de_vmg = 38.0  # [ft/s]

        # Let us define aeroplane mass ratio formula and alleviation factor formula
        def mu_g(x):
            return (2.0 * mass * g / wing_area) / (
                atm.density * mean_chord * x * g
            )  # [x = cl_alpha]

        def k_g(x):
            return (0.88 * x) / (5.3 + x)  # [x = mu_g]

        # Now, define the gust function
        def load_factor_gust_p(u_de_v, x):
            return float(
                1.0
                + k_g(mu_g(cl_alpha_fct(x)))
                * atm_0.density
                * u_de_v
                * self.ft_to_m
                * x
                * cl_alpha_fct(x)
                / (2.0 * weight_lbf / wing_area_sft * self.lbf_to_N / self.ft_to_m ** 2)
            )

        def load_factor_gust_n(u_de_v, x):
            return float(
                1
                - k_g(mu_g(cl_alpha_fct(x)))
                * atm_0.density
                * u_de_v
                * self.ft_to_m
                * x
                * cl_alpha_fct(x)
                / (2.0 * weight_lbf / wing_area_sft * self.lbf_to_N / self.ft_to_m ** 2)
            )

        def load_factor_stall_p(x):
            return (x / vs_1g_ps) ** 2.0

        def load_factor_stall_n(x):
            return -((x / vs_1g_ng) ** 2.0)

        # We can now go back to the computation of the maneuvering speeds, we will first compute
        # it "traditionally" and should we find out that the line limited by the Cl max is under
        # the gust line, we will adjust it (see Step 10. of section 16.4.1 of (1)). As for the
        # traditional computation they can be found in CS 23.335 (c)
        # https://www.easa.europa.eu/sites/default/files/dfu/CS-23%20Amendment%204.pdf
        # https://www.astm.org/Standards/F3116.htm

        vma_ps = vs_1g_ps * math.sqrt(n_lim_ps)  # [m/s]
        vma_ng = vs_1g_ng * math.sqrt(abs(n_lim_ng))  # [m/s]
        velocity_array.append(float(vma_ps))
        velocity_array.append(float(vma_ng))

        # We now need to check if we are in the aforementioned case (usually happens for low
        # design wing loading aircraft and/or mission wing loading)

        n_ma_ps = load_factor_gust_p(u_de_vc, vma_ps)

        if n_ma_ps > n_lim_ps:
            # In case the gust line load factor is above the maneuvering load factor, we need to
            # solve the difference between both curve to be 0.0 to find intersect
            def delta_maneuver_pos(x):
                return load_factor_gust_p(u_de_vc, x) - load_factor_stall_p(x)

            vma_ps = max(optimize.fsolve(delta_maneuver_pos, np.array(1000.0)))
            n_ma_ps = load_factor_gust_p(u_de_vc, vma_ps)  # [-]
            velocity_array.append(float(vma_ps))
            load_factor_array.append(float(n_ma_ps))
        else:
            velocity_array.append(0.0)
            load_factor_array.append(0.0)

        # We now need to do the same thing for the negative maneuvering speed

        n_ma_ng = load_factor_gust_n(u_de_vc, vma_ng)  # [-]

        if n_ma_ng < n_lim_ng:
            # In case the gust line load factor is above the maneuvering load factor, we need to
            # solve the difference between both curve to be 0.0 to find intersect
            def delta_maneuver_neg(x):
                return load_factor_gust_n(u_de_vc, x) - load_factor_stall_n(x)

            vma_ng = max(optimize.fsolve(delta_maneuver_neg, np.array(1000.0)))
            n_ma_ng = load_factor_gust_n(u_de_vc, vma_ng)  # [-]
            velocity_array.append(float(vma_ng))
            load_factor_array.append(float(n_ma_ng))
        else:
            velocity_array.append(0.0)
            load_factor_array.append(0.0)

        # For the cruise velocity, things will be different since it is an entry choice. As such
        # we will simply check that it complies with the values given in the certification papers
        # and re-adjust it if necessary. For airplane certified for aerobatics, the coefficient
        # in front of the wing loading in psf is slightly different than for normal aircraft but
        # for either case it becomes 28.6 at wing loading superior to 100 psf Values and
        # methodology used can be found in CS 23.335 (a)
        # https://www.easa.europa.eu/sites/default/files/dfu/CS-23%20Amendment%204.pdf
        # https://www.astm.org/Standards/F3116.htm

        if category == 1.0:
            if mtow_loading_psf < 20.0:
                k_c = 36.0
            elif mtow_loading_psf < 100.0:
                # Linear variation from 33.0 to 28.6
                k_c = 36.0 + (mtow_loading_psf - 20.0) * (28.6 - 36.0) / (100.0 - 20.0)
            else:
                k_c = 28.6
        else:
            if mtow_loading_psf < 20.0:
                k_c = 33.0
            elif mtow_loading_psf < 100.0:
                # Linear variation from 33.0 to 28.6
                k_c = 33.0 + (mtow_loading_psf - 20.0) * (28.6 - 33.0) / (100.0 - 20.0)
            else:
                k_c = 28.6

        vc_min_1 = k_c * math.sqrt(weight_lbf / wing_area_sft) * self.kts_to_ms  # [m/s]

        # This second constraint rather refers to the paragraph on maneuvering speeds,
        # which needs to be chosen so that they are smaller than cruising speeds
        vc_min_2 = vma_ps  # [m/s]
        vc_min = max(vc_min_1, vc_min_2)  # [m/s]

        # The certifications specifies that Vc need not be more than 0.9 Vh so we will simply
        # take the minimum value between the Vc_min and this value

        vc_min_fin = min(vc_min, 0.9 * vh)  # [m/s]

        # The constraint regarding the maximum velocity for cruise does not appear in the
        # certifications but from a physics point of view we can easily infer that the cruise
        # speed will never be greater than the maximum level velocity at sea level hence

        vc = max(min(design_vc, vh), vc_min_fin)  # [m/s]
        velocity_array.append(float(vc))
        load_factor_array.append(float(n_lim_ng))

        # Lets now look at the load factors associated with the Vc, since it is here that the
        # greatest load factors can appear

        n_vc_ps = max(load_factor_gust_p(u_de_vc, vc), n_lim_ps)  # [-]
        n_vc_ng = min(load_factor_gust_n(u_de_vc, vc), n_lim_ng)  # [-]

        velocity_array.append(float(vc))
        load_factor_array.append(float(n_vc_ps))
        velocity_array.append(float(vc))
        load_factor_array.append(float(n_vc_ng))

        # We now compute the diving speed, methods are described in CS 23.335 (b). We will take
        # the minimum diving speed allowable as our design diving speed. We need to keep in mind
        # that this speed could be greater if the designer was willing to show that the structure
        # holds for the wanted Vd. For airplane that needs to be certified for aerobatics use,
        # the factor between Vd_min and Vc_min is slightly different, but they both become 1.35
        # for wing loading higher than 100 psf
        # https://www.easa.europa.eu/sites/default/files/dfu/CS-23%20Amendment%204.pdf
        # https://www.astm.org/Standards/F3116.htm

        vd_min_1 = 1.25 * vc  # [m/s]

        if category == 1.0:
            if mtow_loading_psf < 20.0:
                k_d = 1.55
            elif mtow_loading_psf < 100.0:
                # Linear variation from 1.55 to 1.35
                k_d = 1.55 + (mtow_loading_psf - 20.0) * (1.35 - 1.55) / (100.0 - 20.0)
            else:
                k_d = 1.35
        elif category == 2.0:
            if mtow_loading_psf < 20.0:
                k_d = 1.50
            elif mtow_loading_psf < 100.0:
                # Linear variation from 1.5 to 1.35
                k_d = 1.50 + (mtow_loading_psf - 20.0) * (1.35 - 1.50) / (100.0 - 20.0)
            else:
                k_d = 1.35
        else:
            if mtow_loading_psf < 20.0:
                k_d = 1.4
            elif mtow_loading_psf < 100.0:
                # Linear variation from 1.4 to 1.35
                k_d = 1.4 + (mtow_loading_psf - 20.0) * (1.35 - 1.4) / (100.0 - 20.0)
            else:
                k_d = 1.35

        vd_min_2 = k_d * vc_min_fin  # [m/s]
        vd = max(vd_min_1, vd_min_2)  # [m/s]

        velocity_array.append(float(vd))
        load_factor_array.append(float(n_lim_ps))
        velocity_array.append(float(vd))
        load_factor_array.append(0.0)

        # Similarly to what was done for the design cruising speed we will explore the load
        # factors associated with the diving speed since gusts are likely to broaden the flight
        # domain around these points

        n_vd_ps = load_factor_gust_p(u_de_vd, vd)  # [-]

        # For the negative load factor at the diving speed, it seems that for non_aerobatic
        # airplanes, it is always sized according to the gust lines, regardless of the negative
        # design load factor. For aerobatic airplanes however, it seems as if it is sized for a
        # greater value (more negative) but it does not look to be equal to the negative diving
        # factor as can be seen in figure 16-13 of (1). No information was found for the location
        # of this precises point, so the choice was made to take it as the negative design load
        # factor or the load factor given by the gust, whichever is the greatest (most negative).
        # This way, for non aerobatic airplane, we ensure to be conservative.

        n_vd_ng = load_factor_gust_n(u_de_vd, vd)  # [-]

        velocity_array.append(float(vd))
        load_factor_array.append(float(n_vd_ps))
        velocity_array.append(float(vd))
        load_factor_array.append(float(n_vd_ng))

        # We have now calculated all the velocities need to plot the flight domain. For the sake
        # of thoroughness we will also compute the maximal structural cruising speed and cruise
        # never-exceed speed. The computation for these two can be found in CS 23.1505
        # https://www.easa.europa.eu/sites/default/files/dfu/CS-23%20Amendment%204.pdf
        # https://www.astm.org/Standards/F3116.htm

        # Let us start, as presented in the certifications papers with the never-exceed speed
        # Since we made the choice to take the Vd as the minimum value allowed by certifications,
        # the V_ne will have a fixed value and not a range as one would have expect. Indeed if Vd
        # = Vd_min and since V_ne has to be greater or equal to 0.9 x Vd_min and smaller or equal
        # to 0.9 x Vd, V_ne will be equal to 0.9 Vd. For future implementations, it should be
        # noted that this section will need to be rewritten should the Vd become a design
        # parameter like what was made on Vc. Additionally the effect of buffeting which serves
        # as an additional upper limit is not included but should be taken into account in
        # detailed analysis phases

        v_ne = 0.9 * vd  # [m/s]

        velocity_array.append(float(v_ne))
        load_factor_array.append(0.0)

        v_no_min = vc_min  # [m/s]
        v_no_max = min(vc, 0.89 * v_ne)  # [m/s]

        # Again we need to make a choice for this speed : what value would be retained. We will
        # take the highest speed acceptable for certification, i.e

        v_no = max(v_no_min, v_no_max)  # [m/s]

        velocity_array.append(float(v_no))
        load_factor_array.append(0.0)

        # One additional velocity needs to be computed if we are talking about commuter aircraft.
        # It is the maximum gust intensity velocity. Due to the way we are returning the values,
        # even if we are not investigating a commuter aircraft we need to return a value for Vmg
        # so we will put it to 0.0. If we are investigating a commuter aircraft, we will compute
        # it according ot the guidelines from CS 23.335 (d)
        # https://www.easa.europa.eu/sites/default/files/dfu/CS-23%20Amendment%204.pdf
        # https://www.astm.org/Standards/F3116.htm We decided to put this computation here as we
        # may need the gust load factor in cruise conditions for one of the possible candidates
        # for the Vmg. While writing this program, the writer realized they were no paragraph
        # that impeach the Vc from being at a value such that one one of the conditions for the
        # minimum speed was above the Vc creating a problem with point (2). This case may however
        # never appear in practice as it would suppose that the Vc chosen is above the stall line
        # which is more than certainly avoided by the correlation between Vc_min and W/S in CS
        # 23.335 (a)

        if (level == 4.0) or (category == 4.0):

            # We first need to compute the intersection of the stall line with the gust line
            # given by the gust of maximum intensity. Similar calculation were already done in
            # case the maneuvering speed is dictated by the Vc gust line so the computation will
            # be very similar
            def delta_max_gust_pos(x):
                return load_factor_gust_p(u_de_vmg, x) - load_factor_stall_p(x)

            vmg_min_1 = max(optimize.fsolve(delta_max_gust_pos, np.array(1000.0)))

            # The second candidate for the Vmg is given by the stall speed and the load factor at
            # the cruise speed
            vmg_min_2 = vs_1g_ps * math.sqrt(load_factor_gust_p(u_de_vc, vc))  # [m/s]
            vmg = min([vmg_min_1, vmg_min_2])  # [m/s]

            # As for the computation of the associated load factor, no source were found for any
            # formula or hint as to its computation. It can however be guessed that depending on
            # the minimum value found above, it will either be on the stall line or at the
            # maximum design load factor

            if vmg == vmg_min_1:  # On the gust line
                n_vmg = load_factor_gust_p(u_de_vmg, vmg_min_1)  # [-]
            else:
                n_vmg = n_vc_ps  # [-]

        else:
            vmg = 0.0  # [m/s]
            n_vmg = 0.0

        velocity_array.append(float(vmg))
        load_factor_array.append(float(n_vmg))

        # Let us now look at the flight domain in the flap extended configuration. For the
        # computation of these speeds and load factors, we will use the formula provided in CS
        # 23.1511
        # https://www.easa.europa.eu/sites/default/files/dfu/CS-23%20Amendment%204.pdf
        # https://www.astm.org/Standards/F3116.htm

        # For the computation of the Vfe CS 23.1511, refers to CS 23.345 but there only seems to
        # be a requirement for the lowest the Vfe can be, hence we will take this speed as the
        # Vfe. As for the load factors that are prescribed we will use the guidelines provided in
        # CS 23.345 (b)

        # Let us start by computing the Vfe
        vs_fe_1g_ps = math.sqrt(
            (2.0 * mass * g) / (atm_0.density * wing_area * cl_max_flaps)
        )  # [m/s]
        vfe_min_1 = 1.4 * vs_1g_ps  # [m/s]
        vfe_min_2 = 1.8 * vs_fe_1g_ps  # [m/s]
        vfe_min = max(vfe_min_1, vfe_min_2)  # [m/s]
        vfe = vfe_min  # [m/s]

        velocity_array.append(float(vs_fe_1g_ps))
        load_factor_array.append(1.0)

        # We can then move on to the computation of the load limitation of the flapped flight
        # domain, which must be equal to either a constant load factor of 2 or a load factor
        # dictated by a gust of 25 fps. Also since the use of flaps is limited to take-off,
        # approach and landing, we will use the SL density and a constant gust velocity

        u_de_fe = 25.0  # [ft/s]
        n_lim_ps_fe = 2.0
        n_vfe = max(n_lim_ps_fe, load_factor_gust_n(u_de_fe, vfe))

        velocity_array.append(float(vs_fe_1g_ps * math.sqrt(n_vfe)))
        load_factor_array.append(float(n_vfe))
        velocity_array.append(float(vfe))
        load_factor_array.append(float(n_vfe))

        # We also store the conditions in which the values were computed so that we can easily
        # access them when drawing the flight domains

        conditions = [mass, altitude]

        return velocity_array, load_factor_array, conditions
