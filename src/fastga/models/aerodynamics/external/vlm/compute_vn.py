"""
    Estimation of speed/load factors for aircraft design
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
import warnings
from scipy.constants import g
import scipy.optimize as optimize
from scipy import interpolate
from scipy.constants import knot, foot, lbf
import openmdao.api as om

# noinspection PyProtectedMember
from fastoad.module_management._bundle_loader import BundleLoader
from fastoad.model_base import Atmosphere, FlightPoint
from fastoad.constants import EngineSetting

from .vlm import VLMSimpleGeometry

from fastga.models.propulsion.fuel_propulsion.base import FuelEngineSet

INPUT_AOA = 10.0  # only one value given since calculation is done by default around 0.0!
MACH_NB_PTS = 5  # number of points for cl_alpha_aircraft fitting along mach axis
DOMAIN_PTS_NB = 19  # number of (V,n) calculated for the flight domain


class ComputeVNvlmNoVH(om.Group):

    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)
        self.options.declare("compute_cl_alpha", default=False, types=bool)

    def setup(self):
        self.add_subsystem("compute_vh", ComputeVh(propulsion_id=self.options["propulsion_id"]), promotes=["*"])
        self.add_subsystem("compute_vn_diagram",
                           ComputeVNvlm(compute_cl_alpha=self.options["compute_cl_alpha"]), promotes=["*"])


class ComputeVh(om.ExplicitComponent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._engine_wrapper = None

    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):
        self._engine_wrapper = BundleLoader().instantiate_component(self.options["propulsion_id"])
        self._engine_wrapper.setup(self)

        self.add_input("data:geometry:propulsion:count", np.nan)
        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="kg")
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:aerodynamics:aircraft:cruise:CD0", val=np.nan)
        self.add_input("data:aerodynamics:wing:cruise:induced_drag_coefficient", val=np.nan)

        self.add_output("data:TLAR:v_max_sl", units="kn")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        # The maximum Sea Level flight velocity is computed using a method which finds for which speed
        # the thrust required for flight (drag) is equal to the thrust available
        design_mass = inputs["data:weight:aircraft:MTOW"]
        Vh = self.max_speed(inputs, 0.0, design_mass)

        outputs["data:TLAR:v_max_sl"] = Vh


    def max_speed(self, inputs, altitude, mass):

        # noinspection PyTypeChecker
        roots = optimize.fsolve(
            self.delta_axial_load,
            300.0,
            args=(inputs, altitude, mass)
        )[0]

        return np.max(roots[roots > 0.0])


    def delta_axial_load(self, air_speed, inputs, altitude, mass):

        propulsion_model = FuelEngineSet(
            self._engine_wrapper.get_model(inputs), inputs["data:geometry:propulsion:count"]
        )
        wing_area = inputs["data:geometry:wing:area"]
        cd0 = inputs["data:aerodynamics:aircraft:cruise:CD0"]
        coef_k = inputs["data:aerodynamics:wing:cruise:induced_drag_coefficient"]

        # Get the available thrust from propulsion system
        atm = Atmosphere(altitude, altitude_in_feet=False)
        flight_point = FlightPoint(
            mach=air_speed/atm.speed_of_sound, altitude=altitude, engine_setting=EngineSetting.TAKEOFF,
            thrust_rate=1.0
        )
        propulsion_model.compute_flight_points(flight_point)
        thrust = float(flight_point.thrust)

        # Get the necessary thrust to overcome
        cl = (mass * g) / (0.5 * atm.density * wing_area * air_speed**2.0)
        cd = cd0 + coef_k * cl ** 2.0
        drag = 0.5 * atm.density * wing_area * cd * air_speed**2.0

        return thrust - drag


class ComputeVNvlm(VLMSimpleGeometry):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kts_to_ms = knot  # Converting from knots to meters per seconds
        self.ft_to_m = foot  # Converting from feet to meters
        self.lbf_to_N = lbf  # Converting from pound force to Newtons

    def initialize(self):
        super().initialize()
        self.options.declare("compute_cl_alpha", default=False, types=bool)
        
    def setup(self):
        super().setup()
        nans_array = np.full(MACH_NB_PTS + 1, np.nan)
        self.add_input("data:TLAR:category", val=3.0)
        self.add_input("data:TLAR:level", val=1.0)
        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="kg")
        self.add_input("data:TLAR:v_max_sl", val=np.nan, units="kn")
        self.add_input("data:aerodynamics:aircraft:landing:CL_max", val=np.nan)
        self.add_input("data:aerodynamics:wing:low_speed:CL_max_clean", val=np.nan)
        self.add_input("data:aerodynamics:wing:low_speed:CL_min_clean", val=np.nan)
        self.add_input("data:aerodynamics:aircraft:mach_interpolation:CL_alpha_vector", val=nans_array, units="rad**-1",
                       shape=MACH_NB_PTS + 1)
        self.add_input("data:aerodynamics:aircraft:mach_interpolation:mach_vector", val=nans_array,
                       shape=MACH_NB_PTS + 1)
        self.add_input("data:TLAR:v_cruise", val=np.nan, units="m/s")
        self.add_input("data:mission:sizing:main_route:cruise:altitude", val=np.nan, units="m")
        if not(self.options["compute_cl_alpha"]):
            self.add_input("data:aerodynamics:wing:low_speed:CL_alpha", val=np.nan)
            self.add_input("data:aerodynamics:wing:cruise:CL_alpha", val=np.nan)
            self.add_input("data:aerodynamics:horizontal_tail:low_speed:CL_alpha", val=np.nan)
            self.add_input("data:aerodynamics:horizontal_tail:cruise:CL_alpha", val=np.nan)
            self.add_input("data:TLAR:v_approach", val=np.nan, units="m/s")

        self.add_output("data:flight_domain:velocity", units="m/s", shape=DOMAIN_PTS_NB)
        self.add_output("data:flight_domain:load_factor", shape=DOMAIN_PTS_NB)
        
        self.declare_partials("*", "*", method="fd")

    def check_config(self, logger):
        # let void to avoid logger error on "The command cannot be empty"
        pass
    
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        v_tas = inputs["data:TLAR:v_cruise"]
        cruise_altitude = inputs["data:mission:sizing:main_route:cruise:altitude"]
        design_mass = inputs["data:weight:aircraft:MTOW"]

        atm = Atmosphere(cruise_altitude, altitude_in_feet=False)
        atm.true_airspeed = v_tas
        design_vc = atm.equivalent_airspeed
        velocity_array, load_factor_array, _ = self.flight_domain(inputs, outputs, design_mass, cruise_altitude,
                                                                  design_vc, design_n_ps=0.0, design_n_ng=0.0)

        if DOMAIN_PTS_NB < len(velocity_array):
            velocity_array = velocity_array[0:DOMAIN_PTS_NB-1]
            load_factor_array = load_factor_array[0:DOMAIN_PTS_NB-1]
            warnings.warn("Defined maximum stored domain points in fast compute_vn.py exceeded!")
        else:
            additional_zeros = list(np.zeros(DOMAIN_PTS_NB - len(velocity_array)))
            velocity_array.extend(additional_zeros)
            load_factor_array.extend(additional_zeros)

        outputs["data:flight_domain:velocity"] = np.array(velocity_array)
        outputs["data:flight_domain:load_factor"] = np.array(load_factor_array)

    # noinspection PyUnusedLocal
    def flight_domain(self, inputs, outputs, mass, altitude, design_vc, design_n_ps=0.0, design_n_ng=0.0):

        # Get necessary inputs
        wing_area = inputs["data:geometry:wing:area"]
        mtow = inputs["data:weight:aircraft:MTOW"]
        category = inputs["data:TLAR:category"]  # Aerobatic = 1.0, Utility = 2.0, Normal = 3.0, Commuter = 4.0
        level = inputs["data:TLAR:level"]
        Vh = inputs["data:TLAR:v_max_sl"]
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
        Vs_1g_ps = math.sqrt((2. * mass * g) / (atm_0.density * wing_area * cl_max))  # [m/s]
        Vs_1g_ng = math.sqrt((2. * mass * g) / (atm_0.density * wing_area * abs(cl_min)))  # [m/s]
        velocity_array.append(float(Vs_1g_ps))
        load_factor_array.append(1.0)
        velocity_array.append(float(Vs_1g_ng))
        load_factor_array.append(-1.0)


        # As we will consider all the calculated speed to be Vs_1g_ps < V < 1.4*Vh, we will compute cl_alpha for N
        # points equally spaced on log scale (to take into account the high non-linearity effect).
        # If the option is not selected, we will only consider low_speed and cruise cl_alpha points and consider a
        # square regression between both.

        if self.options["compute_cl_alpha"]:
            mach_interp = inputs["data:aerodynamics:aircraft:mach_interpolation:mach_vector"]
            v_interp = []
            for mach in mach_interp:
                v_interp.append(float(mach * atm.speed_of_sound))
            cl_alpha_interp = inputs["data:aerodynamics:aircraft:mach_interpolation:CL_alpha_vector"]
            cl_alpha_fct = interpolate.interp1d(v_interp, cl_alpha_interp, fill_value="extrapolate", kind="quadratic")
        else:
            v_interp = np.array([float(inputs["data:TLAR:v_approach"]), float(inputs["data:TLAR:v_cruise"])])
            cl_alpha_1 = float(
                inputs["data:aerodynamics:wing:low_speed:CL_alpha"]
                + inputs["data:aerodynamics:horizontal_tail:low_speed:CL_alpha"]
            )
            cl_alpha_2 = float(
                inputs["data:aerodynamics:wing:cruise:CL_alpha"]
                + inputs["data:aerodynamics:horizontal_tail:cruise:CL_alpha"]
            )
            cl_alpha_interp = np.array([cl_alpha_1, cl_alpha_2])
            cl_alpha_fct = interpolate.interp1d(v_interp, cl_alpha_interp, fill_value="extrapolate", kind="linear")


        # We will now establish the minimum limit maneuvering load factors outside of gust load
        # factors. Th designer can take higher load factor if he so wish. As will later be done for the
        # the cruising speed, we will simply ensure that the designer choice agrees with certifications
        # The limit load factor can be found in section CS 23.337 (a) and (b)
        # https://www.easa.europa.eu/sites/default/files/dfu/CS-23%20Amendment%204.pdf
        # https://www.astm.org/Standards/F3116.htm

        if category == 1.0:
            n_lim_1 = 6.0  # For aerobatic GA aircraft
        else:
            n_lim_1 = 3.80  # For non aerobatic GA aircraft
        n_lim_2 = 2.1 + 24000. / (mtow_lbf + 10000.)  # CS 23.337 (a)
        n_lim_ps_min = min(n_lim_1, n_lim_2)  # CS 23.337 (a)
        n_lim_ps = max(n_lim_ps_min, design_n_ps)

        if category == 1.0:
            n_lim_ng_max = - 0.5 * n_lim_ps  # CS 23.337 (b)
        else:
            n_lim_ng_max = - 0.4 * n_lim_ps  # CS 23.337 (b)
        n_lim_ng = min(n_lim_ng_max, design_n_ng)

        load_factor_array.append(float(n_lim_ps))
        load_factor_array.append(float(n_lim_ng))

        # Starting from there, we need to compute the gust lines as it can have an impact on the choice
        # of the maneuvering speed. We will also compute the maximum intensity gust line for later
        # use but keep in mind that this is specific for commuter or level 4 aircraft
        # The values used to compute the gust lines can be found in CS 23.341
        # https://www.easa.europa.eu/sites/default/files/dfu/CS-23%20Amendment%204.pdf

        # We first compute the gust velocities as presented in CS 23.333 (c), for now, we don't take into account
        # the case of the commuter nor do we implement the reduction of gust intensity with the location
        # of the gust center

        if altitude < 20000.0:
            U_de_Vc = 50.  # [ft/s]
            U_de_Vd = 25.  # [ft/s]
            U_de_Vmg = 66.  # [ft/s]
        elif 20000.0 < altitude < 50000.0:
            U_de_Vc = 66.7 - 0.000833 * altitude  # [ft/s]
            U_de_Vd = 33.4 - 0.000417 * altitude  # [ft/s]
            U_de_Vmg = 84.7 - 0.000933 * altitude  # [ft/s]
        else:
            U_de_Vc = 25.  # [ft/s]
            U_de_Vd = 12.5  # [ft/s]
            U_de_Vmg = 38.  # [ft/s]

        # Let us define aeroplane mass ratio formula and alleviation factor formula
        mu_g = lambda x: (2.0 * mass * g / wing_area) / (atm.density * mean_chord * x * g)  # [x = cl_alpha]
        K_g = lambda x: (0.88 * x) / (5.3 + x)  # [x = mu_g]
        # Now, define the gust function
        load_factor_gust_p = lambda u_de_v, x: float(
                1 + K_g(mu_g(cl_alpha_fct(x))) * atm_0.density * u_de_v * self.ft_to_m * x * cl_alpha_fct(x)
                / (2.0 * weight_lbf / wing_area_sft * self.lbf_to_N / self.ft_to_m**2)
        )
        load_factor_gust_n = lambda u_de_v, x: float(
                1 - K_g(mu_g(cl_alpha_fct(x))) * atm_0.density * u_de_v * self.ft_to_m * x * cl_alpha_fct(x)
                / (2.0 * weight_lbf / wing_area_sft * self.lbf_to_N / self.ft_to_m ** 2)
        )
        load_factor_stall_p = lambda x: (x / Vs_1g_ps) ** 2.0
        load_factor_stall_n = lambda x: -(x / Vs_1g_ng) ** 2.0

        # We can now go back to the computation of the maneuvering speeds, we will first compute it
        # "traditionally" and should we find out that the line limited by the Cl max is under the gust
        # line, we will adjust it (see Step 10. of section 16.4.1 of (1)). As for the traditional
        # computation they can be found in CS 23.335 (c)
        # https://www.easa.europa.eu/sites/default/files/dfu/CS-23%20Amendment%204.pdf
        # https://www.astm.org/Standards/F3116.htm

        Vma_ps = Vs_1g_ps * math.sqrt(n_lim_ps)  # [m/s]
        Vma_ng = Vs_1g_ng * math.sqrt(abs(n_lim_ng))  # [m/s]
        velocity_array.append(float(Vma_ps))
        velocity_array.append(float(Vma_ng))

        # We now need to check if we are in the aforementioned case (usually happens for low design wing
        # loading aircraft and/or mission wing loading)

        n_ma_ps = load_factor_gust_p(U_de_Vc, Vma_ps)

        if n_ma_ps > n_lim_ps:
            # In case the gust line load factor is above the maneuvering load factor, we need to solve the difference
            # between both curve to be 0.0 to find intersect
            delta = lambda x: load_factor_gust_p(U_de_Vc, x) - load_factor_stall_p(x)
            Vma_ps = max(optimize.fsolve(delta, np.array(1000.0)))
            n_ma_ps = load_factor_gust_p(U_de_Vc, Vma_ps)  # [-]
            velocity_array.append(float(Vma_ps))
            load_factor_array.append(float(n_ma_ps))
        else:
            velocity_array.append(0.0)
            load_factor_array.append(0.0)

        # We now need to do the same thing for the negative maneuvering speed

        n_ma_ng = load_factor_gust_n(U_de_Vc, Vma_ng)  # [-]

        if n_ma_ng < n_lim_ng:
            # In case the gust line load factor is above the maneuvering load factor, we need to solve the difference
            # between both curve to be 0.0 to find intersect
            delta = lambda x: load_factor_gust_n(U_de_Vc, x) - load_factor_stall_n(x)
            Vma_ng = max(optimize.fsolve(delta, np.array(1000.0))[0])
            n_ma_ng = load_factor_gust_n(U_de_Vc, Vma_ng)  # [-]
            velocity_array.append(float(Vma_ng))
            load_factor_array.append(float(n_ma_ng))
        else:
            velocity_array.append(0.0)
            load_factor_array.append(0.0)


        # For the cruise velocity, things will be different since it is an entry choice. As such we will
        # simply check that it complies with the values given in the certification papers and re-adjust
        # it if necessary. For airplane certified for aerobatics, the coefficient in front of the wing
        # loading in psf is slightly different than for normal aircraft but for either case it becomes
        # 28.6 at wing loading superior to 100 psf
        # Values and methodology used can be found in CS 23.335 (a)
        # https://www.easa.europa.eu/sites/default/files/dfu/CS-23%20Amendment%204.pdf
        # https://www.astm.org/Standards/F3116.htm

        if category == 1.0:
            if mtow_loading_psf < 20.0:
                k_c = 36.0
            elif mtow_loading_psf < 100.:
                # Linear variation from 33.0 to 28.6
                k_c = 36.0 + (mtow_loading_psf - 20.0) * (28.6 - 36.0) / (100.0 - 20.0)
            else:
                k_c = 28.6
        else:
            if mtow_loading_psf < 20.0:
                k_c = 33.0
            elif mtow_loading_psf < 100.:
                # Linear variation from 33.0 to 28.6
                k_c = 33.0 + (mtow_loading_psf - 20.0) * (28.6 - 33.0) / (100.0 - 20.0)
            else:
                k_c = 28.6

        Vc_min_1 = k_c * math.sqrt(weight_lbf / wing_area_sft) * self.kts_to_ms  # [m/s]

        # This second constraint rather refers to the paragraph on maneuvering speeds, which needs to be chosen
        # so that they are smaller than cruising speeds
        Vc_min_2 = Vma_ps  # [m/s]
        Vc_min = max(Vc_min_1, Vc_min_2)  # [m/s]

        # The certifications specifies that Vc need not be more than 0.9 Vh so we will simply take the
        # minimum value between the Vc_min and this value

        Vc_min_fin = min(Vc_min, 0.9 * Vh)  # [m/s]

        # The constraint regarding the maximum velocity for cruise does not appear in the certifications but
        # from a physics point of view we can easily infer that the cruise speed will never be greater than
        # the maximum level velocity at sea level hence

        Vc = max(min(design_vc, Vh), Vc_min_fin)  # [m/s]
        velocity_array.append(float(Vc))
        load_factor_array.append(float(n_lim_ng))

        # Lets now look at the load factors associated with the Vc, since it is here that the greatest
        # load factors can appear

        n_Vc_ps = max(load_factor_gust_p(U_de_Vc, Vc), n_lim_ps)  # [-]
        n_Vc_ng = min(load_factor_gust_n(U_de_Vc, Vc), n_lim_ng)  # [-]

        velocity_array.append(float(Vc))
        load_factor_array.append(float(n_Vc_ps))
        velocity_array.append(float(Vc))
        load_factor_array.append(float(n_Vc_ng))

        # We now compute the diving speed, methods are described in CS 23.335 (b). We will take the minimum
        # diving speed allowable as our design diving speed. We need to keep in mind that this speed could
        # be greater if the designer was willing to show that the structure holds for the wanted Vd. For
        # airplane that needs to be certified for aerobatics use, the factor between Vd_min and Vc_min is
        # slightly different, but they both become 1.35 for wing loading higher than 100 psf
        # https://www.easa.europa.eu/sites/default/files/dfu/CS-23%20Amendment%204.pdf
        # https://www.astm.org/Standards/F3116.htm

        Vd_min_1 = 1.25 * Vc  # [m/s]

        if category == 1.0:
            if mtow_loading_psf < 20.0:
                k_d = 1.55
            elif mtow_loading_psf < 100.:
                # Linear variation from 1.55 to 1.35
                k_d = 1.55 + (mtow_loading_psf - 20.0) * (1.35 - 1.55) / (100.0 - 20.0)
            else:
                k_d = 1.35
        elif category == 2.0:
            if mtow_loading_psf < 20.0:
                k_d = 1.50
            elif mtow_loading_psf < 100.:
                # Linear variation from 1.5 to 1.35
                k_d = 1.50 + (mtow_loading_psf - 20.0) * (1.35 - 1.50) / (100.0 - 20.0)
            else:
                k_d = 1.35
        else:
            if mtow_loading_psf < 20.0:
                k_d = 1.4
            elif mtow_loading_psf < 100.:
                # Linear variation from 1.4 to 1.35
                k_d = 1.4 + (mtow_loading_psf - 20.0) * (1.35 - 1.4) / (100.0 - 20.0)
            else:
                k_d = 1.35

        Vd_min_2 = k_d * Vc_min_fin  # [m/s]
        Vd = max(Vd_min_1, Vd_min_2)  # [m/s]

        velocity_array.append(float(Vd))
        load_factor_array.append(float(n_lim_ps))
        velocity_array.append(float(Vd))
        load_factor_array.append(0.0)

        # Similarly to what was done for the design cruising speed we will explore the load factors
        # associated with the diving speed since gusts are likely to broaden the flight domain around
        # these points

        n_Vd_ps = load_factor_gust_p(U_de_Vd, Vd)  # [-]

        # For the negative load factor at the diving speed, it seems that for non_aerobatic airplanes, it is
        # always sized according to the gust lines, regardless of the negative design load factor. For aerobatic
        # airplanes however, it seems as if it is sized for a greater value (more negative) but it does not look
        # to be equal to the negative diving factor as can be seen in figure 16-13 of (1). No information was
        # found for the location of this precises point, so the choice was made to take it as the negative
        # design load factor or the load factor given by the gust, whichever is the greatest (most negative).
        # This way, for non aerobatic airplane, we ensure to be conservative.


        n_Vd_ng = load_factor_gust_n(U_de_Vd, Vd)  # [-]

        velocity_array.append(float(Vd))
        load_factor_array.append(float(n_Vd_ps))
        velocity_array.append(float(Vd))
        load_factor_array.append(float(n_Vd_ng))

        # We have now calculated all the velocities need to plot the flight domain. For the sake of
        # thoroughness we will also compute the maximal structural cruising speed and cruise never-exceed
        # speed. The computation for these two can be found in CS 23.1505
        # https://www.easa.europa.eu/sites/default/files/dfu/CS-23%20Amendment%204.pdf
        # https://www.astm.org/Standards/F3116.htm

        # Let us start, as presented in the certifications papers with the never-exceed speed
        # Since we made the choice to take the Vd as the minimum value allowed by certifications, the V_ne
        # will have a fixed value and not a range as one would have expect. Indeed if Vd = Vd_min and since
        # V_ne has to be greater or equal to 0.9 x Vd_min and smaller or equal to 0.9 x Vd, V_ne will be equal
        # to 0.9 Vd. For future implementations, it should be noted that this section will need to be rewritten
        # should the Vd become a design parameter like what was made on Vc. Additionally the effect of
        # buffeting which serves as an additional upper limit is not included but should be taken into
        # account in detailed analysis phases

        V_ne = 0.9 * Vd  # [m/s]

        velocity_array.append(float(V_ne))
        load_factor_array.append(0.0)

        V_no_min = Vc_min  # [m/s]
        V_no_max = min(Vc, 0.89 * V_ne)  # [m/s]

        # Again we need to make a choice for this speed : what value would be retained. We will take the
        # highest speed acceptable for certification, i.e

        V_no = max(V_no_min, V_no_max)  # [m/s]

        velocity_array.append(float(V_no))
        load_factor_array.append(0.0)

        # One additional velocity needs to be computed if we are talking about commuter aircraft. It is
        # the maximum gust intensity velocity. Due to the way we are returning the values, even if we are not
        # investigating a commuter aircraft we need to return a value for Vmg so we will put it to 0.0. If we
        # are investigating a commuter aircraft, we will compute it according ot the guidelines from CS 23.335 (d)
        # https://www.easa.europa.eu/sites/default/files/dfu/CS-23%20Amendment%204.pdf
        # https://www.astm.org/Standards/F3116.htm
        # We decided to put this computation here as we may need the gust load factor in cruise conditions for
        # one of the possible candidates for the Vmg. While writing this program, the writer realized they were
        # no paragraph that impeach the Vc from being at a value such that one one of the conditions for the
        # minimum speed was above the Vc creating a problem with point (2). This case may however never appear
        # in practice as it would suppose that the Vc chosen is above the stall line which is more than certainly
        # avoided by the correlation between Vc_min and W/S in CS 23.335 (a)

        if (level == 4.0) or (category == 4.0):

            # We first need to compute the intersection of the stall line with the gust line given by the
            # gust of maximum intensity. Similar calculation were already done in case the maneuvering speed
            # is dictated by the Vc gust line so the computation will be very similar
            delta = lambda x: load_factor_gust_p(U_de_Vmg, x) - load_factor_stall_p(x)
            Vmg_min_1 = max(optimize.fsolve(delta, np.array(1000.0))[0])

            # The second candidate for the Vmg is given by the stall speed and the load factor at the cruise
            # speed
            Vmg_min_2 = Vs_1g_ps * math.sqrt(load_factor_gust_p(U_de_Vc, Vc))  # [m/s]
            Vmg = min(Vmg_min_1, Vmg_min_2)  # [m/s]

            # As for the computation of the associated load factor, no source were found for any formula or
            # hint as to its computation. It can however be guessed that depending on the minimum value found
            # above, it will either be on the stall line or at the maximum design load factor

            if Vmg == Vmg_min_1:  # On the gust line
                n_Vmg = load_factor_gust_n(U_de_Vmg, Vmg_min_1)  # [-]
            else:
                n_Vmg = n_Vc_ps  # [-]

        else:
            Vmg = 0.0  # [m/s]
            n_Vmg = 0.0

        velocity_array.append(float(Vmg))
        load_factor_array.append(float(n_Vmg))

        # Let us now look at the flight domain in the flap extended configuration. For the computation of these
        # speeds and load factors, we will use the formula provided in CS 23.1511
        # https://www.easa.europa.eu/sites/default/files/dfu/CS-23%20Amendment%204.pdf
        # https://www.astm.org/Standards/F3116.htm

        # For the computation of the Vfe CS 23.1511, refers to CS 23.345 but there only seems to be a
        # requirement for the lowest the Vfe can be, hence we will take this speed as the Vfe. As for the
        # load factors that are prescribed we will use the guidelines provided in CS 23.345 (b)

        # Let us start by computing the Vfe
        Vsfe_1g_ps = math.sqrt((2. * mass * g) / (atm_0.density * wing_area * cl_max_flaps))  # [m/s]
        Vfe_min_1 = 1.4 * Vs_1g_ps  # [m/s]
        Vfe_min_2 = 1.8 * Vsfe_1g_ps  # [m/s]
        Vfe_min = max(Vfe_min_1, Vfe_min_2)  # [m/s]
        Vfe = Vfe_min  # [m/s]

        velocity_array.append(float(Vsfe_1g_ps))
        load_factor_array.append(1.0)


        # We can then move on to the computation of the load limitation of the flapped flight domain, which
        # must be equal to either a constant load factor of 2 or a load factor dictated by a gust of 25 fps.
        # Also since the use of flaps is limited to take-off, approach and landing, we will use the SL density
        # and a constant gust velocity

        U_de_fe = 25.  # [ft/s]
        n_lim_ps_fe = 2.0
        n_Vfe = max(n_lim_ps_fe, load_factor_gust_n(U_de_fe, Vfe))

        velocity_array.append(float(Vsfe_1g_ps * math.sqrt(n_Vfe)))
        load_factor_array.append(float(n_Vfe))
        velocity_array.append(float(Vfe))
        load_factor_array.append(float(n_Vfe))

        # We also store the conditions in which the values were computed so that we can easily access
        # them when drawing the flight domains

        conditions = [mass, altitude]

        return velocity_array, load_factor_array, conditions
