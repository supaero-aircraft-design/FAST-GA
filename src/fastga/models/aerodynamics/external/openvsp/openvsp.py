"""Estimation of cl/cm/oswald aero coefficients using OPENVSP."""
#  This file is part of FAST-OAD_CS23 : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2026  ONERA & ISAE-SUPAERO
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
import os.path as pth
from pathlib import Path
import warnings
from importlib.resources import path
import json
import copy
import atexit
import logging
import numpy as np

from distutils.dir_util import copy_tree
from typing import Optional, Union

# noinspection PyProtectedMember
from fastoad._utils.resource_management.copy import copy_resource

from openmdao.components.external_code_comp import ExternalCodeComp
from openmdao.utils.file_wrap import InputFileGenerator
from stdatm import Atmosphere

# noinspection PyProtectedMember
from fastga.command.api import _create_tmp_directory
from fastga.utils.resource_management.copy import copy_resource_from_path
from . import openvsp3201
from . import resources as local_resources
from ... import airfoil_folder
from ...constants import SPAN_MESH_POINT, MACH_NB_PTS, GEOMETRY_SET_LABELS, RESULT_LABELS

DEFAULT_WING_AIRFOIL = "naca23012.af"
DEFAULT_HTP_AIRFOIL = "naca0012.af"
INPUT_WING_SCRIPT = "wing_openvsp.vspscript"
INPUT_WING_ROTOR_SCRIPT = "wing_rotor_openvsp.vspscript"
INPUT_HTP_SCRIPT = "ht_openvsp.vspscript"
INPUT_AIRCRAFT_SCRIPT = "wing_ht_openvsp.vspscript"
STDERR_FILE_NAME = "vspaero_calc.err"
VSPSCRIPT_EXE_NAME = "vspscript.exe"
VSPAERO_EXE_NAME = "vspaero.exe"

# Sub-dictionary names used inside a geometry cache entry, next to the mach-keyed
# post-processed results of compute_aero_coeff. They can never collide with a mach
# key since mach keys are string representations of floats.
RAW_AERO_CACHE_NAMESPACE = "raw_aero"
ROTOR_CACHE_NAMESPACE = "wing_rotor"
WING_ROTOR_CONDITION_LABELS = [
    "altitude",
    "mach",
    "AoA",
    "thrust_coefficient",
    "power_coefficient",
    "engine_count",
    "engine_config",
    "engine_rpm",
    "propeller_diameter",
    "nac_length",
    "y_ratio_key",
]
RAW_AERO_CONDITION_LABELS = ["comp_opt", "altitude", "mach", "AoA"]


_LOGGER = logging.getLogger(__name__)


class _NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that also knows how to serialize numpy arrays/scalars, which
    the OpenVSP result cache stores (e.g. y_vector_wing, cl_vector_wing)."""

    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()
        return super().default(o)


class OpenVSPSimpleGeometry(ExternalCodeComp):
    """Execution of OpenVSP for clean surfaces."""

    _cache: dict = {}

    # File the cache should be saved to (only set when a `result_file_name` is
    # configured).
    _cache_file: Optional[str] = None

    # Folder/file the cache was last loaded from, used only to avoid redundant
    # reloads on repeated component instantiation. Distinct from `_cache_file`:
    # this is set even when no `result_file_name` is configured (folder-only mode).
    _cache_loaded_from: Optional[str] = None

    # Guards against registering the atexit save more than once.
    _atexit_registered: bool = False

    @staticmethod
    def _resolve_cache_path(folder_path: Union[str, Path], file_name: str) -> Path:
        """
        Turns the `result_folder_path`/`result_file_name` options into the full cache
        file path.
        """
        return (Path(folder_path) / file_name).resolve()

    @classmethod
    def load_cache(cls, folder_path: Union[str, Path]) -> None:
        """
        Loads a previously saved OpenVSP result cache from disk and merges it into the in-memory cache.

        :param folder_path: the result folder (e.g. `result_folder_path`) inside
            which the cache file lives.
        """

        search_folder = Path(folder_path).resolve()
        cls._cache_loaded_from = str(search_folder)

        no_openvsp_cache = True

        for file in search_folder.glob("*.json"):
            try:
                with open(file, "r", encoding="utf-8") as cache_fp:
                    saved_cache = json.load(cache_fp)
            except (json.JSONDecodeError, OSError) as exc:
                # The result folder may contain JSON files unrelated to the OpenVSP cache.
                # Skip them rather than aborting the whole load.
                _LOGGER.warning("Skipping unreadable cache file %s: %s", file, exc)
                continue

            if isinstance(saved_cache, dict):
                for key, value in saved_cache.items():
                    # Register in cache only if the key contains all the geometry labels
                    if all(label in key for label in GEOMETRY_SET_LABELS) and isinstance(
                        value, dict
                    ):
                        cls._cache[key] = value
                        no_openvsp_cache = False

        if no_openvsp_cache:
            _LOGGER.info("No existing OpenVSP cache found in %s, starting empty.", search_folder)
        else:
            _LOGGER.info(
                "Loaded OpenVSP cache from %s (%d geometry entries).",
                search_folder,
                len(cls._cache),
            )

    @classmethod
    def save_cache(
        cls,
        folder_path: Optional[Union[str, Path]] = None,
        file_name: Optional[str] = None,
    ) -> None:
        """
        Persists the current in-memory OpenVSP result cache to disk so it can be reused
        by a later run via `load_cache`. Intended to be called once, at the end of
        a run.

        :param folder_path: raw result folder to save into, together with `file_name`.
             Ignored unless `file_name` is given.
        :param file_name: cache file name to use together with `folder_path`.
        """
        if folder_path is not None and file_name:
            path = cls._resolve_cache_path(folder_path, file_name)
        elif cls._cache_file is not None:
            path = Path(cls._cache_file)
        else:
            _LOGGER.warning("save_cache() called with no folder_path and none set; skipping.")
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf8") as cache_fp:
            json.dump(cls._cache, cache_fp, cls=_NumpyJSONEncoder, indent=2)

        cls._cache_file = str(path)
        _LOGGER.info("Saved OpenVSP cache to %s (%d geometry entries).", path, len(cls._cache))

    def _setup_cache_persistence(self) -> None:
        """
        Loads any existing cache from `result_folder_path`/`result_file_name` and
        registers an `atexit` callback process that writes the cache into the
        json file when the process ends.
        """
        folder_path = self.options["result_folder_path"]
        file_name = self.options["result_file_name"]
        if not folder_path and not file_name:
            return
        # The resolved folder path
        resolved = (
            OpenVSPSimpleGeometry._resolve_cache_path(folder_path, file_name).parent
            if file_name
            else Path(folder_path).resolve()
        )

        # Prevent reloading the cache if it's already been loaded from the same path.
        if OpenVSPSimpleGeometry._cache_loaded_from is None:
            OpenVSPSimpleGeometry.load_cache(folder_path)

        # Register the atexit callback process only once, since multiple instances of this
        # component may be created. Only register it when `result_file_name` is set.
        if not OpenVSPSimpleGeometry._atexit_registered:
            if resolved is not None and file_name:
                atexit.register(
                    OpenVSPSimpleGeometry.save_cache, folder_path=folder_path, file_name=file_name
                )
            OpenVSPSimpleGeometry._atexit_registered = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.stderr = None

    def initialize(self):
        self.options.declare("result_folder_path", default="", types=str)
        self.options.declare(
            "result_file_name",
            default="",
            types=str,
            desc="Name of the file to store the results cache as a JSON file. If not set, "
            "the cache will not be saved to disk.",
        )
        self.options.declare("openvsp_exe_path", default="", types=str, allow_none=True)
        self.options.declare("airfoil_folder_path", default=None, types=str, allow_none=True)
        self.options.declare(
            "wing_airfoil_file", default=DEFAULT_WING_AIRFOIL, types=str, allow_none=True
        )
        self.options.declare(
            "htp_airfoil_file", default=DEFAULT_HTP_AIRFOIL, types=str, allow_none=True
        )

    def setup(self):
        # Setup cache persistence, placed under setup() due to options usage.
        self._setup_cache_persistence()

        self.add_input("data:geometry:wing:sweep_25", val=np.nan, units="deg")
        self.add_input("data:geometry:wing:taper_ratio", val=np.nan)
        self.add_input("data:geometry:wing:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:wing:MAC:leading_edge:x:local", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:z", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:sweep_0", val=np.nan, units="deg")
        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("data:geometry:wing:dihedral", val=np.nan, units="rad")
        self.add_input(
            "data:geometry:wing:twist",
            val=np.nan,
            units="deg",
            desc="Negative twist means tip AOA is smaller than root",
        )
        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:sweep_25", val=np.nan, units="deg")
        self.add_input("data:geometry:horizontal_tail:taper_ratio", val=np.nan)
        self.add_input("data:geometry:horizontal_tail:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:horizontal_tail:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:horizontal_tail:span", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:tip:chord", val=np.nan, units="m")
        self.add_input(
            "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="m"
        )
        self.add_input("data:geometry:horizontal_tail:MAC:length", val=np.nan, units="m")
        self.add_input(
            "data:geometry:horizontal_tail:MAC:at25percent:x:local", val=np.nan, units="m"
        )
        self.add_input("data:geometry:horizontal_tail:z:from_wingMAC25", val=np.nan, units="m")

    def check_config(self, logger):
        # let void to avoid logger error on "The command cannot be empty"
        pass

    def compute_cl_alpha_aircraft(self, inputs, outputs, altitude, mach, aoa_angle):
        """
        Function that perform a complete calculation of aerodynamic parameters under OpenVSP and
        returns only the cl_alpha_aircraft parameter.
        """
        (
            _,
            _,
            cl_alpha_wing,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            cl_alpha_htp,
            _,
            _,
            _,
            _,
            _,
        ) = self.compute_aero_coeff(inputs, outputs, altitude, mach, aoa_angle)
        return float(cl_alpha_wing + cl_alpha_htp)

    def compute_cl_alpha_mach(self, inputs, outputs, aoa_angle, altitude, cruise_mach):
        """
        Function that performs multiple run of OpenVSP to get an interpolation of Cl_alpha as a
        function of Mach for later use in the computation of the V-n diagram.
        """
        mach_interp = np.log(np.linspace(np.exp(0.15), np.exp(1.55 * cruise_mach), MACH_NB_PTS))
        cl_alpha_interp = np.zeros_like(mach_interp)
        for idx, mach in enumerate(mach_interp):
            cl_alpha_interp[idx] = self.compute_cl_alpha_aircraft(
                inputs, outputs, altitude, mach, aoa_angle
            )

        # We add the case were M=0, for thoroughness and since we are in an incompressible flow,
        # the Cl_alpha is approximately the same as for the first Mach of the interpolation
        mach_interp = np.insert(mach_interp, 0, 0.0)
        cl_alpha_inc = cl_alpha_interp[0]
        cl_alpha_interp = np.insert(cl_alpha_interp, 0, cl_alpha_inc)

        return mach_interp, cl_alpha_interp

    def compute_aero_coeff(self, inputs, outputs, altitude, mach, aoa_angle):
        """
        Function that computes in OpenVSP environment all the aerodynamic parameters @0° and
        aoa_angle and calculate the associated derivatives.

        @param inputs: inputs parameters defined within FAST-OAD-GA
        @param outputs: outputs parameters defined within FAST-OAD-GA
        @param altitude: altitude for aerodynamic calculation in meters
        @param mach: air speed expressed in mach
        @param aoa_angle: air speed angle of attack with respect to aircraft
        @return: cl_0_wing, cl_alpha_wing, cm_0_wing, y_vector_wing, cl_vector_wing, coef_k_wing,
        cl_0_htp,  cl_aoa_htp, cl_alpha_htp, cl_alpha_htp_isolated, y_vector_htp, cl_vector_htp,
        coef_k_htp parameters.
        """
        # initialize
        results = [None] * 17
        # Fix mach number of digits to consider similar results
        mach = round(float(mach) * 1e3) / 1e3

        # Get inputs necessary to define global geometry
        s_ref_wing, area_ratio, geometry_set = self.define_geometry(inputs)

        # Create a key to store/retrieve results in cache
        key = str(dict(zip(GEOMETRY_SET_LABELS, geometry_set)))

        # If no result saved for that geometry under this mach condition, computation is done
        if self._cache.get(key) is None or self._cache[key].get(str(mach)) is None:
            self.register_geometry(key)

            # Compute wing alone @ 0°/X° angle of attack
            wing_0 = self.compute_aero(inputs, outputs, altitude, mach, 0.0, comp_opt="wing")
            wing_aoa = self.compute_aero(
                inputs, outputs, altitude, mach, aoa_angle, comp_opt="wing"
            )

            # Compute complete aircraft @ 0°/X° angle of attack
            _, htp_0, _ = self.compute_aero(inputs, outputs, altitude, mach, 0.0, comp_opt="ac")
            _, htp_aoa, _ = self.compute_aero(
                inputs, outputs, altitude, mach, aoa_angle, comp_opt="ac"
            )

            # Small remark: we could used the full aircraft simulation to get the wing aerodynamic
            # data because HTP should have a minor impact on the wing characteristics.
            # Unfortunately, results are a bit different in value and format

            # Compute isolated HTP @ 0°/X° angle of attack
            htp_0_isolated = self.compute_aero(inputs, outputs, altitude, mach, 0.0, comp_opt="htp")
            htp_aoa_isolated = self.compute_aero(
                inputs, outputs, altitude, mach, aoa_angle, comp_opt="htp"
            )

            # Post-process wing data ---------------------------------------------------------------
            (
                cl_0_wing,
                cl_x_wing,
                cl_alpha_wing,
                cm_0_wing,
                y_vector_wing,
                cl_vector_wing,
                chord_vector_wing,
                coef_k_wing,
            ) = self.post_process_wing(inputs, wing_0, wing_aoa, s_ref_wing, aoa_angle)

            # Post-process HTP data ----------------------------------------------------------------
            (
                cl_0_htp,
                cl_aoa_htp,
                cl_alpha_htp,
                cl_alpha_htp_isolated,
                y_vector_htp,
                cl_vector_htp,
                coef_k_htp,
            ) = self.post_process_htp(
                htp_0, htp_aoa, htp_0_isolated, htp_aoa_isolated, aoa_angle, area_ratio
            )

            # Resize vectors -----------------------------------------------------------------------
            vector_wing = [y_vector_wing, cl_vector_wing, chord_vector_wing]
            y_vector_wing, cl_vector_wing, chord_vector_wing = self.resize_vector(vector_wing)
            vector_htp = [y_vector_htp, cl_vector_htp]
            y_vector_htp, cl_vector_htp = self.resize_vector(vector_htp)

            # Save results to defined path ---------------------------------------------------------
            results = [
                float(cl_0_wing),
                float(cl_x_wing),
                float(cl_alpha_wing),
                float(cm_0_wing),
                np.array(y_vector_wing).tolist(),
                np.array(cl_vector_wing).tolist(),
                np.array(chord_vector_wing).tolist(),
                float(coef_k_wing),
                float(cl_0_htp),
                float(cl_aoa_htp),
                float(cl_alpha_htp),
                float(cl_alpha_htp_isolated),
                np.array(y_vector_htp).tolist(),
                np.array(cl_vector_htp).tolist(),
                float(coef_k_htp),
                float(s_ref_wing),
                float(area_ratio),
            ]

            self.save_results(key, results, mach)

        # Else retrieved results are used, eventually adapted with new area ratio
        else:
            # Read values from result file ---------------------------------------------------------
            results = self.assign_read_data(key, mach, s_ref_wing, area_ratio)

        # last two values are s_ref_wing and area_ratio, not needed in return

        return results[:-1]

    def compute_aero(
        self, inputs, outputs, altitude, mach, aoa_angle, comp_opt="wing", use_cache=False
    ):
        """
        Thin cache-checking wrapper around :meth:`_compute_aero_impl`, which does the
        actual OpenVSP run. Mirrors the compute_wing_rotor / _compute_wing_rotor split:
        this method itself never runs OpenVSP - it either calls the impl directly
        (use_cache=False) or looks the result up in (and stores it into) the in-memory
        cache before calling the impl on a miss (use_cache=True). Keeping the cache
        check and the actual computation in two separate methods means the cache-miss
        branch can call the impl unconditionally, with no risk of accidentally
        re-entering the cache-check branch and recursing forever.

        :param inputs: inputs parameters defined within FAST-OAD-GA
        :param outputs: outputs parameters defined within FAST-OAD-GA
        :param altitude: altitude for aerodynamic calculation in meters
        :param mach: air speed expressed in mach
        :param aoa_angle: air speed angle of attack with respect to aircraft
        :param comp_opt: component to evaluate can be "wing", "htp" or "ac"
        :param use_cache: if True, the raw OpenVSP result is looked up in (and stored
            into) the in-memory cache, keyed by geometry, comp_opt, altitude, mach and
            aoa_angle, so the OpenVSP run is skipped when an identical combination has
            already been computed. Defaults to False so the calls made by
            compute_aero_coeff (which caches its own post-processed results) are
            unaffected.

        :return: wing/htp and aircraft dictionaries including their respective aerodynamic
        coefficients
        """
        if not use_cache:
            return self._compute_aero(inputs, outputs, altitude, mach, aoa_angle, comp_opt)

        # Fix mach number of digits to consider similar results (same rounding as
        # compute_aero_coeff)
        mach = round(float(mach) * 1e3) / 1e3

        _, _, geometry_set = self.define_geometry(inputs)
        key = str(dict(zip(GEOMETRY_SET_LABELS, geometry_set)))
        condition_values = [
            comp_opt,
            round(float(altitude), 1),
            mach,
            round(float(aoa_angle), 2),
        ]

        condition_key = str(dict(zip(RAW_AERO_CONDITION_LABELS, condition_values)))

        result = self.get_or_compute_cached(
            key,
            RAW_AERO_CACHE_NAMESPACE,
            condition_key,
            lambda: self._compute_aero(inputs, outputs, altitude, mach, aoa_angle, comp_opt),
        )
        # The "ac" option returns a (wing, htp, aircraft) tuple, which a JSON
        # save/load round-trip of the cache turns into a list. Normalize back so
        # the return type does not depend on where the result came from.
        if comp_opt == "ac" and isinstance(result, list):
            result = tuple(result)
        return result

    def _compute_aero(self, inputs, outputs, altitude, mach, aoa_angle, comp_opt="wing"):
        """
        Function that computes in OpenVSP environment the wing, horizontal stablizer(htp),
        and complete aircraft (considering wing and horizontal tail plan) and returns the
        different aerodynamic parameters. The downwash is done by OpenVSP considering far field.

        This is the actual (expensive) OpenVSP run, with no `use_cache` parameter by
        design. Call `compute_aero` instead, which decides whether to consult the cache
        before reaching this method - that keeps this method itself free of any cache
        logic it could recurse into.

        :param inputs: inputs parameters defined within FAST-OAD-GA
        :param outputs: outputs parameters defined within FAST-OAD-GA
        :param altitude: altitude for aerodynamic calculation in meters
        :param mach: air speed expressed in mach
        :param aoa_angle: air speed angle of attack with respect to aircraft
        :param comp_opt: component to evaluate can be "wing", "htp" or "ac"

        :return: wing/htp and aircraft dictionaries including their respective aerodynamic
        coefficients
        """
        _LOGGER.info(
            "Entring OpenVSP aerodynamic evaluation for %s at altitude %.1f m, Mach %.3f, AoA %.2f°",
        )

        output_result = {}
        # STEP 1/XX - DEFINE OR CALCULATE INPUT DATA FOR AERODYNAMIC EVALUATION ####################
        ############################################################################################

        # WING
        # Get inputs (and calculate missing ones)
        s_ref_wing = float(inputs["data:geometry:wing:area"])
        x0_wing = inputs["data:geometry:wing:MAC:leading_edge:x:local"]
        l0_wing = inputs["data:geometry:wing:MAC:length"]
        width_max = inputs["data:geometry:fuselage:maximum_width"]
        y1_wing = width_max / 2.0
        y2_wing = inputs["data:geometry:wing:root:y"]
        l2_wing = inputs["data:geometry:wing:root:chord"]
        y4_wing = inputs["data:geometry:wing:tip:y"]
        l4_wing = inputs["data:geometry:wing:tip:chord"]
        sweep_0_wing = inputs["data:geometry:wing:sweep_0"]
        fa_length = inputs["data:geometry:wing:MAC:at25percent:x"]
        span_wing = inputs["data:geometry:wing:span"]
        dihedral_angle = inputs["data:geometry:wing:dihedral"]
        twist = inputs["data:geometry:wing:twist"]
        height_max = inputs["data:geometry:fuselage:maximum_height"]
        # Compute remaining inputs
        atm = Atmosphere(altitude, altitude_in_feet=False)
        x_wing = fa_length - x0_wing - 0.25 * l0_wing
        # In the rest of the code the convention for z_wing is positive when wing below the
        # fuselage centerline, for OpenVSP it seems to be the other way around, hence the - sign
        z_wing = -inputs["data:geometry:wing:root:z"]
        span2_wing = y4_wing - y2_wing
        rho = atm.density
        v_inf = max(atm.speed_of_sound * mach, 0.01)  # avoid V=0 m/s crashes
        reynolds_wing = v_inf * l0_wing / atm.kinematic_viscosity

        # HTP
        # Get inputs (and calculate missing ones)
        s_ref_htp = float(inputs["data:geometry:horizontal_tail:area"])
        sweep_25_htp = inputs["data:geometry:horizontal_tail:sweep_25"]
        semi_span_htp = inputs["data:geometry:horizontal_tail:span"] / 2.0
        span_htp = (
            inputs["data:geometry:horizontal_tail:span"] / 2.0
        )  # full span? half span for htp?
        root_chord_htp = inputs["data:geometry:horizontal_tail:root:chord"]
        tip_chord_htp = inputs["data:geometry:horizontal_tail:tip:chord"]
        lp_htp = inputs["data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"]
        l0_htp = inputs["data:geometry:horizontal_tail:MAC:length"]
        x0_htp = inputs["data:geometry:horizontal_tail:MAC:at25percent:x:local"]
        height_htp = inputs["data:geometry:horizontal_tail:z:from_wingMAC25"]
        # Compute remaining inputs
        x_htp = fa_length + lp_htp - x0_htp - 0.25 * l0_htp
        z_htp = -(height_max - 0.12 * l0_htp) * 0.5 - height_htp
        reynolds_htp = v_inf * l0_htp / atm.kinematic_viscosity

        # A/C
        distance_htp = fa_length + lp_htp - 0.25 * l0_htp - x0_htp

        # STEP 2/XX - DEFINE WORK DIRECTORY, COPY RESOURCES AND CREATE COMMAND BATCH ###############
        ############################################################################################

        # If a folder path is specified for openvsp .exe, it becomes working directory (target),
        # if not temporary folder is created
        if self.options["openvsp_exe_path"]:
            target_directory = pth.abspath(self.options["openvsp_exe_path"])
        else:
            tmp_directory = _create_tmp_directory()
            target_directory = tmp_directory.name

        # Define the list of necessary input files: geometry script and foil file for wing, HTP,
        # and aircraft
        if comp_opt == "wing":
            input_file_list = [
                pth.join(target_directory, INPUT_WING_SCRIPT),
                pth.join(target_directory, self.options["wing_airfoil_file"]),
            ]
        elif comp_opt == "htp":
            input_file_list = [
                pth.join(target_directory, INPUT_HTP_SCRIPT),
                pth.join(target_directory, self.options["htp_airfoil_file"]),
            ]
        else:
            # When in doubt we compute everything
            input_file_list = [
                pth.join(target_directory, INPUT_AIRCRAFT_SCRIPT),
                pth.join(target_directory, self.options["wing_airfoil_file"]),
                pth.join(target_directory, self.options["htp_airfoil_file"]),
            ]

        self.options["external_input_files"] = input_file_list
        # Define standard error file by default to avoid error code return
        self.stderr = pth.join(target_directory, STDERR_FILE_NAME)
        # Copy resource in working (target) directory
        # noinspection PyTypeChecker
        copy_tree(pth.dirname(openvsp3201.__file__), target_directory, verbose=0)
        if self.options["airfoil_folder_path"] is None:
            if comp_opt == "wing":
                copy_resource(airfoil_folder, self.options["wing_airfoil_file"], target_directory)
            elif comp_opt == "htp":
                copy_resource(airfoil_folder, self.options["htp_airfoil_file"], target_directory)
            elif comp_opt == "ac":
                copy_resource(airfoil_folder, self.options["wing_airfoil_file"], target_directory)
                copy_resource(airfoil_folder, self.options["htp_airfoil_file"], target_directory)
        else:
            if comp_opt == "wing":
                copy_resource_from_path(
                    self.options["airfoil_folder_path"],
                    self.options["wing_airfoil_file"],
                    target_directory,
                )
            elif comp_opt == "htp":
                copy_resource_from_path(
                    self.options["airfoil_folder_path"],
                    self.options["htp_airfoil_file"],
                    target_directory,
                )
            elif comp_opt == "ac":
                copy_resource_from_path(
                    self.options["airfoil_folder_path"],
                    self.options["wing_airfoil_file"],
                    target_directory,
                )
                copy_resource_from_path(
                    self.options["airfoil_folder_path"],
                    self.options["htp_airfoil_file"],
                    target_directory,
                )

        # Create corresponding .bat files (one for each geometry configuration)
        self.options["command"] = [pth.join(target_directory, "vspscript.bat")]
        batch_file = open(self.options["command"][0], "w+")
        batch_file.write("@echo off\n")

        if comp_opt == "wing":
            command = (
                pth.join(target_directory, VSPSCRIPT_EXE_NAME)
                + " -script "
                + pth.join(target_directory, INPUT_WING_SCRIPT)
                + " >nul 2>nul\n"
            )
        elif comp_opt == "htp":
            command = (
                pth.join(target_directory, VSPSCRIPT_EXE_NAME)
                + " -script "
                + pth.join(target_directory, INPUT_HTP_SCRIPT)
                + " >nul 2>nul\n"
            )
        else:
            # When in doubt we compute everything
            command = (
                pth.join(target_directory, VSPSCRIPT_EXE_NAME)
                + " -script "
                + pth.join(target_directory, INPUT_AIRCRAFT_SCRIPT)
                + " >nul 2>nul\n"
            )

        batch_file.write(command)
        batch_file.close()

        # STEP 3/XX - OPEN THE TEMPLATE SCRIPT FOR GEOMETRY GENERATION, MODIFY VALUES AND SAVE TO
        # WORKDIR #################################################################################
        if comp_opt == "wing":
            output_file_list = [
                pth.join(
                    target_directory,
                    INPUT_WING_SCRIPT.replace(".vspscript", "_DegenGeom.csv"),
                )
            ]
            parser = InputFileGenerator()
            with path(local_resources, INPUT_WING_SCRIPT) as input_template_path:
                parser.set_template_file(str(input_template_path))
                parser.set_generated_file(input_file_list[0])
                # Modify wing parameters
                parser.mark_anchor("x_wing")
                parser.transfer_var(float(x_wing), 0, 5)
                parser.mark_anchor("z_wing")
                parser.transfer_var(float(z_wing), 0, 5)
                parser.mark_anchor("y1_wing")
                parser.transfer_var(float(y1_wing), 0, 5)
                for i in range(3):
                    parser.mark_anchor("l2_wing")
                    parser.transfer_var(float(l2_wing), 0, 5)
                parser.reset_anchor()
                parser.mark_anchor("span2_wing")
                parser.transfer_var(float(span2_wing), 0, 5)
                parser.mark_anchor("l4_wing")
                parser.transfer_var(float(l4_wing), 0, 5)
                parser.mark_anchor("sweep_0_wing")
                parser.transfer_var(float(sweep_0_wing), 0, 5)
                parser.mark_anchor("twist")
                parser.transfer_var(float(twist), 0, 5)
                parser.mark_anchor("dihedral_angle")
                parser.transfer_var(float(dihedral_angle), 0, 5)
                parser.mark_anchor("airfoil_0_file")
                parser.transfer_var('"' + input_file_list[1].replace("\\", "/") + '"', 0, 3)
                parser.mark_anchor("airfoil_1_file")
                parser.transfer_var('"' + input_file_list[1].replace("\\", "/") + '"', 0, 3)
                parser.mark_anchor("airfoil_2_file")
                parser.transfer_var('"' + input_file_list[1].replace("\\", "/") + '"', 0, 3)
                parser.mark_anchor("csv_file")
                csv_name = output_file_list[0]
                parser.transfer_var('"' + csv_name.replace("\\", "/") + '"', 0, 3)
                parser.generate()
        elif comp_opt == "htp":
            output_file_list = [
                pth.join(
                    target_directory,
                    INPUT_HTP_SCRIPT.replace(".vspscript", "_DegenGeom.csv"),
                )
            ]
            parser = InputFileGenerator()
            with path(local_resources, INPUT_HTP_SCRIPT) as input_template_path:
                parser.set_template_file(str(input_template_path))
                parser.set_generated_file(input_file_list[0])
                # Modify htp parameters
                parser.mark_anchor("x_htp")
                parser.transfer_var(float(x_htp), 0, 5)
                parser.mark_anchor("z_htp")
                parser.transfer_var(float(z_htp), 0, 5)
                parser.mark_anchor("semi_span_htp")
                parser.transfer_var(float(semi_span_htp), 0, 5)
                parser.mark_anchor("root_chord_htp")
                parser.transfer_var(float(root_chord_htp), 0, 5)
                parser.mark_anchor("tip_chord_htp")
                parser.transfer_var(float(tip_chord_htp), 0, 5)
                parser.mark_anchor("sweep_25_htp")
                parser.transfer_var(float(sweep_25_htp), 0, 5)
                parser.mark_anchor("airfoil_0_file")
                parser.transfer_var('"' + input_file_list[1].replace("\\", "/") + '"', 0, 3)
                parser.mark_anchor("airfoil_1_file")
                parser.transfer_var('"' + input_file_list[1].replace("\\", "/") + '"', 0, 3)
                parser.mark_anchor("csv_file")
                csv_name = output_file_list[0]
                parser.transfer_var('"' + csv_name.replace("\\", "/") + '"', 0, 3)
                parser.generate()
        else:
            # When in doubt we compute everything
            output_file_list = [
                pth.join(
                    target_directory,
                    INPUT_AIRCRAFT_SCRIPT.replace(".vspscript", "_DegenGeom.csv"),
                )
            ]
            parser = InputFileGenerator()
            with path(local_resources, INPUT_AIRCRAFT_SCRIPT) as input_template_path:
                parser.set_template_file(str(input_template_path))
                parser.set_generated_file(input_file_list[0])
                # Modify wing parameters
                parser.mark_anchor("x_wing")
                parser.transfer_var(float(x_wing), 0, 5)
                parser.mark_anchor("z_wing")
                parser.transfer_var(float(z_wing), 0, 5)
                parser.mark_anchor("y1_wing")
                parser.transfer_var(float(y1_wing), 0, 5)
                for i in range(3):
                    parser.mark_anchor("l2_wing")
                    parser.transfer_var(float(l2_wing), 0, 5)
                parser.reset_anchor()
                parser.mark_anchor("span2_wing")
                parser.transfer_var(float(span2_wing), 0, 5)
                parser.mark_anchor("l4_wing")
                parser.transfer_var(float(l4_wing), 0, 5)
                parser.mark_anchor("sweep_0_wing")
                parser.transfer_var(float(sweep_0_wing), 0, 5)
                parser.mark_anchor("twist")
                parser.transfer_var(float(twist), 0, 5)
                parser.mark_anchor("dihedral_angle")
                parser.transfer_var(float(dihedral_angle), 0, 5)
                parser.mark_anchor("airfoil_0_file")
                parser.transfer_var('"' + input_file_list[-2].replace("\\", "/") + '"', 0, 3)
                parser.mark_anchor("airfoil_1_file")
                parser.transfer_var('"' + input_file_list[-2].replace("\\", "/") + '"', 0, 3)
                parser.mark_anchor("airfoil_2_file")
                parser.transfer_var('"' + input_file_list[-2].replace("\\", "/") + '"', 0, 3)
                # Modify HTP parameters
                parser.mark_anchor("distance_htp")
                parser.transfer_var(float(distance_htp), 0, 5)
                parser.mark_anchor("height_htp")
                parser.transfer_var(float(height_htp), 0, 5)
                parser.mark_anchor("span_htp")
                parser.transfer_var(float(span_htp), 0, 5)
                parser.mark_anchor("root_chord_htp")
                parser.transfer_var(float(root_chord_htp), 0, 5)
                parser.mark_anchor("tip_chord_htp")
                parser.transfer_var(float(tip_chord_htp), 0, 5)
                parser.mark_anchor("sweep_25_htp")
                parser.transfer_var(float(sweep_25_htp), 0, 5)
                parser.mark_anchor("airfoil_3_file")
                parser.transfer_var('"' + input_file_list[-1].replace("\\", "/") + '"', 0, 3)
                parser.mark_anchor("airfoil_4_file")
                parser.transfer_var('"' + input_file_list[-1].replace("\\", "/") + '"', 0, 3)
                parser.mark_anchor("csv_file")
                csv_name = output_file_list[0]
                parser.transfer_var('"' + csv_name.replace("\\", "/") + '"', 0, 3)
                parser.generate()

        # STEP 4/XX - RUN BATCH TO GENERATE GEOMETRY .CSV FILE #####################################
        ############################################################################################
        self.options["external_output_files"] = output_file_list
        super().compute(inputs, outputs)

        # STEP 5/XX - DEFINE NEW INPUT/OUTPUT FILES LIST AND CREATE BATCH FOR VLM COMPUTATION ######
        ############################################################################################
        input_file_list = output_file_list
        input_file_list.append(input_file_list[0].replace(".csv", ".vspaero"))
        output_file_list = [
            input_file_list[0].replace(".csv", ".lod"),
            input_file_list[0].replace(".csv", ".polar"),
        ]
        self.options["external_input_files"] = input_file_list
        self.options["external_output_files"] = output_file_list
        self.options["command"] = [pth.join(target_directory, "vspaero.bat")]
        batch_file = open(self.options["command"][0], "w+")
        batch_file.write("@echo off\n")
        command = (
            pth.join(target_directory, VSPAERO_EXE_NAME)
            + " "
            + input_file_list[1].replace(".vspaero", "")
            + " >nul 2>nul\n"
        )
        batch_file.write(command)
        batch_file.close()

        # STEP 6/XX - OPEN THE TEMPLATE VSPAERO FOR COMPUTATION, MODIFY VALUES AND SAVE TO WORKDIR #
        ############################################################################################
        parser = InputFileGenerator()
        template_file = pth.split(input_file_list[1])[1]
        with path(local_resources, template_file) as input_template_path:
            parser.set_template_file(str(input_template_path))
            parser.set_generated_file(input_file_list[1])
            parser.reset_anchor()

            if comp_opt == "htp":
                parser.mark_anchor("Sref")
                parser.transfer_var(float(s_ref_htp), 0, 3)
                parser.mark_anchor("Cref")
                parser.transfer_var(float(l0_htp), 0, 3)
                parser.mark_anchor("Bref")
                parser.transfer_var(float(2.0 * semi_span_htp), 0, 3)
                parser.mark_anchor("X_cg")
                parser.transfer_var(float(fa_length + lp_htp), 0, 3)
                reynolds = reynolds_htp
            else:
                # For wing and AC evaluation, same reference length and area
                parser.mark_anchor("Sref")
                parser.transfer_var(float(s_ref_wing), 0, 3)
                parser.mark_anchor("Cref")
                parser.transfer_var(float(l0_wing), 0, 3)
                parser.mark_anchor("Bref")
                parser.transfer_var(float(span_wing), 0, 3)
                parser.mark_anchor("X_cg")
                parser.transfer_var(float(fa_length), 0, 3)
                reynolds = reynolds_wing

            parser.mark_anchor("Mach")
            parser.transfer_var(float(mach), 0, 3)
            parser.mark_anchor("AOA")
            parser.transfer_var(float(aoa_angle), 0, 3)
            parser.mark_anchor("Vinf")
            parser.transfer_var(float(v_inf), 0, 3)
            parser.mark_anchor("Rho")
            parser.transfer_var(float(rho), 0, 3)
            parser.mark_anchor("ReCref")
            parser.transfer_var(float(reynolds), 0, 3)
            parser.generate()

        # STEP 7/XX - RUN BATCH TO GENERATE AERO OUTPUT FILES (.lod, .polar...) ####################
        ############################################################################################
        super().compute(inputs, outputs)

        # STEP 8/XX - READ FILES, RETURN RESULTS (AND CLEAR TEMPORARY WORKDIR) #####################
        ############################################################################################
        if comp_opt == "wing":
            # Open .lod file and extract data
            wing_y_vect = []
            wing_chord_vect = []
            wing_cl_vect = []
            wing_cd_vect = []
            wing_cm_vect = []
            with open(output_file_list[0], "r") as file_stream:
                data = file_stream.readlines()
                for i, _ in enumerate(data):
                    line = data[i].split()
                    line.append("**")
                    if line[0] == "1":
                        wing_y_vect.append(float(line[2]))
                        wing_chord_vect.append(float(line[3]))
                        wing_cl_vect.append(float(line[5]))
                        wing_cd_vect.append(float(line[6]))
                        wing_cm_vect.append(float(line[12]))
                    if line[0] == "Comp":
                        cl_wing = float(data[i + 1].split()[5]) + float(
                            data[i + 2].split()[5]
                        )  # sum CL left/right
                        cdi_wing = float(data[i + 1].split()[6]) + float(
                            data[i + 2].split()[6]
                        )  # sum CDi left/right
                        cm_wing = float(data[i + 1].split()[12]) + float(
                            data[i + 2].split()[12]
                        )  # sum CM left/right
                        break
            # Open .polar file and extract data
            with open(output_file_list[1], "r") as file_stream:
                data = file_stream.readlines()
                wing_e = float(data[1].split()[10])
            # Delete temporary directory
            if not self.options["openvsp_exe_path"]:
                # noinspection PyUnboundLocalVariable
                tmp_directory.cleanup()
            # Return values
            wing = {
                "y_vector": wing_y_vect,
                "cl_vector": wing_cl_vect,
                "chord_vector": wing_chord_vect,
                "cd_vector": wing_cd_vect,
                "cm_vector": wing_cm_vect,
                "cl": cl_wing,
                "cdi": cdi_wing,
                "cm": cm_wing,
                "coef_e": wing_e,
            }
            output_result = wing
        elif comp_opt == "htp":
            # Open .lod file and extract data
            htp_y_vect = []
            htp_cl_vect = []
            htp_cd_vect = []
            htp_cm_vect = []
            with open(output_file_list[0], "r") as lf:
                data = lf.readlines()
                for i, _ in enumerate(data):
                    line = data[i].split()
                    line.append("**")
                    if line[0] == "1":
                        htp_y_vect.append(float(line[2]))
                        htp_cl_vect.append(float(line[5]))
                        htp_cd_vect.append(float(line[6]))
                        htp_cm_vect.append(float(line[12]))
                    if line[0] == "Comp":
                        cl_htp = float(data[i + 1].split()[5]) + float(
                            data[i + 2].split()[5]
                        )  # sum CL left/right
                        cdi_htp = float(data[i + 1].split()[6]) + float(
                            data[i + 2].split()[6]
                        )  # sum CDi left/right
                        cm_htp = float(data[i + 1].split()[12]) + float(
                            data[i + 2].split()[12]
                        )  # sum CM left/right
                        break
            # Open .polar file and extract data
            with open(output_file_list[1], "r") as lf:
                data = lf.readlines()
                htp_e = float(data[1].split()[10])
            # Delete temporary directory
            if not self.options["openvsp_exe_path"]:
                # noinspection PyUnboundLocalVariable
                tmp_directory.cleanup()
            # Return values
            htp = {
                "y_vector": htp_y_vect,
                "cl_vector": htp_cl_vect,
                "cd_vector": htp_cd_vect,
                "cm_vector": htp_cm_vect,
                "cl": cl_htp,
                "cdi": cdi_htp,
                "cm": cm_htp,
                "coef_e": htp_e,
            }
            output_result = htp
        elif comp_opt == "ac":
            # Open .lod file and extract data
            wing_y_vect = []
            wing_cl_vect = []
            wing_cd_vect = []
            wing_cm_vect = []
            htp_y_vect = []
            htp_cl_vect = []
            htp_cd_vect = []
            htp_cm_vect = []
            with open(output_file_list[0], "r") as lf:
                data = lf.readlines()
                for i in range(len(data)):
                    line = data[i].split()
                    line.append("**")
                    if line[0] == "1":
                        wing_y_vect.append(float(line[2]))
                        wing_cl_vect.append(float(line[5]))
                        wing_cd_vect.append(float(line[6]))
                        wing_cm_vect.append(float(line[12]))
                    elif line[0] == "3":
                        htp_y_vect.append(float(line[2]))
                        htp_cl_vect.append(float(line[5]))
                        htp_cd_vect.append(float(line[6]))
                        htp_cm_vect.append(float(line[12]))
                    if line[0] == "Comp":
                        cl_wing = float(data[i + 1].split()[5]) + float(
                            data[i + 2].split()[5]
                        )  # sum CL left/right
                        cdi_wing = float(data[i + 1].split()[6]) + float(
                            data[i + 2].split()[6]
                        )  # sum CDi left/right
                        cm_wing = float(data[i + 1].split()[12]) + float(
                            data[i + 2].split()[12]
                        )  # sum CM left/right
                        cl_htp = float(data[i + 3].split()[5]) + float(
                            data[i + 4].split()[5]
                        )  # sum CL left/right
                        cdi_htp = float(data[i + 3].split()[6]) + float(
                            data[i + 4].split()[6]
                        )  # sum CDi left/right
                        cm_htp = float(data[i + 3].split()[12]) + float(
                            data[i + 4].split()[12]
                        )  # sum CM left/right
                        break
            # Open .polar file and extract data
            with open(output_file_list[1], "r") as lf:
                data = lf.readlines()
                aircraft_cl = float(data[1].split()[4])
                aircraft_cd0 = float(data[1].split()[5])
                aircraft_cdi = float(data[1].split()[6])
                aircraft_e = float(data[1].split()[10])
            # Delete temporary directory
            if not (self.options["openvsp_exe_path"]):
                # noinspection PyUnboundLocalVariable
                tmp_directory.cleanup()
            # Return values
            wing = {
                "y_vector": wing_y_vect,
                "cl_vector": wing_cl_vect,
                "cd_vector": wing_cd_vect,
                "cm_vector": wing_cm_vect,
                "cl": cl_wing,
                "cdi": cdi_wing,
                "cm": cm_wing,
            }
            htp = {
                "y_vector": htp_y_vect,
                "cl_vector": htp_cl_vect,
                "cd_vector": htp_cd_vect,
                "cm_vector": htp_cm_vect,
                "cl": cl_htp,
                "cdi": cdi_htp,
                "cm": cm_htp,
            }
            aircraft = {
                "cl": aircraft_cl,
                "cd0": aircraft_cd0,
                "cdi": aircraft_cdi,
                "coef_e": aircraft_e,
            }
            output_result = wing, htp, aircraft

        return output_result

    @staticmethod
    def define_geometry(inputs):
        """
        Extract geometrical parameters

        :param inputs: inputs in the OpenMDAO format

        :return s_ref_wing: wing reference surface area
        :return area_ratio: rea ratio between wing and horizontal stabilizer
        :return s_ref_wing: geometry dataset for openvsp calculation
        """
        s_ref_wing = float(inputs["data:geometry:wing:area"])
        s_ref_htp = float(inputs["data:geometry:horizontal_tail:area"])
        area_ratio = s_ref_htp / s_ref_wing
        sweep25_wing = float(inputs["data:geometry:wing:sweep_25"])
        taper_ratio_wing = float(inputs["data:geometry:wing:taper_ratio"])
        aspect_ratio_wing = float(inputs["data:geometry:wing:aspect_ratio"])
        sweep25_htp = float(inputs["data:geometry:horizontal_tail:sweep_25"])
        aspect_ratio_htp = float(inputs["data:geometry:horizontal_tail:aspect_ratio"])
        taper_ratio_htp = float(inputs["data:geometry:horizontal_tail:taper_ratio"])
        dihedral_angle = float(inputs["data:geometry:wing:dihedral"])
        twist_angle = float(inputs["data:geometry:wing:twist"])
        geometry_set = np.around(
            np.array(
                [
                    sweep25_wing,
                    taper_ratio_wing,
                    aspect_ratio_wing,
                    dihedral_angle,
                    twist_angle,
                    sweep25_htp,
                    taper_ratio_htp,
                    aspect_ratio_htp,
                ]
            ),
            decimals=6,
        )

        return s_ref_wing, area_ratio, geometry_set

    def register_geometry(self, key):
        """Register the geometry set as a key in the in-memory cache."""
        self._cache.setdefault(key, {})

    def save_results(self, key, results, mach):
        """Store OpenVSP results in the in-memory cache."""
        self._cache[key][str(mach)] = dict(zip(RESULT_LABELS, results))

    def get_or_compute_cached(self, key, namespace, condition_key, compute_fn):
        """
        Generic check-cache/compute-on-miss/store helper for raw OpenVSP results.

        Results are stored under ``self._cache[key][namespace][condition_key]`` so
        that they live next to the mach-keyed post-processed results used by
        :meth:`compute_aero_coeff` and are persisted/reloaded by the same
        ``save_cache``/``load_cache`` mechanism without any change.

        A deep copy of the cached value is returned so that callers mutating the
        result in place (e.g. padding vectors with ``list.extend``) cannot corrupt
        the cache for subsequent retrievals.

        :param key: geometry cache key, as built from GEOMETRY_SET_LABELS
        :param namespace: sub-dictionary name (e.g. RAW_AERO_CACHE_NAMESPACE)
        :param condition_key: string key identifying the run conditions
        :param compute_fn: zero-argument callable performing the actual (costly)
            computation, only invoked on a cache miss
        :return: the (copied) cached result
        """
        self.register_geometry(key)
        namespace_cache = self._cache[key].setdefault(namespace, {})

        if condition_key not in namespace_cache:
            namespace_cache[condition_key] = compute_fn()
        return copy.deepcopy(namespace_cache[condition_key])

    @staticmethod
    def post_process_wing(inputs, wing_0, wing_aoa, s_ref_wing, aoa_angle):
        """
        Computes wing data not produced by OpenVSP based on available ones

        :param inputs: inputs in the OpenMDAO format
        :param wing_0: wing aerodynamic coefficient with 0 angle of attack
        :param wing_aoa: wing aerodynamic coefficient with input angle of attack
        :param s_ref_wing: wing reference surface area
        :param aoa_angle: input angle of attack

        :return: lift coefficient at 0 aoa, lift coefficient at input aoa, lift slope coefficient,
        pitching moment coefficient, span vector, lift coefficient vector, chord vector,
        lift induced drag coefficient
        """
        width_max = inputs["data:geometry:fuselage:maximum_width"]
        span_wing = inputs["data:geometry:wing:span"]
        k_fus = 1 + 0.025 * width_max / span_wing - 0.025 * (width_max / span_wing) ** 2
        cl_0_wing = float(wing_0["cl"] * k_fus)
        cl_x_wing = float(wing_aoa["cl"] * k_fus)
        cm_0_wing = float(wing_0["cm"] * k_fus)
        cl_alpha_wing = (cl_x_wing - cl_0_wing) / (aoa_angle * np.pi / 180)
        y_vector_wing = wing_aoa["y_vector"]
        cl_vector_wing = (np.array(wing_aoa["cl_vector"]) * k_fus).tolist()
        chord_vector_wing = wing_aoa["chord_vector"]
        k_fus = 1 - 2 * (width_max / span_wing) ** 2  # Fuselage correction
        coef_e = float(wing_aoa["coef_e"] * k_fus)
        coef_k_wing = float(1.0 / (np.pi * span_wing**2 / s_ref_wing * coef_e))

        return (
            cl_0_wing,
            cl_x_wing,
            cl_alpha_wing,
            cm_0_wing,
            y_vector_wing,
            cl_vector_wing,
            chord_vector_wing,
            coef_k_wing,
        )

    @staticmethod
    def post_process_htp(htp_0, htp_aoa, htp_0_isolated, htp_aoa_isolated, aoa_angle, area_ratio):
        """
        Computes htp data not produced by OpenVSP based on available ones

        :param htp_0: htp aerodynamic coefficient with 0 angle of attack
        :param htp_aoa: htp aerodynamic coefficient with input angle of attack
        :param htp_0_isolated: isolated htp aerodynamic coefficient with 0 angle of attack
        :param htp_aoa_isolated: isolated htp aerodynamic coefficient with input angle of attack
        :param aoa_angle: input angle of attack
        :param area_ratio: ratio between htp reference area and wing reference area

        :return: htp lift coefficient at 0 aoa, htp lift coefficient at input aoa, htp lift slope
        coefficient, isolated htp lift slope coefficient, span vector, lift coefficient vector,
        lift induced drag coefficient
        """

        # Post-process HTP-aircraft data -------------------------------------------------------
        cl_0_htp = float(htp_0["cl"])
        cl_aoa_htp = float(htp_aoa["cl"])
        cl_alpha_htp = float((cl_aoa_htp - cl_0_htp) / (aoa_angle * np.pi / 180))
        coef_k_htp = float(htp_aoa["cdi"]) / cl_aoa_htp**2  # area ratio missing ?
        y_vector_htp = htp_aoa["y_vector"]
        cl_vector_htp = (np.array(htp_aoa["cl_vector"]) * area_ratio).tolist()

        # Post-process HTP-isolated data -------------------------------------------------------
        cl_alpha_htp_isolated = (
            float(htp_aoa_isolated["cl"] - htp_0_isolated["cl"])
            * area_ratio
            / (aoa_angle * np.pi / 180)
        )

        return (
            cl_0_htp,
            cl_aoa_htp,
            cl_alpha_htp,
            cl_alpha_htp_isolated,
            y_vector_htp,
            cl_vector_htp,
            coef_k_htp,
        )

    @staticmethod
    def resize_vector(vectors):
        """
        Format the size of results that need to be passed as array to the size declared to
        OpenMDAO if the original array is bigger we resize it, otherwise we complete with zeros.
        First vector always has to be the y vector, any other vector can be a quantity expressed
        as that vector.

        :param vectors: a tuple with wing aerodynamic results

        :return: length-modified aerodynamic results
        """
        y_vector = vectors[0]

        resized_vectors = []

        # shorter
        if SPAN_MESH_POINT < len(y_vector):
            y_interp = np.linspace(y_vector[0], y_vector[-1], SPAN_MESH_POINT)
            warnings.warn("Defined maximum span mesh in fast aerodynamics\\constants.py exceeded!")

            resized_vectors.append(y_interp)

            for vector in vectors[1:]:
                resized_vector = np.interp(y_interp, y_vector, vector)
                resized_vectors.append(resized_vector)

        # longer
        else:
            additional_zeros = list(np.zeros(SPAN_MESH_POINT - len(y_vector)))
            y_vector = y_vector + additional_zeros

            resized_vectors.append(y_vector)

            for vector in vectors[1:]:
                resized_vector = vector + additional_zeros
                resized_vectors.append(resized_vector)

        return resized_vectors

    def assign_read_data(self, key, mach, s_ref_wing, area_ratio):
        """
        When results already exists, read and assign existing data and apply the new area ratio

        :param data: existing results under the form of a dataframe
        :param area_ratio: area ratio between the wing and the horizontal stabilizer
        :param saved_area_ratio: area ratio between the wing and the horizontal stabilizer in
        existing results
        :param s_ref_wing: wing reference surface area

        :return: aerodynamic characteristic parameters of wing and horizontal stabilizer
        """
        data = self._cache[key][str(mach)]
        saved_area_wing = data.get("saved_ref_area")
        saved_area_ratio = data.get("area_ratio")
        cl_0_wing = data.get("cl_0_wing")
        cl_x_wing = data.get("cl_X_wing")
        cl_alpha_wing = data.get("cl_alpha_wing")
        cm_0_wing = data.get("cm_0_wing")
        y_vector_wing = np.array(data.get("y_vector_wing")) * np.sqrt(s_ref_wing / saved_area_wing)
        cl_vector_wing = np.array(data.get("cl_vector_wing"))
        chord_vector_wing = np.array(data.get("chord_vector_wing")) * np.sqrt(
            s_ref_wing / saved_area_wing
        )
        coef_k_wing = data.get("coef_k_wing")
        cl_0_htp = data.get("cl_0_htp") * (area_ratio / saved_area_ratio)
        cl_aoa_htp = data.get("cl_X_htp") * (area_ratio / saved_area_ratio)
        cl_alpha_htp = data.get("cl_alpha_htp") * (area_ratio / saved_area_ratio)
        cl_alpha_htp_isolated = data.get("cl_alpha_htp_isolated") * (area_ratio / saved_area_ratio)
        y_vector_htp = np.array(data.get("y_vector_htp"))
        cl_vector_htp = np.array(data.get("cl_vector_htp")) * (area_ratio / saved_area_ratio)
        coef_k_htp = data.get("coef_k_htp") * (area_ratio / saved_area_ratio)

        return (
            cl_0_wing,
            cl_x_wing,
            cl_alpha_wing,
            cm_0_wing,
            y_vector_wing,
            cl_vector_wing,
            chord_vector_wing,
            coef_k_wing,
            cl_0_htp,
            cl_aoa_htp,
            cl_alpha_htp,
            cl_alpha_htp_isolated,
            y_vector_htp,
            cl_vector_htp,
            coef_k_htp,
            s_ref_wing,
            area_ratio,
        )


class OpenVSPSimpleGeometryDP(OpenVSPSimpleGeometry):
    """Execution of OpenVSP for surfaces with slipstream effects."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._engine_wrapper = None

    def initialize(self):
        super().initialize()
        self.options.declare("propulsion_id", default=None, types=str, allow_none=True)

    def setup(self):
        super().setup()
        self.add_input("data:geometry:propulsion:engine:layout", val=np.nan)
        self.add_input("data:geometry:propulsion:engine:count", val=np.nan)
        self.add_input("data:geometry:wing:tip:leading_edge:x:local", val=np.nan, units="m")
        self.add_input("data:propulsion:max_rpm", val=np.nan, units="1/min")
        self.add_input("data:geometry:propeller:diameter", val=np.nan, units="m")
        self.add_input("data:geometry:propulsion:nacelle:length", val=np.nan, units="m")
        self.add_input(
            "data:geometry:propulsion:engine:y_ratio",
            shape_by_conn=True,
        )

    def compute_wing_rotor(self, inputs, outputs, altitude, mach, aoa_angle, thrust, power):
        """
        Cached wrapper around :meth:`_compute_wing_rotor`.

        Skips the OpenVSP run when an identical combination of geometry, flight
        condition and propulsion state has already been computed (in this process
        or in a reloaded cache file). Thrust and power are not cached on their raw
        values but through the derived, rounded thrust/power coefficients, so that
        negligible variations between iterations (e.g. finite-difference steps)
        still produce cache hits, mirroring the mach rounding of
        :meth:`compute_aero_coeff`.

        Parameters and return value are identical to :meth:`_compute_wing_rotor`.
        """
        # Fix mach number of digits to consider similar results (same rounding as
        # compute_aero_coeff)
        mach = round(float(mach) * 1e3) / 1e3

        _, _, geometry_set = self.define_geometry(inputs)
        key = str(dict(zip(GEOMETRY_SET_LABELS, geometry_set)))

        # Propulsion state, part of the cache key since the base geometry_set only
        # describes the wing and HTP planform
        engine_count = int(float(inputs["data:geometry:propulsion:engine:count"]))
        engine_config = float(inputs["data:geometry:propulsion:engine:layout"])
        engine_rpm = float(inputs["data:propulsion:max_rpm"])
        propeller_diameter = float(inputs["data:geometry:propeller:diameter"])
        nac_length = float(inputs["data:geometry:propulsion:nacelle:length"])
        if engine_config != 1.0:
            y_ratio_key = (0.0,)
        else:
            y_ratio_key = tuple(
                np.around(
                    np.atleast_1d(inputs["data:geometry:propulsion:engine:y_ratio"]).astype(float),
                    6,
                ).tolist()
            )

        # Same derivation and rounding as in _compute_wing_rotor
        atm = Atmosphere(altitude, altitude_in_feet=False)
        engine_rps = engine_rpm / 60.0
        thrust_coefficient = round(
            float(
                thrust / engine_count / (atm.density * engine_rps**2.0 * propeller_diameter**4.0)
            ),
            5,
        )
        power_coefficient = round(
            float(power / engine_count / (atm.density * engine_rps**3.0 * propeller_diameter**5.0)),
            5,
        )

        condition_values = [
            round(float(altitude), 1),
            mach,
            round(float(aoa_angle), 2),
            thrust_coefficient,
            power_coefficient,
            engine_count,
            engine_config,
            round(engine_rpm, 1),
            round(propeller_diameter, 3),
            round(nac_length, 3),
            y_ratio_key,
        ]

        condition_key = str(dict(zip(WING_ROTOR_CONDITION_LABELS, condition_values)))

        return self.get_or_compute_cached(
            key,
            ROTOR_CACHE_NAMESPACE,
            condition_key,
            lambda: self._compute_wing_rotor(
                inputs, outputs, altitude, mach, aoa_angle, thrust, power
            ),
        )

    def _compute_wing_rotor(self, inputs, outputs, altitude, mach, aoa_angle, thrust, power):
        """
        Function that computes in OpenVSP environment the wing with a rotor and returns the
        different aerodynamic parameters.

        @param inputs: inputs parameters defined within FAST-OAD-GA
        @param outputs: outputs parameters defined within FAST-OAD-GA
        @param altitude: altitude for aerodynamic calculation in meters
        @param mach: air speed expressed in mach
        @param aoa_angle: air speed angle of attack with respect to wing (degree)
        @param thrust: total aircraft thrust for computation of thrust coefficient (will be divided
        by engine count)
        @param power: total aircraft power for computation of power coefficient (will be divided
        by engine count)
        @return: wing dictionary including aero parameters as keys: y_vector, cl_vector, cd_vector,
        cm_vector, cl, cdi, cm, coef_e
        """

        # TODO : Check for rules that would allow the scaling of these results i.e, same D/span
        #  gives same results...
        # STEP 1/XX - DEFINE OR CALCULATE INPUT DATA FOR AERODYNAMIC EVALUATION ####################
        ############################################################################################

        _LOGGER.info(
            "Computing OpenVSP wing+rotor for altitude=%s, mach=%s, aoa=%s, thrust=%s, power=%s",
            altitude,
            mach,
            aoa_angle,
            thrust,
            power,
        )

        # Get inputs (and calculate missing ones)
        s_ref_wing = float(inputs["data:geometry:wing:area"])
        x0_wing = inputs["data:geometry:wing:MAC:leading_edge:x:local"]
        l0_wing = inputs["data:geometry:wing:MAC:length"]
        width_max = inputs["data:geometry:fuselage:maximum_width"]
        y1_wing = width_max / 2.0
        y2_wing = inputs["data:geometry:wing:root:y"]
        l2_wing = inputs["data:geometry:wing:root:chord"]
        y4_wing = inputs["data:geometry:wing:tip:y"]
        l4_wing = inputs["data:geometry:wing:tip:chord"]
        x4_wing = inputs["data:geometry:wing:tip:leading_edge:x:local"]
        sweep_0_wing = inputs["data:geometry:wing:sweep_0"]
        dihedral_angle = inputs["data:geometry:wing:dihedral"]
        twist = inputs["data:geometry:wing:twist"]
        fa_length = inputs["data:geometry:wing:MAC:at25percent:x"]
        span_wing = inputs["data:geometry:wing:span"]
        engine_rpm = inputs["data:propulsion:max_rpm"]
        propeller_diameter = float(inputs["data:geometry:propeller:diameter"])
        nac_length = inputs["data:geometry:propulsion:nacelle:length"]
        engine_config = inputs["data:geometry:propulsion:engine:layout"]
        engine_count = int(float(inputs["data:geometry:propulsion:engine:count"]))
        semi_span = span_wing / 2.0

        if engine_config != 1.0:
            y_ratio_array = 0.0
        else:
            y_ratio_array = np.array(inputs["data:geometry:propulsion:engine:y_ratio"])

        # Compute remaining inputs
        atm = Atmosphere(altitude, altitude_in_feet=False)
        x_wing = fa_length - x0_wing - 0.25 * l0_wing
        # In the rest of the code the convention for z_wing is positive when wing below the
        # fuselage centerline, for OpenVSP it seems to be the other way around, hence the - sign
        z_wing = -inputs["data:geometry:wing:root:z"]
        span2_wing = y4_wing - y2_wing
        rho = atm.density
        v_inf = max(atm.speed_of_sound * mach, 0.01)  # avoid V=0 m/s crashes
        atm.true_airspeed = v_inf
        reynolds = v_inf * l0_wing / atm.kinematic_viscosity

        # STEP 1.5/XX - COMPUTE THE PARAMETERS RELATED TO THE COMPUTATION OF THE SLIPSTREAM ########
        # EFFECTS ON THE WING ######################################################################

        thrust_one_prop = thrust / engine_count
        shaft_power_one_prop = power / engine_count
        engine_rps = engine_rpm / 60.0
        # For now thrust is distributed equally on each engine
        thrust_coefficient = round(
            float(thrust_one_prop / (rho * engine_rps**2.0 * propeller_diameter**4.0)), 5
        )
        power_coefficient = round(
            float(shaft_power_one_prop / (rho * engine_rps**3.0 * propeller_diameter**5.0)), 5
        )

        prop_radius = round(propeller_diameter / 2.0, 3)
        prop_hub_radius = round(0.2 * prop_radius, 3)

        # Writing propeller properties
        motor_pos_x = np.zeros(engine_count)
        motor_pos_y = np.zeros(engine_count)
        motor_pos_z = np.zeros(engine_count)
        motor_rpm_signed = np.zeros(engine_count)
        eng_start = 0
        if engine_config != 1.0:  # For now, we will just put a motor on the nose of the aircraft
            motor_pos_x[0] = 0.0
            motor_pos_y[0] = 0.0
            motor_pos_z[0] = 0.0
            motor_rpm_signed[0] = engine_rpm
            eng_per_wing = 1
            # Even if there is no engine of the wing, we put one so that we pick the correct
            # template, the engine will be placed on the nose
        else:
            if (
                engine_count % 2 == 1.0
            ):  # Put one motor on the nose if there is an odd number of engine
                motor_pos_x[0] = 0.0
                motor_pos_y[0] = 0.0
                motor_pos_z[0] = z_wing
                motor_rpm_signed[0] = engine_rpm
                eng_start += 1
                eng_per_wing = int((engine_count - 1) / 2)
            else:
                eng_per_wing = int(engine_count / 2)

            i = 0
            # We put engine on the wings now, later, their position will be described by an array
            # in the xml
            for y_ratio in y_ratio_array:
                y_engine = y_ratio * semi_span

                if y_engine > y2_wing:  # engine in the tapered part of the wing
                    l_wing_eng = l4_wing + (l2_wing - l4_wing) * (y4_wing - y_engine) / (
                        y4_wing - y2_wing
                    )
                    delta_x_eng = 0.05 * l_wing_eng
                    x_eng_rel = (
                        x4_wing * (y_engine - y2_wing) / (y4_wing - y2_wing)
                        - delta_x_eng
                        - nac_length
                    )
                    x_eng = fa_length - 0.25 * l0_wing - (x0_wing - x_eng_rel)

                else:  # engine in the straight part of the wing
                    l_wing_eng = l2_wing
                    delta_x_eng = 0.05 * l_wing_eng
                    x_eng_rel = -delta_x_eng - nac_length
                    x_eng = fa_length - 0.25 * l0_wing - (x0_wing - x_eng_rel)

                if i % 2 == 0:
                    prop_rpm_loop = -engine_rpm
                else:
                    prop_rpm_loop = engine_rpm

                motor_pos_x[eng_start + i] = round(float(x_eng), 2)
                motor_pos_y[eng_start + i] = round(float(y_engine), 2)
                motor_pos_z[eng_start + i] = round(float(z_wing), 2)
                motor_rpm_signed[eng_start + i] = float(prop_rpm_loop)
                motor_pos_x[eng_start + eng_per_wing + i] = round(float(x_eng), 2)
                motor_pos_y[eng_start + eng_per_wing + i] = round(-float(y_engine), 2)
                motor_pos_z[eng_start + eng_per_wing + i] = round(float(z_wing), 2)
                motor_rpm_signed[eng_start + eng_per_wing + i] = -float(prop_rpm_loop)
                i += 1

        # STEP 2/XX - DEFINE WORK DIRECTORY, COPY RESOURCES AND CREATE COMMAND BATCH ###############
        ############################################################################################

        # If a folder path is specified for openvsp .exe, it becomes working directory (target),
        # if not temporary folder is created
        if self.options["openvsp_exe_path"]:
            target_directory = pth.abspath(self.options["openvsp_exe_path"])
        else:
            tmp_directory = _create_tmp_directory()
            target_directory = tmp_directory.name
        # Define the list of necessary input files: geometry script and foil file for both wing/HTP
        input_file_list = [
            pth.join(target_directory, INPUT_WING_ROTOR_SCRIPT),
            pth.join(target_directory, self.options["wing_airfoil_file"]),
        ]
        self.options["external_input_files"] = input_file_list
        # Define standard error file by default to avoid error code return
        self.stderr = pth.join(target_directory, STDERR_FILE_NAME)
        # Copy resource in working (target) directory
        # noinspection PyTypeChecker
        copy_tree(pth.dirname(openvsp3201.__file__), target_directory, verbose=0)
        # noinspection PyTypeChecker
        if self.options["airfoil_folder_path"] is None:
            copy_resource(airfoil_folder, self.options["wing_airfoil_file"], target_directory)
        else:
            copy_resource_from_path(
                self.options["airfoil_folder_path"],
                self.options["wing_airfoil_file"],
                target_directory,
            )
        # Create corresponding .bat files (one for each geometry configuration)
        self.options["command"] = [pth.join(target_directory, "vspscript.bat")]
        batch_file = open(self.options["command"][0], "w+")
        batch_file.write("@echo off\n")
        command = (
            pth.join(target_directory, VSPSCRIPT_EXE_NAME)
            + " -script "
            + pth.join(target_directory, INPUT_WING_ROTOR_SCRIPT)
            + " >nul 2>nul\n"
        )
        batch_file.write(command)
        batch_file.close()

        # STEP 3/XX - OPEN THE TEMPLATE SCRIPT FOR GEOMETRY GENERATION, MODIFY VALUES AND SAVE TO
        # WORKDIR ##################################################################################

        output_file_list = [
            pth.join(
                target_directory, INPUT_WING_ROTOR_SCRIPT.replace(".vspscript", "_DegenGeom.csv")
            )
        ]
        parser = InputFileGenerator()
        with path(local_resources, INPUT_WING_ROTOR_SCRIPT) as input_template_path:
            parser.set_template_file(str(input_template_path))
            parser.set_generated_file(input_file_list[0])
            # Modify wing parameters
            parser.mark_anchor("x_wing")
            parser.transfer_var(float(x_wing), 0, 5)
            parser.mark_anchor("z_wing")
            parser.transfer_var(float(z_wing), 0, 5)
            parser.mark_anchor("y1_wing")
            parser.transfer_var(float(y1_wing), 0, 5)
            for i in range(3):
                parser.mark_anchor("l2_wing")
                parser.transfer_var(float(l2_wing), 0, 5)
            parser.reset_anchor()
            parser.mark_anchor("span2_wing")
            parser.transfer_var(float(span2_wing), 0, 5)
            parser.mark_anchor("l4_wing")
            parser.transfer_var(float(l4_wing), 0, 5)
            parser.mark_anchor("sweep_0_wing")
            parser.transfer_var(float(sweep_0_wing), 0, 5)
            parser.mark_anchor("twist")
            parser.transfer_var(float(twist), 0, 5)
            parser.mark_anchor("dihedral_angle")
            parser.transfer_var(float(dihedral_angle), 0, 5)
            parser.mark_anchor("airfoil_0_file")
            parser.transfer_var('"' + input_file_list[1].replace("\\", "/") + '"', 0, 3)
            parser.mark_anchor("airfoil_1_file")
            parser.transfer_var('"' + input_file_list[1].replace("\\", "/") + '"', 0, 3)
            parser.mark_anchor("airfoil_2_file")
            parser.transfer_var('"' + input_file_list[1].replace("\\", "/") + '"', 0, 3)
            parser.mark_anchor("csv_file")
            csv_name = output_file_list[0]
            parser.transfer_var('"' + csv_name.replace("\\", "/") + '"', 0, 3)
            parser.generate()

        # STEP 4/XX - RUN BATCH TO GENERATE GEOMETRY .CSV FILE #####################################
        ############################################################################################

        self.options["external_output_files"] = output_file_list
        super().compute(inputs, outputs)

        # STEP 5/XX - DEFINE NEW INPUT/OUTPUT FILES LIST AND CREATE BATCH FOR VLM COMPUTATION ######
        ############################################################################################

        input_file_list = output_file_list
        input_file_list.append(input_file_list[0].replace(".csv", ".vspaero"))
        output_file_list = [
            input_file_list[0].replace(".csv", ".lod"),
            input_file_list[0].replace(".csv", ".polar"),
        ]
        self.options["external_input_files"] = input_file_list
        self.options["external_output_files"] = output_file_list
        self.options["command"] = [pth.join(target_directory, "vspaero.bat")]
        batch_file = open(self.options["command"][0], "w+")
        batch_file.write("@echo off\n")
        command = (
            pth.join(target_directory, VSPAERO_EXE_NAME)
            + " "
            + input_file_list[1].replace(".vspaero", "")
            + " >nul 2>nul\n"
        )
        batch_file.write(command)
        batch_file.close()

        # STEP 6/XX - OPEN THE TEMPLATE VSPAERO FOR COMPUTATION, MODIFY VALUES AND SAVE TO WORKDIR #
        ############################################################################################

        parser = InputFileGenerator()

        if engine_config == 1.0:
            rotor_template_file_name = generate_wing_rotor_file(int(engine_count / 2.0))
        else:
            rotor_template_file_name = generate_wing_rotor_file(int(1))

        with path(local_resources, rotor_template_file_name) as input_template_path:
            parser.set_template_file(str(input_template_path))
            parser.set_generated_file(input_file_list[1])
            parser.reset_anchor()
            parser.mark_anchor("Sref")
            parser.transfer_var(float(s_ref_wing), 0, 3)
            parser.mark_anchor("Cref")
            parser.transfer_var(float(l0_wing), 0, 3)
            parser.mark_anchor("Bref")
            parser.transfer_var(float(span_wing), 0, 3)
            parser.mark_anchor("X_cg")
            parser.transfer_var(float(fa_length), 0, 3)
            parser.mark_anchor("Mach")
            parser.transfer_var(float(mach), 0, 3)
            parser.mark_anchor("AOA")
            parser.transfer_var(float(aoa_angle), 0, 3)
            parser.mark_anchor("Vinf")
            parser.transfer_var(float(v_inf), 0, 3)
            parser.mark_anchor("Rho")
            parser.transfer_var(float(rho), 0, 3)
            parser.mark_anchor("ReCref")
            parser.transfer_var(float(reynolds), 0, 3)
            for i in range(1, eng_per_wing + 1):
                parser.mark_anchor("Prop_" + str(i) + "_name")
                parser.transfer_var("Prop_element_" + str(i), 0, 1)
                parser.mark_anchor("Disc_" + str(i) + "_ID")
                parser.transfer_var(i, 0, 1)
                parser.mark_anchor("Disc_" + str(i) + "_x")
                parser.transfer_var(motor_pos_x[i - 1], 0, 1)
                parser.transfer_var(motor_pos_y[i - 1], 0, 2)
                parser.transfer_var(motor_pos_z[i - 1], 0, 3)
                parser.mark_anchor("Disc_" + str(i) + "_nx")
                parser.transfer_var(1.0, 0, 1)
                parser.transfer_var(0.0, 0, 2)
                parser.transfer_var(0.0, 0, 3)
                parser.mark_anchor("Disc_" + str(i) + "_radius")
                parser.transfer_var(prop_radius, 0, 1)
                parser.mark_anchor("Disc_" + str(i) + "_hub_radius")
                parser.transfer_var(prop_hub_radius, 0, 1)
                parser.mark_anchor("Disc_" + str(i) + "_rpm")
                parser.transfer_var(motor_rpm_signed[i - 1], 0, 1)
                parser.mark_anchor("Disc_" + str(i) + "_CT")
                parser.transfer_var(thrust_coefficient, 0, 1)
                parser.mark_anchor("Disc_" + str(i) + "_CP")
                parser.transfer_var(power_coefficient, 0, 1)
            parser.generate()

        os.remove(pth.join(local_resources.__path__[0], rotor_template_file_name))

        # STEP 7/XX - RUN BATCH TO GENERATE AERO OUTPUT FILES (.lod, .polar...) ####################
        ############################################################################################

        super().compute(inputs, outputs)

        # STEP 8/XX - READ FILES, RETURN RESULTS (AND CLEAR TEMPORARY WORKDIR) #####################
        ############################################################################################

        # Open .lod file and extract data
        wing_y_vect = []
        wing_chord_vect = []
        wing_cl_vect = []
        wing_cd_vect = []
        wing_cm_vect = []
        with open(output_file_list[0], "r") as lf:
            data = lf.readlines()
            for i in range(len(data)):
                line = data[i].split()
                line.append("**")
                if line[0] == "1":
                    wing_y_vect.append(float(line[2]))
                    wing_chord_vect.append(float(line[3]))
                    wing_cl_vect.append(float(line[5]))
                    wing_cd_vect.append(float(line[6]))
                    wing_cm_vect.append(float(line[12]))
                if line[0] == "Comp":
                    cl_wing = float(data[i + 1].split()[5]) + float(
                        data[i + 2].split()[5]
                    )  # sum CL left/right
                    cdi_wing = float(data[i + 1].split()[6]) + float(
                        data[i + 2].split()[6]
                    )  # sum CDi left/right
                    cm_wing = float(data[i + 1].split()[12]) + float(
                        data[i + 2].split()[12]
                    )  # sum CM left/right
                    break
        # Open .polar file and extract data
        with open(output_file_list[1], "r") as lf:
            data = lf.readlines()
            wing_e = float(data[1].split()[10])
        # Delete temporary directory
        if not (self.options["openvsp_exe_path"]):
            # noinspection PyUnboundLocalVariable
            tmp_directory.cleanup()
        # Return values
        wing_rotor = {
            "y_vector": wing_y_vect,
            "cl_vector": wing_cl_vect,
            "chord_vector": wing_chord_vect,
            "cd_vector": wing_cd_vect,
            "cm_vector": wing_cm_vect,
            "cl": cl_wing,
            "cdi": cdi_wing,
            "cm": cm_wing,
            "coef_e": wing_e,
            "ct": thrust_coefficient,
        }
        return wing_rotor


def generate_wing_rotor_file(engine_count: int):
    """
    Uses the base VSPAERO template file to generate a file with all the line required to launch
    OpenVSP with n rotors in the run

    :param engine_count: the number of engine in the run

    return the path to the new template file for the n rotor run.
    """

    rotor_template_file_name = "wing_" + str(engine_count) + "_rotor_openvsp_DegenGeom.vspaero"
    original_template = pth.join(local_resources.__path__[0], "wing_openvsp_DegenGeom.vspaero")
    new_template = pth.join(local_resources.__path__[0], rotor_template_file_name)

    file_to_copy = open(original_template, "r").readlines()
    file = open(new_template, "w")

    for i, _ in enumerate(file_to_copy):
        if "NumberOfRotors" in file_to_copy[i]:
            new_line = list(file_to_copy[i][:])
            new_line[-2] = str(engine_count)
            file.write("".join(new_line))
            for j in range(engine_count):
                engine_number = str(int(j + 1))
                file.write("Prop_" + engine_number + "_name\n")
                file.write("Disc_" + engine_number + "_ID\n")
                file.write(
                    "Disc_"
                    + engine_number
                    + "_x Disc_"
                    + engine_number
                    + "_y Disc_"
                    + engine_number
                    + "_z\n"
                )
                file.write(
                    "Disc_"
                    + engine_number
                    + "_nx Disc_"
                    + engine_number
                    + "_ny Disc_"
                    + engine_number
                    + "_nz\n"
                )
                file.write("Disc_" + engine_number + "_radius\n")
                file.write("Disc_" + engine_number + "_hub_radius\n")
                file.write("Disc_" + engine_number + "_rpm\n")
                file.write("Disc_" + engine_number + "_CT\n")
                file.write("Disc_" + engine_number + "_CP\n")
        else:
            file.write(file_to_copy[i])

    file.close()

    return rotor_template_file_name
