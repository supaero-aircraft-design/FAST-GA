"""
Test module for mass breakdown functions.
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

from platform import system

import pytest

import numpy as np

from .test_functions import comp_low_speed

XML_FILE = "partenavia_p68.xml"
SKIP_STEPS = True  # avoid some tests to accelerate validation process (intermediary VLM/OpenVSP)


@pytest.mark.skipif(
    system() != "Windows" or SKIP_STEPS,
    reason="No XFOIL executable available: VLM basic function not computed with "
    "empty result folder (or skipped)",
)
def test_openvsp_comp_low_speed():
    """Tests vlm components @ low speed."""
    y_vector_wing = np.array(
        [
            0.03786,
            0.11357,
            0.18929,
            0.265,
            0.34071,
            0.41643,
            0.49214,
            0.6157,
            0.78762,
            0.9605,
            1.13424,
            1.30875,
            1.48392,
            1.65965,
            1.83581,
            2.01232,
            2.18905,
            2.3659,
            2.54276,
            2.71951,
            2.89605,
            3.07227,
            3.24806,
            3.42331,
            3.59792,
            3.77177,
            3.94477,
            4.11682,
            4.28782,
            4.45767,
            4.62628,
            4.79356,
            4.95942,
            5.12377,
            5.28655,
            5.44766,
            5.60703,
            5.76461,
            5.92031,
        ]
    )
    cl_vector_wing = np.array(
        [
            0.98415808,
            0.98970923,
            0.99004992,
            0.98926835,
            0.98704388,
            0.98697374,
            0.98643265,
            0.98318612,
            0.97279524,
            0.97362691,
            0.97776523,
            0.97471911,
            0.96492943,
            0.95875702,
            0.95463875,
            0.94799539,
            0.93736403,
            0.92885693,
            0.91980874,
            0.91045995,
            0.89688266,
            0.88544968,
            0.87264394,
            0.86041937,
            0.84289415,
            0.82815452,
            0.80993791,
            0.7931642,
            0.77262291,
            0.75319386,
            0.7290854,
            0.70487674,
            0.67452574,
            0.64489619,
            0.60393386,
            0.55915385,
            0.49916328,
            0.41898212,
            0.35356063,
        ]
    )
    chord_vector_wing = np.array(
        [
            1.51419,
            1.51419,
            1.51419,
            1.51419,
            1.51419,
            1.51419,
            1.51419,
            1.51417,
            1.51415,
            1.51412,
            1.5141,
            1.51408,
            1.51406,
            1.51404,
            1.51402,
            1.51401,
            1.51399,
            1.51398,
            1.51397,
            1.51397,
            1.51396,
            1.51396,
            1.51396,
            1.51396,
            1.51396,
            1.51397,
            1.51397,
            1.51398,
            1.51399,
            1.514,
            1.51402,
            1.51403,
            1.51405,
            1.51407,
            1.51408,
            1.51411,
            1.51413,
            1.51415,
            1.51418,
        ]
    )
    y_vector_htp = np.array(
        [
            0.04063,
            0.12188,
            0.20312,
            0.28438,
            0.36563,
            0.44687,
            0.52813,
            0.60938,
            0.69062,
            0.77187,
            0.85313,
            0.93437,
            1.01563,
            1.09687,
            1.17813,
            1.25937,
            1.34062,
            1.42187,
            1.50312,
            1.58438,
            1.66563,
            1.74687,
            1.82813,
            1.90938,
        ]
    )
    cl_vector_htp = np.array(
        [
            0.09837503,
            0.09620438,
            0.0958988,
            0.09597568,
            0.09530931,
            0.0941126,
            0.09284294,
            0.09190843,
            0.09131895,
            0.0902425,
            0.08914633,
            0.08776232,
            0.08577897,
            0.08401643,
            0.08191676,
            0.07980723,
            0.07758335,
            0.07478773,
            0.07099847,
            0.06677349,
            0.06137744,
            0.0537713,
            0.04306594,
            0.02686199,
        ]
    )
    comp_low_speed(
        XML_FILE,
        use_openvsp=True,
        cl0_wing=0.0529865,
        cl_ref_wing=0.82957738,
        cl_alpha_wing=4.44953801,
        cm0=-0.02545115,
        coeff_k_wing=0.04827684,
        cl0_htp=-0.00732,
        cl_alpha_htp=0.50168185,
        cl_alpha_htp_isolated=0.79058386,
        coeff_k_htp=0.71756316,
        y_vector_wing=y_vector_wing,
        cl_vector_wing=cl_vector_wing,
        chord_vector_wing=chord_vector_wing,
        cl_ref_htp=0.08024,
        y_vector_htp=y_vector_htp,
        cl_vector_htp=cl_vector_htp,
    )
