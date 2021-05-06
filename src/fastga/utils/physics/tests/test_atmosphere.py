"""Tests for Atmosphere class"""
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


from numpy.testing import assert_allclose

from ..atmosphere import Atmosphere


def test_speed_conversions():
    """Tests for speed conversions."""
    # noinspection PyTypeChecker
    atm = Atmosphere([[0.0, 1000.0, 35000.0], [0.0, 1000.0, 35000.0]])
    TAS = [[100.0, 100.0, 100.0], [800.0, 800.0, 800.0]]

    # source:  http://www.aerospaceweb.org/design/scripts/atmosphere/
    expected_EAS = [[100.0, 98.5797, 56.2686], [800.0, 795.4567, 630.3015]]

    assert_allclose(atm.get_equivalent_airspeed(TAS), expected_EAS, rtol=3e-3)

    assert_allclose(atm.get_true_airspeed(expected_EAS), TAS, rtol=3e-3)
