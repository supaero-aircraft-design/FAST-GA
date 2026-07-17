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

import contextlib
import importlib.metadata as importlib_metadata
from typing import ClassVar
from unittest.mock import Mock

import pytest
import wrapt

from fastoad.module_management._plugins import MODEL_PLUGIN_ID, FastoadLoader


@pytest.fixture
def with_dummy_plugin_1():
    """
    Reduces plugin list to dummy-dist-1 with plugin test_plugin_1
    (one configuration file, no models, notebook folder, no source data files).

    Any previous state of plugins is restored during teardown.
    """
    _setup()
    dummy_dist_1 = Mock(importlib_metadata.Distribution)
    dummy_dist_1.name = "dummy-dist-1"
    new_entry_points = [
        importlib_metadata.EntryPoint(
            name="test_plugin_1",
            value="tests.dummy_plugins.dummy_plugin_1",
            group=MODEL_PLUGIN_ID,
        )
    ]
    new_entry_points[0].dist = dummy_dist_1
    _update_entry_map(new_entry_points)
    yield
    _teardown()


def _update_entry_map(new_plugin_entry_points: list[importlib_metadata.EntryPoint]):
    """
    Modified plugin entry_points of FAST-OAD distribution.

    This is done by replacing the entry_points property of Distribution class

    :param new_plugin_entry_points:
    """
    BypassEntryPointReading.entry_points = new_plugin_entry_points
    BypassEntryPointReading.active = True
    FastoadLoader._loaded = False


def _setup():
    MakeEntryPointMutable.active = True


def _teardown():
    MakeEntryPointMutable.active = False
    BypassEntryPointReading.active = False
    FastoadLoader._loaded = False


# Monkey-patching using wrapt module ###########################################


def _BypassEntryPointReading_enabled():  # noqa: N802 more readable with camelcase
    return BypassEntryPointReading.active


class BypassEntryPointReading:
    active: bool = False
    entry_points: ClassVar[list] = []

    @wrapt.decorator(enabled=_BypassEntryPointReading_enabled)
    def __call__(self, wrapped, instance, args, kwargs):
        if kwargs.get("group") == MODEL_PLUGIN_ID:
            return self.entry_points
        return wrapped(*args, **kwargs)


importlib_metadata.entry_points = BypassEntryPointReading()(importlib_metadata.entry_points)


def _MakeEntryPointMutable_enabled():  # noqa: N802 more readable with camelcase
    return MakeEntryPointMutable.active


class MakeEntryPointMutable:
    active = True

    @classmethod
    def _enabled(cls):
        return cls.active

    @wrapt.decorator(enabled=_MakeEntryPointMutable_enabled)
    def __call__(self, wrapped, instance, args, kwargs):
        with contextlib.suppress(AttributeError):
            delattr(wrapped, "__setattr__")
        return wrapped(*args, **kwargs)


importlib_metadata.EntryPoint = MakeEntryPointMutable()(importlib_metadata.EntryPoint)
