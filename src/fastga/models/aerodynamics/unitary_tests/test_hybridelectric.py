from platform import system

import numpy as np
import pytest

from .dummy_engines import ENGINE_WRAPPER_SR22 as ENGINE_WRAPPER
from .test_functions import (
    cd_inlets,
    cooling_airflow,
)

XML_FILE = "hybrid_electric.xml"
SKIP_STEPS = True  # avoid some tests to accelerate validation process (intermediary VLM/OpenVSP)


def test_cd_inlets():
    """Tests the drag produced by the flush inlets by calculating the drag coefficient."""
    cd_inlets(XML_FILE)


def test_cooling_airflow():
    """Calculates the air-flow needed to cool the condenser of the refrigeration cycle with
    R134a as the refrigerant."""
    cooling_airflow(XML_FILE)


