from __future__ import annotations

import pytest

from config import smoke_profile


@pytest.fixture
def smoke_config(tmp_path):
    """All test artifacts stay in pytest's temporary directory, never output/."""
    return smoke_profile(tmp_path / "isolated_smoke")
