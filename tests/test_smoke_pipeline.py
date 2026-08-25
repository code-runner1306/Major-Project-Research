from __future__ import annotations

from artifacts import read_json
from orchestration import run_cpu_smoke


def test_complete_cpu_smoke_pipeline(smoke_config):
    result = run_cpu_smoke(smoke_config)
    assert result["smoke"]["elapsed_seconds"] < 300
    assert result["smoke"]["measured_rss_mb_at_completion"] is None or result["smoke"]["measured_rss_mb_at_completion"] < 2048
    assert result["smoke"]["network_required"] is False
    assert (smoke_config.output_dir / "aggregate_report.json").is_file()
    assert (smoke_config.output_dir / "plots" / "quality_and_convergence.png").is_file()
    assert read_json(smoke_config.output_dir / "smoke_result.json")["status"] == "complete"
