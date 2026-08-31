"""
Tests for Zero-Egress Air-Gap Network Monitor.
"""
from src.security.network_monitor import AirGapNetworkMonitor


def test_network_monitor_zero_egress_assertion():
    netmon = AirGapNetworkMonitor()
    summary = netmon.get_summary()
    assert summary["air_gap_perimeter_status"] == "STRICT_AIR_GAP_ACTIVE"
    assert summary["workbench_external_egress_count"] == 0
    assert summary["cloud_dependencies"] == "ZERO_CLOUD_TELEMETRY"

    snap = netmon.inspect_current_egress()
    assert snap.is_air_gapped is True
    assert snap.external_connections == 0
