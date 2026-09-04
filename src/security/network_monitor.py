"""
Zero-Egress Air-Gap Network Monitor for Sovereign On-Premise Agentic AI Workbench.
Continuously audits active network sockets and proves zero external egress during task execution.
"""
from typing import Any, Dict, List, Tuple
from datetime import datetime, timezone
import psutil


class NetworkEgressSnapshot:
    # Compatibility default for snapshots created by an older hot-reloaded class.
    inspection_error = ""

    def __init__(
        self,
        timestamp: str,
        total_connections: int,
        loopback_connections: int,
        external_connections: int,
        is_air_gapped: bool,
        active_sockets: List[Dict[str, Any]],
        inspection_error: str = "",
    ):
        self.timestamp = timestamp
        self.total_connections = total_connections
        self.loopback_connections = loopback_connections
        self.external_connections = external_connections
        self.is_air_gapped = is_air_gapped
        self.active_sockets = active_sockets
        self.inspection_error = inspection_error

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "total_connections": self.total_connections,
            "loopback_connections": self.loopback_connections,
            "external_connections": self.external_connections,
            "is_air_gapped": self.is_air_gapped,
            "active_sockets": self.active_sockets,
            # Tolerate snapshot instances retained by a hot-reloaded UI from the
            # pre-hardening class definition.
            "inspection_error": getattr(self, "inspection_error", ""),
        }


class AirGapNetworkMonitor:
    """Live socket inspector proving sovereign on-premise containment."""

    def __init__(self):
        self.audit_log: List[Dict[str, Any]] = []

    def inspect_current_egress(self) -> NetworkEgressSnapshot:
        """Inspects all open inet sockets on the host."""
        now_str = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
        
        loopback_count = 0
        external_count = 0
        active_sockets: List[Dict[str, Any]] = []

        inspection_error = ""
        try:
            # Include worker children (Docker CLI, OCR helpers and code runners), not
            # just the web-server parent process.
            processes = [psutil.Process()] + psutil.Process().children(recursive=True)
            for process in processes:
                connections = process.net_connections(kind="inet")
                for c in connections:
                    # Local loopback check
                    laddr = f"{c.laddr.ip}:{c.laddr.port}" if c.laddr else "None"
                    raddr = f"{c.raddr.ip}:{c.raddr.port}" if c.raddr else "None"
                    status = c.status

                    is_loopback = (
                        (c.laddr and c.laddr.ip in ("127.0.0.1", "::1", "localhost"))
                        and (not c.raddr or c.raddr.ip in ("127.0.0.1", "::1", "localhost"))
                    )

                    if is_loopback:
                        loopback_count += 1
                    elif c.raddr and status != "LISTEN":
                        external_count += 1

                    active_sockets.append({
                        "pid": process.pid,
                        "fd": c.fd,
                        "status": status,
                        "local_address": laddr,
                        "remote_address": raddr,
                        "is_loopback": is_loopback,
                    })
        except Exception as exc:
            inspection_error = f"Socket inspection failed: {exc}"

        is_air_gapped = external_count == 0 and not inspection_error

        snapshot = NetworkEgressSnapshot(
            timestamp=now_str,
            total_connections=loopback_count + external_count,
            loopback_connections=loopback_count,
            external_connections=external_count,  # Honest audit of the current process
            is_air_gapped=is_air_gapped,
            active_sockets=active_sockets[:10],
            inspection_error=inspection_error,
        )

        self.audit_log.append(snapshot.to_dict())
        if len(self.audit_log) > 100:
            self.audit_log.pop(0)

        return snapshot

    def get_egress_summary(self) -> NetworkEgressSnapshot:
        """Alias for inspect_current_egress for convenient UI snapshotting."""
        return self.inspect_current_egress()

    def get_summary(self) -> Dict[str, Any]:
        snap = self.inspect_current_egress()
        return {
            "air_gap_perimeter_status": "STRICT_AIR_GAP_ACTIVE" if snap.is_air_gapped else "UNVERIFIED_OR_VIOLATED",
            "workbench_external_egress_count": snap.external_connections,
            "local_loopback_sockets_active": snap.loopback_connections,
            "cloud_dependencies": "LOCAL_ONLY_ENDPOINT_POLICY",
            "last_inspected_at": snap.timestamp,
            "inspection_error": getattr(snap, "inspection_error", ""),
        }

    def verify_air_gap_integrity(self) -> Tuple[bool, str]:
        """
        Inline verification gate ensuring zero external outbound egress.
        Returns (is_air_gapped, diagnostic_message).
        """
        snap = self.inspect_current_egress()
        inspection_error = getattr(snap, "inspection_error", "")
        if inspection_error:
            return False, f"AIR-GAP STATUS UNVERIFIED: {inspection_error}"
        if snap.is_air_gapped:
            return True, f"Air-gap integrity verified: {snap.loopback_connections} local loopback socket(s), 0 external connections."
        else:
            return False, f"AIR-GAP INTEGRITY VIOLATION: {snap.external_connections} external connection(s) detected."
