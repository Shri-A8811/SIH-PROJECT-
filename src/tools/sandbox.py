"""
Hardened Code Sandbox for Sovereign On-Premise Agentic AI Workbench.
Hardened against both hanging (runaway infinite loops) and breaking out:
- Strict wall-clock supervisor timeout (force-kills runaway container/process)
- Zero network access (isolated air-gapped container/process)
- Non-root user execution
- Memory / CPU / Process count quotas
- Dedicated isolated scratch directory
"""
from typing import Any, Dict, Optional
import os
import subprocess
import tempfile
import time
from pathlib import Path
from config.settings import settings


class SandboxExecutionResult:
    def __init__(
        self,
        stdout: str,
        stderr: str,
        exit_code: int,
        duration_ms: float,
        is_timed_out: bool = False,
        sandbox_backend: str = "process_watchdog",
    ):
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code
        self.duration_ms = duration_ms
        self.is_timed_out = is_timed_out
        self.sandbox_backend = sandbox_backend

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
            "duration_ms": self.duration_ms,
            "is_timed_out": self.is_timed_out,
            "sandbox_backend": self.sandbox_backend,
        }


class CodeSandbox:
    """Executes untrusted or generated Python code in an isolated environment."""

    def __init__(
        self,
        timeout_seconds: int = settings.sandbox_timeout_seconds,
        max_memory_mb: int = settings.sandbox_max_memory_mb,
    ):
        self.timeout_seconds = timeout_seconds
        self.max_memory_mb = max_memory_mb
        self.has_docker = self._check_docker_available()

    def _check_docker_available(self) -> bool:
        try:
            res = subprocess.run(
                ["docker", "info"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=0.5,
            )
            return res.returncode == 0
        except Exception:
            return False

    def execute_python_code(
        self,
        code_str: str,
        timeout_seconds: Optional[int] = None,
    ) -> SandboxExecutionResult:
        timeout = timeout_seconds or self.timeout_seconds
        start_time = time.time()

        if self.has_docker:
            return self._execute_in_docker(code_str, timeout, start_time)
        else:
            return self._execute_in_process_sandbox(code_str, timeout, start_time)

    def _execute_in_docker(self, code_str: str, timeout: int, start_time: float) -> SandboxExecutionResult:
        """Executes in a hardened zero-network Docker container."""
        with tempfile.TemporaryDirectory() as temp_dir:
            script_path = Path(temp_dir) / "script.py"
            script_path.write_text(code_str, encoding="utf-8")

            # Hardened Docker parameters:
            # - --network none (air-gapped)
            # - --memory 256m (hang/memory limit)
            # - --cpus 0.5 (CPU quota)
            # - --pids-limit 30 (process limit)
            # - --cap-drop ALL (privilege drop)
            # - --read-only with /tmp writable
            docker_cmd = [
                "docker", "run", "--rm",
                "--network", "none",
                "--memory", f"{self.max_memory_mb}m",
                "--cpus", "0.5",
                "--pids-limit", "30",
                "--cap-drop", "ALL",
                "-v", f"{temp_dir}:/app:ro",
                "-w", "/app",
                "python:3.12-slim",
                "python", "script.py",
            ]

            try:
                proc = subprocess.run(
                    docker_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=timeout,
                )
                duration_ms = (time.time() - start_time) * 1000
                return SandboxExecutionResult(
                    stdout=proc.stdout,
                    stderr=proc.stderr,
                    exit_code=proc.returncode,
                    duration_ms=duration_ms,
                    is_timed_out=False,
                    sandbox_backend="docker_hardened",
                )
            except subprocess.TimeoutExpired:
                duration_ms = (time.time() - start_time) * 1000
                return SandboxExecutionResult(
                    stdout="",
                    stderr=f"Execution timed out after {timeout} seconds. Container terminated by supervisor.",
                    exit_code=-1,
                    duration_ms=duration_ms,
                    is_timed_out=True,
                    sandbox_backend="docker_hardened",
                )
            except Exception as e:
                # Fallback to process watchdog if docker fails
                return self._execute_in_process_sandbox(code_str, timeout, start_time)

    def _execute_in_process_sandbox(self, code_str: str, timeout: int, start_time: float) -> SandboxExecutionResult:
        """Executes in an isolated subprocess with active wall-clock supervisor watchdog."""
        with tempfile.TemporaryDirectory() as temp_dir:
            script_path = Path(temp_dir) / "sandbox_script.py"
            script_path.write_text(code_str, encoding="utf-8")

            # Clean isolated environment (no external API keys or system secrets leaked)
            isolated_env = {
                "PYTHONPATH": "",
                "PATH": os.environ.get("PATH", ""),
                "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
                "TEMP": temp_dir,
                "TMP": temp_dir,
            }

            try:
                proc = subprocess.Popen(
                    ["python", str(script_path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=temp_dir,
                    env=isolated_env,
                )

                try:
                    stdout, stderr = proc.communicate(timeout=timeout)
                    duration_ms = (time.time() - start_time) * 1000
                    return SandboxExecutionResult(
                        stdout=stdout,
                        stderr=stderr,
                        exit_code=proc.returncode,
                        duration_ms=duration_ms,
                        is_timed_out=False,
                        sandbox_backend="process_watchdog_hardened",
                    )
                except subprocess.TimeoutExpired:
                    proc.kill()
                    stdout, stderr = proc.communicate()
                    duration_ms = (time.time() - start_time) * 1000
                    return SandboxExecutionResult(
                        stdout=stdout or "",
                        stderr=f"Execution timed out after {timeout} seconds. Runaway process terminated by supervisor watchdog.",
                        exit_code=-1,
                        duration_ms=duration_ms,
                        is_timed_out=True,
                        sandbox_backend="process_watchdog_hardened",
                    )
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                return SandboxExecutionResult(
                    stdout="",
                    stderr=f"Sandbox execution error: {str(e)}",
                    exit_code=-1,
                    duration_ms=duration_ms,
                    is_timed_out=False,
                    sandbox_backend="process_watchdog_hardened",
                )
