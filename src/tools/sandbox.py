"""
Hardened Code Sandbox for Sovereign On-Premise Agentic AI Workbench.
Hardened against both hanging (runaway infinite loops) and breaking out:
- Strict wall-clock supervisor timeout (force-kills runaway container/process)
- Zero network access (isolated air-gapped container/process)
- Non-root user execution
- Memory / CPU / Process count quotas
- Dedicated isolated scratch directory
"""
from typing import Any, Dict, List, Optional, Tuple
import os
import ast
import subprocess
import tempfile
import time
from pathlib import Path
from config.settings import settings


class SecurityASTValidator(ast.NodeVisitor):
    """
    Statically analyzes untrusted Python AST before running in subprocess/container.
    Bans unauthorized network modules, process spawners, and system manipulation.
    """

    FORBIDDEN_MODULES = {
        "socket",
        "requests",
        "urllib",
        "urllib3",
        "http",
        "ftplib",
        "telnetlib",
        "poplib",
        "imaplib",
        "smtplib",
        "paramiko",
        "subprocess",
        "winreg",
        "ctypes",
        "multiprocessing",
        "threading",
    }

    FORBIDDEN_CALLS = {
        "eval",
        "exec",
        "__import__",
        "globals",
        "locals",
    }

    def __init__(self):
        self.violations: List[str] = []

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            root_mod = alias.name.split(".")[0]
            if root_mod in self.FORBIDDEN_MODULES:
                self.violations.append(f"Forbidden module import: '{alias.name}' (violates zero-egress sandbox policy)")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            root_mod = node.module.split(".")[0]
            if root_mod in self.FORBIDDEN_MODULES:
                self.violations.append(f"Forbidden module import from: '{node.module}' (violates zero-egress sandbox policy)")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "os":
                if node.func.attr in {"system", "popen", "spawn", "exec", "execl", "execle", "execlp", "execv", "execve", "fork"}:
                    self.violations.append(f"Forbidden system call: 'os.{node.func.attr}'")
        elif isinstance(node.func, ast.Name):
            if node.func.id in self.FORBIDDEN_CALLS:
                self.violations.append(f"Forbidden builtin call: '{node.func.id}()'")
        self.generic_visit(node)


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

    def validate_code_security(self, code_str: str) -> Tuple[bool, List[str]]:
        """Performs pre-execution static AST security analysis."""
        try:
            tree = ast.parse(code_str)
        except SyntaxError:
            # Let the runner or compiler report syntax errors normally
            return True, []

        validator = SecurityASTValidator()
        validator.visit(tree)
        if validator.violations:
            return False, validator.violations
        return True, []

    def _check_docker_available(self) -> bool:
        if os.getenv("WORKBENCH_TEST_MODE") or os.getenv("DISABLE_DOCKER_SANDBOX"):
            return False
        try:
            res = subprocess.run(
                ["docker", "info"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=1.0,
            )
            return res.returncode == 0 and b"Server Version" in res.stdout
        except Exception:
            return False

    def execute_python_code(
        self,
        code_str: str,
        timeout_seconds: Optional[int] = None,
    ) -> SandboxExecutionResult:
        # Pre-execution check: Guard against natural language prompts passed to sandbox
        trimmed = (code_str or "").strip()
        first_line = trimmed.split("\n")[0].strip().lower()
        if any(first_line.startswith(p) for p in ("write ", "give me", "create ", "implement ", "how to ", "make me ", "code for ")) and not any(first_line.startswith(p) for p in ("def ", "class ", "import ", "from ", "print(")):
            return SandboxExecutionResult(
                stdout="",
                stderr=f"EXECUTION GUARD: Input appears to be a natural language coding request ('{first_line[:60]}...'), not executable Python code. Route to coding model for generation instead.",
                exit_code=-1,
                duration_ms=0.0,
                is_timed_out=False,
                sandbox_backend="prompt_guard",
            )

        # 1. Pre-execution static AST security check
        is_safe, violations = self.validate_code_security(code_str)
        if not is_safe:
            return SandboxExecutionResult(
                stdout="",
                stderr="SECURITY POLICY VIOLATION: " + "; ".join(violations),
                exit_code=-2,
                duration_ms=0.0,
                is_timed_out=False,
                sandbox_backend="ast_security_shield",
            )

        timeout = timeout_seconds or self.timeout_seconds
        start_time = time.time()

        if not self.has_docker:
            return SandboxExecutionResult(
                stdout="",
                stderr="Sandbox unavailable: a preloaded Docker runtime is mandatory. Host-process execution is disabled.",
                exit_code=-3,
                duration_ms=(time.time() - start_time) * 1000,
                sandbox_backend="unavailable_fail_closed",
            )
        return self._execute_in_docker(code_str, timeout, start_time)

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
                "--pull", "never",
                "--network", "none",
                "--memory", f"{self.max_memory_mb}m",
                "--cpus", "0.5",
                "--pids-limit", "30",
                "--cap-drop", "ALL",
                "--security-opt", "no-new-privileges:true",
                "--user", "65534:65534",
                "--read-only",
                "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
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
                duration_ms = (time.time() - start_time) * 1000
                return SandboxExecutionResult(
                    stdout="",
                    stderr=f"Hardened Docker sandbox failed; host fallback is disabled: {e}",
                    exit_code=-3,
                    duration_ms=duration_ms,
                    sandbox_backend="docker_unavailable_fail_closed",
                )

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

            import sys
            try:
                proc = subprocess.Popen(
                    [sys.executable, str(script_path)],
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
