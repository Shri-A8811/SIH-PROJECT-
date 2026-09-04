"""
Comprehensive Verification Suite for Component 4: Deterministic Tools & Sandbox Hardening.
Validates:
1. ASME B31.3 Sec 304.1.2 process piping minimum required wall thickness calculations.
2. API 570 Sec 7.1 corrosion rate, remaining useful life (RUL), and half-life inspection intervals.
3. AST arithmetic & math expression evaluation (sqrt, abs, round, log10) + recursion depth watchdog.
4. Pre-execution static AST security scanner blocking forbidden network modules and OS process spawners.
5. Autonomous Agentic Orchestrator tool dispatch for ASME and API 570 calculations.
6. Artifact Verifier numeric tolerance matching and table integrity checks.
"""
import pytest
from pathlib import Path
from src.tools.calculator import DeterministicCalculator
from src.tools.sandbox import CodeSandbox, SecurityASTValidator
from src.core.state_store import StateStore
from src.core.orchestrator import AgenticOrchestrator
from src.generation.docx_generator import DocxApprovalNoteGenerator
from src.generation.verifier import ArtifactVerifier


@pytest.fixture
def state_store(tmp_path):
    db_file = tmp_path / "test_comp4_store.db"
    return StateStore(database_url=f"sqlite:///{db_file}")


def test_asme_b31_3_min_thickness_compliant_and_breached():
    calc = DeterministicCalculator()
    
    # Standard refinery transfer line: P = 35.0 bar (3.5 MPa), D = 219.1 mm, S = 115 MPa, E = 1.0, Y = 0.4, c = 1.5 mm
    # t_pressure = (3.5 * 219.1) / (2 * (115*1.0 + 3.5*0.4)) = 766.85 / 232.8 = 3.294 mm
    # t_min = 3.294 + 1.5 = 4.794 mm
    res = calc.calculate_asme_b31_3_min_thickness(
        design_pressure_bar=35.0,
        outside_diameter_mm=219.1,
        allowable_stress_mpa=115.0,
        weld_efficiency_e=1.0,
        temp_factor_y=0.4,
        corrosion_allowance_mm=1.5,
        measured_thickness_mm=3.42,
    )
    
    assert res["standard"] == "ASME B31.3"
    assert abs(res["min_required_thickness_mm"] - 4.794) < 0.01
    assert res["is_compliant"] is False
    assert res["status"] == "NON_COMPLIANT_RETIREMENT_REQUIRED"
    assert res["margin_mm"] < 0
    assert len(res["audit_trail"]) >= 6

    # Test compliant reading: 5.50 mm
    res_compliant = calc.calculate_asme_b31_3_min_thickness(
        design_pressure_bar=35.0,
        outside_diameter_mm=219.1,
        allowable_stress_mpa=115.0,
        measured_thickness_mm=5.50,
    )
    assert res_compliant["is_compliant"] is True
    assert res_compliant["status"] == "COMPLIANT"
    assert res_compliant["margin_mm"] > 0


def test_asme_b31_3_invalid_inputs():
    calc = DeterministicCalculator()
    with pytest.raises(ValueError):
        calc.calculate_asme_b31_3_min_thickness(-10.0, 200.0, 100.0)
    with pytest.raises(ValueError):
        calc.calculate_asme_b31_3_min_thickness(10.0, 0.0, 100.0)


def test_api_570_corrosion_rate_and_rul():
    calc = DeterministicCalculator()
    
    # 8.0 mm down to 6.2 mm over 5 years (loss = 1.8 mm, CR = 0.36 mm/yr)
    # Required = 4.8 mm -> margin = 1.4 mm -> RUL = 1.4 / 0.36 = 3.89 years
    # API 570 Next Inspection = min(3.89 / 2, 5.0) = 1.95 years
    res = calc.calculate_corrosion_rate_and_rul(
        previous_thickness_mm=8.0,
        current_thickness_mm=6.2,
        time_interval_years=5.0,
        required_thickness_mm=4.8,
    )
    
    assert res["standard"] == "API 570"
    assert abs(res["corrosion_rate_mm_per_year"] - 0.36) < 0.001
    assert abs(res["remaining_useful_life_years"] - 3.89) < 0.02
    assert abs(res["api_570_next_inspection_interval_years"] - 1.95) < 0.02
    assert res["status"] == "ELEVATED_RISK_MONITORING"


def test_api_570_zero_corrosion_and_immediate_breach():
    calc = DeterministicCalculator()
    
    # Zero corrosion
    res_zero = calc.calculate_corrosion_rate_and_rul(
        previous_thickness_mm=8.0,
        current_thickness_mm=8.0,
        time_interval_years=4.0,
        required_thickness_mm=4.8,
    )
    assert res_zero["corrosion_rate_mm_per_year"] == 0.0
    assert res_zero["remaining_useful_life_years"] == 999.0
    assert res_zero["status"] == "NO_ACTIVE_CORROSION"

    # Already breached
    res_breach = calc.calculate_corrosion_rate_and_rul(
        previous_thickness_mm=5.0,
        current_thickness_mm=3.2,
        time_interval_years=2.0,
        required_thickness_mm=4.0,
    )
    assert res_breach["remaining_useful_life_years"] == 0.0
    assert res_breach["api_570_next_inspection_interval_years"] == 0.0
    assert res_breach["status"] == "CRITICAL_RETIREMENT_BREACH"


def test_compute_expression_hardened_math():
    calc = DeterministicCalculator()
    
    # Math functions
    res = calc.compute_expression("sqrt(144) + abs(-20) + round(3.14159, 2) + log10(100)")
    # 12 + 20 + 3.14 + 2 = 37.14
    assert abs(res["result"] - 37.14) < 0.001

    # Unauthorized function rejected
    with pytest.raises(ValueError, match="Unsupported or unauthorized"):
        calc.compute_expression("__import__('os').system('dir')")


def test_compute_expression_recursion_depth_watchdog():
    calc = DeterministicCalculator()
    
    # Build deeply nested expression > 20 levels
    deep_expr = "1"
    for _ in range(25):
        deep_expr = f"({deep_expr} + 1)"
        
    with pytest.raises(ValueError, match="maximum allowable AST recursion depth"):
        calc.compute_expression(deep_expr)


def test_sandbox_ast_blocks_forbidden_imports():
    sb = CodeSandbox()
    
    # 1. socket
    r1 = sb.execute_python_code("import socket\ns = socket.socket()")
    assert r1.exit_code == -2
    assert "Forbidden module import: 'socket'" in r1.stderr
    assert r1.sandbox_backend == "ast_security_shield"

    # 2. requests / urllib
    r2 = sb.execute_python_code("from urllib.request import urlopen")
    assert r2.exit_code == -2
    assert "Forbidden module import from: 'urllib.request'" in r2.stderr

    # 3. ctypes
    r3 = sb.execute_python_code("import ctypes")
    assert r3.exit_code == -2
    assert "Forbidden module import: 'ctypes'" in r3.stderr

    # 4. subprocess
    r4 = sb.execute_python_code("import subprocess")
    assert r4.exit_code == -2
    assert "Forbidden module import: 'subprocess'" in r4.stderr


def test_sandbox_ast_blocks_forbidden_calls():
    sb = CodeSandbox()
    
    # os.system
    r1 = sb.execute_python_code("import os\nos.system('whoami')")
    assert r1.exit_code == -2
    assert "Forbidden system call: 'os.system'" in r1.stderr

    # eval
    r2 = sb.execute_python_code("eval('1 + 1')")
    assert r2.exit_code == -2
    assert "Forbidden builtin call: 'eval()'" in r2.stderr


def test_sandbox_ast_allows_safe_code():
    sb = CodeSandbox()
    safe_code = """
import math
def calculate_area(radius):
    return math.pi * (radius ** 2)

print(round(calculate_area(5.0), 2))
"""
    res = sb.execute_python_code(safe_code)
    assert res.exit_code == 0
    assert "78.54" in res.stdout.strip()


def test_orchestrator_asme_and_api_tools(state_store):
    orch = AgenticOrchestrator(state_store)
    
    # Verify tool registry contains new tools
    tool_names = {t.name for t in orch.get_registered_tools()}
    assert "calculate_asme_b31_3" in tool_names
    assert "calculate_corrosion_rate_and_rul" in tool_names
    assert len(tool_names) >= 8

    # Test direct execution of calculate_asme_b31_3
    res_asme = orch.execute_tool(
        tool_name="calculate_asme_b31_3",
        params={
            "design_pressure_bar": 35.0,
            "outside_diameter_mm": 219.1,
            "allowable_stress_mpa": 115.0,
            "measured_thickness_mm": 3.42,
        },
        project_id="test_comp4_p1",
    )
    assert res_asme["status"] == "calculated"
    assert res_asme["data"]["is_compliant"] is False

    # Test direct execution of calculate_corrosion_rate_and_rul
    res_api = orch.execute_tool(
        tool_name="calculate_corrosion_rate_and_rul",
        params={
            "previous_thickness_mm": 8.0,
            "current_thickness_mm": 6.2,
            "time_interval_years": 5.0,
            "required_thickness_mm": 4.8,
        },
        project_id="test_comp4_p1",
    )
    assert res_api["status"] == "calculated"
    assert abs(res_api["data"]["corrosion_rate_mm_per_year"] - 0.36) < 0.001


def test_docx_verifier_numeric_tolerance_and_tables(state_store, tmp_path):
    gen = DocxApprovalNoteGenerator(state_store)
    verifier = ArtifactVerifier(state_store)

    findings = [
        {
            "equipment": "CDU-1 Transfer Line P-104B",
            "issue": "Ultrasonic wall thinning to 3.42 mm below retirement threshold.",
            "severity": "Critical",
            "evidence_id": "E_NDT_104",
            "measured_value": "3.42 mm",
            "threshold_value": "4.80 mm",
            "status": "NON-COMPLIANT",
        }
    ]

    doc_path = tmp_path / "test_tolerance_report.docx"
    gen_res = gen.generate_approval_note(
        project_id="PROJ_TOL_TEST",
        title="MRPL Engineering Review Note",
        executive_summary="Comprehensive structural audit of CDU-1 transfer piping with measured 3.420 mm thickness.",
        findings=findings,
        output_filename=str(doc_path),
    )

    actual_file_path = gen_res["file_path"]

    # Verify with floating point number that matches within 0.01
    rep = verifier.verify_docx_deliverable(
        artifact_id="A_TOL_001",
        file_path=actual_file_path,
        expected_numeric_values=["3.42"],
    )

    assert rep.is_passed is True
    assert len(rep.errors) == 0

    # Check table integrity check was executed
    checks_dict = {c["check"]: c["passed"] for c in rep.checks_performed}
    assert "contains_human_review_disclaimer" in checks_dict
    assert "numeric_match_3.42" in checks_dict
    assert "table_1_integrity" in checks_dict
