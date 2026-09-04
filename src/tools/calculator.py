"""
Deterministic Calculator Tool for Sovereign On-Premise Agentic AI Workbench.
Any numeric claim or threshold comparison in a generated document is computed here.
The language model is only asked to explain numbers computed by this tool, never to compute them.
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class CalculationResult(BaseModel):
    operation: str
    result: float
    unit: str = ""
    audit_trail: List[str] = Field(default_factory=list)
    is_threshold_breached: Optional[bool] = None
    severity_level: Optional[str] = None


class DeterministicCalculator:
    """Deterministic arithmetic calculator with audit logging."""

    @staticmethod
    def calculate_wall_thinning_deviation(
        measured_thickness_mm: float,
        nominal_thickness_mm: float,
        retirement_thickness_mm: float,
    ) -> Dict[str, Any]:
        """
        Computes exact corrosion loss, residual percentage, and retirement breach.
        Formula:
          Total Loss = Nominal - Measured
          Retirement Deviation % = ((Retirement - Measured) / Retirement) * 100
        """
        if nominal_thickness_mm <= 0 or retirement_thickness_mm <= 0:
            raise ValueError("Nominal thickness and retirement thickness must be strictly positive numbers > 0.")

        total_loss = round(nominal_thickness_mm - measured_thickness_mm, 4)
        loss_pct_of_nominal = round((total_loss / nominal_thickness_mm) * 100, 2)
        
        is_breached = measured_thickness_mm < retirement_thickness_mm
        breach_margin_mm = round(retirement_thickness_mm - measured_thickness_mm, 4)
        
        if is_breached:
            deviation_pct = round((breach_margin_mm / retirement_thickness_mm) * 100, 2)
            severity = "Critical" if deviation_pct > 20.0 else "High"
        else:
            deviation_pct = 0.0
            severity = "Compliant"

        audit_steps = [
            f"1. Nominal Wall Thickness = {nominal_thickness_mm:.2f} mm",
            f"2. Measured Ultrasonic Residual Thickness = {measured_thickness_mm:.2f} mm",
            f"3. Total Metal Loss = {nominal_thickness_mm:.2f} - {measured_thickness_mm:.2f} = {total_loss:.2f} mm ({loss_pct_of_nominal}% loss from nominal)",
            f"4. Minimum Safe Retirement Thickness (SOP-17) = {retirement_thickness_mm:.2f} mm",
            f"5. Retirement Threshold Breach = {is_breached} (Measured {measured_thickness_mm:.2f} mm vs Limit {retirement_thickness_mm:.2f} mm)",
            f"6. Calculated Breach Margin = {breach_margin_mm:.2f} mm ({deviation_pct}% below allowable safety limit)",
        ]

        return {
            "operation": "wall_thinning_deviation",
            "measured_thickness_mm": measured_thickness_mm,
            "nominal_thickness_mm": nominal_thickness_mm,
            "retirement_thickness_mm": retirement_thickness_mm,
            "total_loss_mm": total_loss,
            "loss_percentage_nominal": loss_pct_of_nominal,
            "is_threshold_breached": is_breached,
            "breach_margin_mm": breach_margin_mm,
            "deviation_percentage_below_retirement": deviation_pct,
            "severity_level": severity,
            "audit_trail": audit_steps,
        }

    @staticmethod
    def evaluate_pressure_differential(
        operating_pressure_bar: float,
        design_pressure_bar: float,
        hydro_test_pressure_bar: float,
    ) -> Dict[str, Any]:
        """Calculates safety margins under operating and hydrostatic test conditions."""
        op_margin = round(design_pressure_bar - operating_pressure_bar, 2)
        op_margin_pct = round((op_margin / design_pressure_bar) * 100, 2)
        hydro_factor = round(hydro_test_pressure_bar / design_pressure_bar, 2)
        
        audit_steps = [
            f"1. Operating Pressure = {operating_pressure_bar} bar, Design Pressure = {design_pressure_bar} bar",
            f"2. Operating Safety Margin = {op_margin} bar ({op_margin_pct}% buffer)",
            f"3. Hydro-test Pressure = {hydro_test_pressure_bar} bar (Design Multiplier = {hydro_factor}x)",
        ]

        return {
            "operation": "pressure_differential",
            "operating_pressure_bar": operating_pressure_bar,
            "design_pressure_bar": design_pressure_bar,
            "hydro_test_pressure_bar": hydro_test_pressure_bar,
            "operating_margin_bar": op_margin,
            "operating_margin_pct": op_margin_pct,
            "hydro_design_multiplier": hydro_factor,
            "audit_trail": audit_steps,
        }

    @staticmethod
    def calculate_asme_b31_3_min_thickness(
        design_pressure_bar: float,
        outside_diameter_mm: float,
        allowable_stress_mpa: float,
        weld_efficiency_e: float = 1.0,
        temp_factor_y: float = 0.4,
        corrosion_allowance_mm: float = 1.5,
        measured_thickness_mm: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Calculates minimum required wall thickness according to ASME B31.3 (Process Piping, Sec 304.1.2).
        Formula:
          t_pressure = (P * D) / (2 * (S * E + P * Y))
          t_min = t_pressure + corrosion_allowance
        Where:
          P = Design internal gauge pressure (converted from bar to MPa: 1 bar = 0.1 MPa)
          D = Outside diameter (mm)
          S = Allowable stress of material (MPa)
          E = Longitudinal weld joint quality factor (1.0 for seamless pipe)
          Y = Material temperature coefficient (0.4 for ferritic steels < 482 deg C)
          c = Mechanical allowances + corrosion allowance (mm)
        """
        if design_pressure_bar <= 0 or outside_diameter_mm <= 0 or allowable_stress_mpa <= 0:
            raise ValueError("Pressure, diameter, and allowable stress must be strictly positive.")

        p_mpa = design_pressure_bar * 0.1
        denominator = 2.0 * ((allowable_stress_mpa * weld_efficiency_e) + (p_mpa * temp_factor_y))
        
        if denominator <= 0:
            raise ValueError("Invalid stress parameters resulting in zero or negative denominator.")

        t_pressure = (p_mpa * outside_diameter_mm) / denominator
        t_min = t_pressure + corrosion_allowance_mm

        t_pressure_rounded = round(t_pressure, 4)
        t_min_rounded = round(t_min, 4)

        audit_steps = [
            f"1. ASME B31.3 Sec 304.1.2 Pressure Piping Minimum Wall Thickness Evaluation",
            f"2. Input Parameters: P = {design_pressure_bar} bar ({p_mpa:.3f} MPa), Outside Diameter D = {outside_diameter_mm:.2f} mm",
            f"3. Material Allowable Stress S = {allowable_stress_mpa:.2f} MPa, Joint Quality Factor E = {weld_efficiency_e}, Temp Factor Y = {temp_factor_y}",
            f"4. Pressure Design Thickness t = (P * D) / [2 * (S*E + P*Y)] = ({p_mpa:.3f} * {outside_diameter_mm:.2f}) / [2 * ({allowable_stress_mpa:.2f}*{weld_efficiency_e} + {p_mpa:.3f}*{temp_factor_y})] = {t_pressure_rounded:.4f} mm",
            f"5. Corrosion Allowance c = {corrosion_allowance_mm:.2f} mm",
            f"6. Total Minimum Required Thickness t_min = {t_pressure_rounded:.4f} + {corrosion_allowance_mm:.2f} = {t_min_rounded:.4f} mm",
        ]

        result: Dict[str, Any] = {
            "standard": "ASME B31.3",
            "operation": "asme_b31_3_min_thickness",
            "design_pressure_bar": design_pressure_bar,
            "design_pressure_mpa": round(p_mpa, 4),
            "outside_diameter_mm": outside_diameter_mm,
            "allowable_stress_mpa": allowable_stress_mpa,
            "weld_efficiency_e": weld_efficiency_e,
            "temp_factor_y": temp_factor_y,
            "corrosion_allowance_mm": corrosion_allowance_mm,
            "pressure_design_thickness_mm": t_pressure_rounded,
            "min_required_thickness_mm": t_min_rounded,
            "audit_trail": audit_steps,
        }

        if measured_thickness_mm is not None:
            is_compliant = measured_thickness_mm >= t_min_rounded
            margin_mm = round(measured_thickness_mm - t_min_rounded, 4)
            result["measured_thickness_mm"] = measured_thickness_mm
            result["is_compliant"] = is_compliant
            result["margin_mm"] = margin_mm
            result["status"] = "COMPLIANT" if is_compliant else "NON_COMPLIANT_RETIREMENT_REQUIRED"
            audit_steps.append(
                f"7. Field Measured Thickness = {measured_thickness_mm:.2f} mm vs Required {t_min_rounded:.2f} mm -> Margin: {margin_mm:+.2f} mm ({result['status']})"
            )

        return result

    @staticmethod
    def calculate_corrosion_rate_and_rul(
        previous_thickness_mm: float,
        current_thickness_mm: float,
        time_interval_years: float,
        required_thickness_mm: float,
    ) -> Dict[str, Any]:
        """
        Calculates Short/Long Term Corrosion Rate (CR) and Remaining Useful Life (RUL)
        according to API 570 (Piping Inspection Code, Section 7.1).
        Formula:
          CR = (previous_thickness - current_thickness) / time_interval (mm/year)
          RUL = (current_thickness - required_thickness) / CR (years)
          Next Inspection Interval = min(RUL / 2, 5.0) years (API 570 half-life rule)
        """
        if time_interval_years <= 0:
            raise ValueError("Time interval between inspections must be strictly positive (> 0 years).")
        if previous_thickness_mm <= 0 or current_thickness_mm <= 0 or required_thickness_mm <= 0:
            raise ValueError("Thickness measurements must be positive numbers.")

        loss_mm = round(previous_thickness_mm - current_thickness_mm, 4)
        if loss_mm <= 0:
            cr = 0.0
            rul_years = 999.0
            next_insp_years = 5.0
            status = "NO_ACTIVE_CORROSION"
        else:
            cr = round(loss_mm / time_interval_years, 4)
            residual_safe_margin = current_thickness_mm - required_thickness_mm
            if residual_safe_margin <= 0:
                rul_years = 0.0
                next_insp_years = 0.0
                status = "CRITICAL_RETIREMENT_BREACH"
            else:
                rul_years = round(residual_safe_margin / cr, 2)
                # API 570 half-life rule: next inspection at half life or max 5 years
                next_insp_years = round(min(rul_years / 2.0, 5.0), 2)
                if rul_years < 2.0:
                    status = "HIGH_RISK_EXPEDITED_MONITORING"
                elif rul_years < 5.0:
                    status = "ELEVATED_RISK_MONITORING"
                else:
                    status = "ACCEPTABLE_SERVICE_LIFE"

        audit_steps = [
            f"1. API 570 Sec 7.1 Corrosion Rate & Remaining Useful Life (RUL) Assessment",
            f"2. Previous Thickness = {previous_thickness_mm:.2f} mm, Current Thickness = {current_thickness_mm:.2f} mm over {time_interval_years:.2f} years",
            f"3. Metal Loss = {loss_mm:.4f} mm -> Corrosion Rate (CR) = {cr:.4f} mm/year",
            f"4. Minimum Allowable Thickness = {required_thickness_mm:.2f} mm (Safe Margin: {current_thickness_mm - required_thickness_mm:+.2f} mm)",
            f"5. Remaining Useful Life (RUL) = (t_actual - t_required) / CR = {rul_years} years ({status})",
            f"6. API 570 Mandatory Next Inspection Interval = min(RUL / 2, 5.0 yrs) = {next_insp_years} years",
        ]

        return {
            "standard": "API 570",
            "operation": "corrosion_rate_and_rul",
            "previous_thickness_mm": previous_thickness_mm,
            "current_thickness_mm": current_thickness_mm,
            "time_interval_years": time_interval_years,
            "required_thickness_mm": required_thickness_mm,
            "metal_loss_mm": loss_mm,
            "corrosion_rate_mm_per_year": cr,
            "remaining_useful_life_years": rul_years,
            "api_570_next_inspection_interval_years": next_insp_years,
            "status": status,
            "audit_trail": audit_steps,
        }

    @staticmethod
    def compute_expression(expr_str: str) -> Dict[str, Any]:
        """
        Safely evaluates mathematical and arithmetic expressions without arbitrary code execution.
        Hardened against AST stack overflows and restricted to a strict whitelist of operators and math functions.
        Supported functions: sqrt, abs, round, min, max, log10, sin, cos, floor, ceil.
        """
        import ast
        import operator
        import math

        allowed_operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
            ast.Pow: operator.pow,
        }

        allowed_functions = {
            "sqrt": math.sqrt,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "log10": math.log10,
            "sin": math.sin,
            "cos": math.cos,
            "floor": math.floor,
            "ceil": math.ceil,
        }

        def eval_node(node, depth: int = 0):
            if depth > 20:
                raise ValueError("Expression exceeded maximum allowable AST recursion depth (20).")

            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return node.value
            elif isinstance(node, ast.BinOp):
                left = eval_node(node.left, depth + 1)
                right = eval_node(node.right, depth + 1)
                op_type = type(node.op)
                if op_type in allowed_operators:
                    return allowed_operators[op_type](left, right)
                raise ValueError(f"Unsupported operator: {op_type}")
            elif isinstance(node, ast.UnaryOp):
                operand = eval_node(node.operand, depth + 1)
                op_type = type(node.op)
                if op_type in allowed_operators:
                    return allowed_operators[op_type](operand)
                raise ValueError(f"Unsupported unary operator: {op_type}")
            elif isinstance(node, ast.Call):
                func_name = None
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                if not func_name or func_name not in allowed_functions:
                    raise ValueError(f"Unsupported or unauthorized function call: {func_name}")
                args = [eval_node(arg, depth + 1) for arg in node.args]
                return allowed_functions[func_name](*args)
            raise ValueError(f"Unsupported AST node: {type(node)}")

        tree = ast.parse(expr_str, mode="eval")
        val = float(eval_node(tree.body))

        return {
            "operation": "expression_evaluation",
            "expression": expr_str,
            "result": round(val, 6),
            "audit_trail": [f"Evaluated arithmetic expression: {expr_str} = {val}"],
        }
