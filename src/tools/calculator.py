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
    def compute_expression(expr_str: str) -> Dict[str, Any]:
        """Safely evaluates basic arithmetic expressions without arbitrary code execution."""
        import ast
        import operator

        allowed_operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.USub: operator.neg,
            ast.Pow: operator.pow,
        }

        def eval_node(node):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return node.value
            elif isinstance(node, ast.BinOp):
                left = eval_node(node.left)
                right = eval_node(node.right)
                op_type = type(node.op)
                if op_type in allowed_operators:
                    return allowed_operators[op_type](left, right)
                raise ValueError(f"Unsupported operator: {op_type}")
            elif isinstance(node, ast.UnaryOp):
                operand = eval_node(node.operand)
                op_type = type(node.op)
                if op_type in allowed_operators:
                    return allowed_operators[op_type](operand)
                raise ValueError(f"Unsupported unary operator: {op_type}")
            raise ValueError(f"Unsupported AST node: {type(node)}")

        tree = ast.parse(expr_str, mode="eval")
        val = float(eval_node(tree.body))
        
        return {
            "operation": "expression_evaluation",
            "expression": expr_str,
            "result": round(val, 6),
            "audit_trail": [f"Evaluated arithmetic expression: {expr_str} = {val}"],
        }
