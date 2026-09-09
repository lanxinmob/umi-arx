"""Pure validation and safety projection for external agent actions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import numbers
from typing import Any, Optional

import numpy as np
import scipy.spatial.transform as st


PLAN_RESULT_KEYS = frozenset(
    {
        "type",
        "status",
        "delta_position_tool",
        "delta_rotation_tool",
        "gripper_delta",
        "confidence",
        "phase",
        "rationale",
    }
)
PLAN_STATUSES = frozenset({"move", "hold", "done", "human"})


class PlanValidationError(ValueError):
    """Raised when a planner response violates the wire schema."""


@dataclass(frozen=True)
class SafetyLimits:
    max_position_step: float = 0.005
    max_rotation_step: float = 0.035
    max_gripper_step: float = 0.002
    min_gripper_width: float = 0.0
    max_gripper_width: float = 0.085
    max_episode_radius: float = 0.12
    min_z_below_start: float = 0.08
    min_confidence: float = 0.6

    def __post_init__(self) -> None:
        values = asdict(self)
        for name, value in values.items():
            if not isinstance(value, numbers.Real) or isinstance(value, bool):
                raise TypeError(f"{name} must be a real number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        for name in (
            "max_position_step",
            "max_rotation_step",
            "max_gripper_step",
            "max_episode_radius",
            "min_z_below_start",
        ):
            if float(values[name]) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.min_gripper_width >= self.max_gripper_width:
            raise ValueError("min_gripper_width must be less than max_gripper_width")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")

    def to_protocol_dict(self) -> dict[str, float]:
        return {name: float(value) for name, value in asdict(self).items()}


@dataclass(frozen=True)
class ValidatedPlan:
    status: str
    delta_position_tool: np.ndarray
    delta_rotation_tool: np.ndarray
    gripper_delta: float
    confidence: float
    phase: str
    rationale: str


@dataclass(frozen=True)
class SafetyDecision:
    target: np.ndarray
    accepted: bool
    rejected: bool
    is_move: bool
    planner_status: str
    reason: str
    confidence: Optional[float] = None
    phase: str = ""
    rationale: str = ""

    def to_record(self) -> dict[str, Any]:
        return {
            "target": self.target.tolist(),
            "accepted": self.accepted,
            "rejected": self.rejected,
            "is_move": self.is_move,
            "planner_status": self.planner_status,
            "reason": self.reason,
            "confidence": self.confidence,
            "phase": self.phase,
            "rationale": self.rationale,
        }


def _strict_scalar(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise PlanValidationError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise PlanValidationError(f"{name} must be finite")
    return result


def _strict_normalized_vector(value: Any, name: str) -> np.ndarray:
    if not isinstance(value, (list, tuple)) or isinstance(value, np.ndarray):
        raise PlanValidationError(f"{name} must be a JSON-style list of length 3")
    if len(value) != 3:
        raise PlanValidationError(f"{name} must have length 3")
    result = np.asarray(
        [_strict_scalar(component, f"{name}[{index}]") for index, component in enumerate(value)],
        dtype=np.float64,
    )
    if np.any(result < -1.0) or np.any(result > 1.0):
        raise PlanValidationError(f"{name} components must be in [-1, 1]")
    return result


def validate_plan_response(response: Any) -> ValidatedPlan:
    """Validate the exact PLAN_RESULT schema without coercing free-form output."""
    if not isinstance(response, dict):
        raise PlanValidationError("planner response must be a dict")
    keys = frozenset(response.keys())
    if keys != PLAN_RESULT_KEYS:
        missing = sorted(PLAN_RESULT_KEYS - keys)
        extra = sorted(keys - PLAN_RESULT_KEYS)
        raise PlanValidationError(
            f"planner response keys mismatch; missing={missing}, extra={extra}"
        )
    if response["type"] != "PLAN_RESULT":
        raise PlanValidationError("type must be 'PLAN_RESULT'")
    status = response["status"]
    if not isinstance(status, str) or status not in PLAN_STATUSES:
        raise PlanValidationError(
            "status must be one of 'move', 'hold', 'done', or 'human'"
        )
    phase = response["phase"]
    rationale = response["rationale"]
    if not isinstance(phase, str) or len(phase) > 256:
        raise PlanValidationError("phase must be a string no longer than 256 chars")
    if not isinstance(rationale, str) or len(rationale) > 4096:
        raise PlanValidationError(
            "rationale must be a string no longer than 4096 chars"
        )
    confidence = _strict_scalar(response["confidence"], "confidence")
    if not 0.0 <= confidence <= 1.0:
        raise PlanValidationError("confidence must be in [0, 1]")
    gripper_delta = _strict_scalar(response["gripper_delta"], "gripper_delta")
    if not -1.0 <= gripper_delta <= 1.0:
        raise PlanValidationError("gripper_delta must be in [-1, 1]")
    return ValidatedPlan(
        status=status,
        delta_position_tool=_strict_normalized_vector(
            response["delta_position_tool"], "delta_position_tool"
        ),
        delta_rotation_tool=_strict_normalized_vector(
            response["delta_rotation_tool"], "delta_rotation_tool"
        ),
        gripper_delta=gripper_delta,
        confidence=confidence,
        phase=phase,
        rationale=rationale,
    )


def _finite_vector(value: Any, length: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},)")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _project_unit_ball(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm > 1.0:
        return vector / norm
    return vector.copy()


def rotation_geodesic(rotvec_a: Any, rotvec_b: Any) -> float:
    a = _finite_vector(rotvec_a, 3, "rotvec_a")
    b = _finite_vector(rotvec_b, 3, "rotvec_b")
    rotation_a = st.Rotation.from_rotvec(a)
    rotation_b = st.Rotation.from_rotvec(b)
    return float((rotation_a * rotation_b.inv()).magnitude())


def _hold_decision(
    current_target: np.ndarray,
    reason: str,
    *,
    rejected: bool,
    plan: Optional[ValidatedPlan] = None,
    planner_status: str = "invalid",
) -> SafetyDecision:
    if plan is not None:
        planner_status = plan.status
    return SafetyDecision(
        target=current_target.copy(),
        accepted=not rejected,
        rejected=rejected,
        is_move=False,
        planner_status=planner_status,
        reason=reason,
        confidence=None if plan is None else plan.confidence,
        phase="" if plan is None else plan.phase,
        rationale="" if plan is None else plan.rationale,
    )


def evaluate_plan_response(
    response: Any,
    *,
    current_pose: Any,
    current_gripper: Any,
    episode_start_pose: Any,
    last_submitted_target: Any = None,
    limits: SafetyLimits = SafetyLimits(),
) -> SafetyDecision:
    """Return one absolute safe target; invalid input always becomes a hold."""
    try:
        pose = _finite_vector(current_pose, 6, "current_pose")
        gripper = _strict_scalar(current_gripper, "current_gripper")
        start_pose = _finite_vector(episode_start_pose, 6, "episode_start_pose")
        current_target = np.concatenate([pose, [gripper]])
    except (TypeError, ValueError, PlanValidationError) as exc:
        raise ValueError(f"invalid measured state: {exc}") from exc

    try:
        plan = validate_plan_response(response)
    except (TypeError, ValueError, PlanValidationError) as exc:
        return _hold_decision(
            current_target,
            f"invalid_response: {exc}",
            rejected=True,
        )

    if plan.confidence < limits.min_confidence:
        return _hold_decision(
            current_target,
            "confidence_below_threshold",
            rejected=True,
            plan=plan,
        )
    if plan.status != "move":
        return _hold_decision(
            current_target,
            f"planner_{plan.status}",
            rejected=False,
            plan=plan,
        )

    position_delta_tool = (
        _project_unit_ball(plan.delta_position_tool) * limits.max_position_step
    )
    rotation_delta_tool = (
        _project_unit_ball(plan.delta_rotation_tool) * limits.max_rotation_step
    )
    current_rotation = st.Rotation.from_rotvec(pose[3:])
    target_position = pose[:3] + current_rotation.apply(position_delta_tool)
    target_rotation = (
        current_rotation * st.Rotation.from_rotvec(rotation_delta_tool)
    ).as_rotvec()
    target_gripper = float(
        np.clip(
            gripper + plan.gripper_delta * limits.max_gripper_step,
            limits.min_gripper_width,
            limits.max_gripper_width,
        )
    )
    target = np.concatenate([target_position, target_rotation, [target_gripper]])

    tolerance = 1e-10
    references = [("current", current_target)]
    if last_submitted_target is not None:
        try:
            submitted = _finite_vector(
                last_submitted_target, 7, "last_submitted_target"
            )
        except ValueError as exc:
            return _hold_decision(
                current_target,
                f"invalid_last_submitted_target: {exc}",
                rejected=True,
                plan=plan,
            )
        references.append(("last_submitted", submitted))

    for reference_name, reference in references:
        if (
            np.linalg.norm(target[:3] - reference[:3])
            > limits.max_position_step + tolerance
        ):
            return _hold_decision(
                current_target,
                f"position_step_exceeds_limit_from_{reference_name}",
                rejected=True,
                plan=plan,
            )
        if (
            rotation_geodesic(target[3:6], reference[3:6])
            > limits.max_rotation_step + tolerance
        ):
            return _hold_decision(
                current_target,
                f"rotation_step_exceeds_limit_from_{reference_name}",
                rejected=True,
                plan=plan,
            )
        if (
            abs(float(target[6] - reference[6]))
            > limits.max_gripper_step + tolerance
        ):
            return _hold_decision(
                current_target,
                f"gripper_step_exceeds_limit_from_{reference_name}",
                rejected=True,
                plan=plan,
            )

    if (
        np.linalg.norm(target[:3] - start_pose[:3])
        > limits.max_episode_radius + tolerance
    ):
        return _hold_decision(
            current_target,
            "episode_radius_exceeded",
            rejected=True,
            plan=plan,
        )
    if target[2] < start_pose[2] - limits.min_z_below_start - tolerance:
        return _hold_decision(
            current_target,
            "minimum_z_exceeded",
            rejected=True,
            plan=plan,
        )

    return SafetyDecision(
        target=target,
        accepted=True,
        rejected=False,
        is_move=True,
        planner_status=plan.status,
        reason="accepted_move",
        confidence=plan.confidence,
        phase=plan.phase,
        rationale=plan.rationale,
    )
