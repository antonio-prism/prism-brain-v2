"""
PRISM Engine — Probability History Archive.

Saves compute-all results to the database for historical tracking.
Called from api_routes.py after compute_all() returns.

Usage:
    from prism_engine.history import archive_compute_run, get_run_list

    # After compute_all():
    archive_compute_run(results, calculation_id, start_time, end_time)

    # Query history:
    runs = get_run_list(limit=50)
    detail = get_run_detail(calculation_id)
    history = get_event_history("OPS-AIR-004")
    comparison = compare_runs(id_a, id_b)
"""

import uuid
import logging
from datetime import datetime

from database.connection import get_session_context
from database.models import ProbabilitySnapshot, CalculationLog

logger = logging.getLogger(__name__)

CONFIDENCE_MAP = {"High": 0.9, "Medium": 0.6, "Low": 0.3}


def generate_calculation_id() -> str:
    """Generate a unique calculation ID."""
    return str(uuid.uuid4())[:8] + "-" + datetime.utcnow().strftime("%Y%m%d-%H%M%S")


def archive_compute_run(
    results: dict[str, dict],
    calculation_id: str,
    start_time: datetime,
    end_time: datetime,
    trigger: str = "manual",
) -> dict:
    """Archive a complete compute-all run to the database.

    Args:
        results: Dict keyed by event_id, each value is the full engine output.
        calculation_id: Unique run ID.
        start_time: When the computation started.
        end_time: When the computation finished.
        trigger: "manual" (user clicked button) or "auto" (scheduled).

    Returns:
        Summary dict with counts.
    """
    events_succeeded = 0
    events_failed = 0
    snapshot_rows = []

    for event_id, result in results.items():
        try:
            layer1 = result.get("layer1", {})
            derivation = layer1.get("derivation", {})

            p_global = layer1.get("p_global")
            if p_global is None:
                events_failed += 1
                continue

            source_id = derivation.get("source_id", "")
            is_dynamic = source_id == "dynamic"

            confidence_str = derivation.get("confidence", "Medium")
            confidence_score = CONFIDENCE_MAP.get(confidence_str, 0.5)

            snapshot = ProbabilitySnapshot(
                event_id=event_id,
                probability_pct=round(p_global * 100, 4),
                confidence_score=confidence_score,
                snapshot_date=end_time,
                calculation_id=calculation_id,
                prior=layer1.get("prior"),
                method=layer1.get("method", "FALLBACK"),
                data_source=(derivation.get("data_source") or "")[:200],
                is_dynamic=is_dynamic,
                modifier_count=len(layer1.get("modifiers", [])),
                domain=result.get("domain", ""),
                family=result.get("family", ""),
                event_name=result.get("event_name", event_id),
            )
            snapshot_rows.append(snapshot)
            events_succeeded += 1

        except Exception as e:
            logger.warning(f"Failed to create snapshot for {event_id}: {e}")
            events_failed += 1

    duration_seconds = (end_time - start_time).total_seconds()

    try:
        with get_session_context() as session:
            # Run-level metadata
            calc_log = CalculationLog(
                calculation_id=calculation_id,
                start_time=start_time,
                end_time=end_time,
                events_processed=len(results),
                events_succeeded=events_succeeded,
                events_failed=events_failed,
                duration_seconds=round(duration_seconds, 2),
                status="COMPLETED" if events_failed == 0 else "COMPLETED_WITH_ERRORS",
                trigger=trigger,
                method="engine_v2",
            )
            session.add(calc_log)
            session.add_all(snapshot_rows)

        logger.info(
            f"Archived run {calculation_id}: {events_succeeded} snapshots, "
            f"{events_failed} failures, {duration_seconds:.1f}s"
        )

    except Exception as e:
        logger.error(f"Failed to archive compute run {calculation_id}: {e}")
        return {
            "status": "error",
            "error": str(e),
            "calculation_id": calculation_id,
        }

    return {
        "status": "archived",
        "calculation_id": calculation_id,
        "events_archived": events_succeeded,
        "events_failed": events_failed,
        "duration_seconds": round(duration_seconds, 2),
    }


def get_run_list(limit: int = 50, offset: int = 0) -> list[dict]:
    """Get a list of historical compute runs, most recent first."""
    try:
        with get_session_context() as session:
            runs = (
                session.query(CalculationLog)
                .filter(CalculationLog.method == "engine_v2")
                .order_by(CalculationLog.start_time.desc())
                .offset(offset)
                .limit(limit)
                .all()
            )
            return [
                {
                    "calculation_id": r.calculation_id,
                    "start_time": r.start_time.isoformat() if r.start_time else None,
                    "end_time": r.end_time.isoformat() if r.end_time else None,
                    "events_processed": r.events_processed,
                    "events_succeeded": r.events_succeeded,
                    "events_failed": r.events_failed,
                    "duration_seconds": r.duration_seconds,
                    "status": r.status,
                    "trigger": r.trigger,
                }
                for r in runs
            ]
    except Exception as e:
        logger.error(f"Failed to get run list: {e}")
        return []


def get_run_detail(calculation_id: str) -> list[dict]:
    """Get all event snapshots for a specific compute run."""
    try:
        with get_session_context() as session:
            snapshots = (
                session.query(ProbabilitySnapshot)
                .filter(ProbabilitySnapshot.calculation_id == calculation_id)
                .order_by(ProbabilitySnapshot.domain, ProbabilitySnapshot.event_id)
                .all()
            )
            return [
                {
                    "event_id": s.event_id,
                    "event_name": s.event_name,
                    "domain": s.domain,
                    "family": s.family,
                    "probability_pct": s.probability_pct,
                    "prior": s.prior,
                    "method": s.method,
                    "confidence_score": s.confidence_score,
                    "data_source": s.data_source,
                    "is_dynamic": s.is_dynamic,
                    "modifier_count": s.modifier_count,
                    "snapshot_date": s.snapshot_date.isoformat() if s.snapshot_date else None,
                }
                for s in snapshots
            ]
    except Exception as e:
        logger.error(f"Failed to get run detail for {calculation_id}: {e}")
        return []


def get_event_history(event_id: str, limit: int = 100) -> list[dict]:
    """Get the probability history for a single event across all runs."""
    try:
        with get_session_context() as session:
            snapshots = (
                session.query(ProbabilitySnapshot)
                .filter(ProbabilitySnapshot.event_id == event_id)
                .order_by(ProbabilitySnapshot.snapshot_date.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "calculation_id": s.calculation_id,
                    "probability_pct": s.probability_pct,
                    "prior": s.prior,
                    "method": s.method,
                    "data_source": s.data_source,
                    "is_dynamic": s.is_dynamic,
                    "modifier_count": s.modifier_count,
                    "confidence_score": s.confidence_score,
                    "snapshot_date": s.snapshot_date.isoformat() if s.snapshot_date else None,
                }
                for s in snapshots
            ]
    except Exception as e:
        logger.error(f"Failed to get event history for {event_id}: {e}")
        return []


def compare_runs(calc_id_a: str, calc_id_b: str) -> list[dict]:
    """Compare two runs side-by-side, showing probability changes per event."""
    run_a = {s["event_id"]: s for s in get_run_detail(calc_id_a)}
    run_b = {s["event_id"]: s for s in get_run_detail(calc_id_b)}

    all_events = sorted(set(list(run_a.keys()) + list(run_b.keys())))
    comparison = []

    for eid in all_events:
        a = run_a.get(eid, {})
        b = run_b.get(eid, {})
        p_a = a.get("probability_pct")
        p_b = b.get("probability_pct")

        delta = None
        if p_a is not None and p_b is not None:
            delta = round(p_b - p_a, 4)

        comparison.append({
            "event_id": eid,
            "event_name": a.get("event_name") or b.get("event_name", eid),
            "domain": a.get("domain") or b.get("domain", ""),
            "prob_a": p_a,
            "prob_b": p_b,
            "delta": delta,
            "method_a": a.get("method"),
            "method_b": b.get("method"),
        })

    return comparison
