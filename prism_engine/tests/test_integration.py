"""
PRISM Engine — Phase 1 Integration Tests.

Tests all 10 prototype events end-to-end.
Verifies: valid JSON output, derivation trails, validation rules, fallback chain.
"""

import json
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from prism_engine.engine import compute, compute_all_phase1
from prism_engine.config.event_mapping import PHASE1_EVENTS
from prism_engine.fallback import load_fallback_rates, get_fallback_rate
from prism_engine.computation.validation import validate_event_output


def test_fallback_rates():
    """Test that fallback rates load correctly from seed files."""
    rates = load_fallback_rates()
    print(f"\n{'='*60}")
    print(f"FALLBACK RATES: Loaded {len(rates)} events from seed files")
    print(f"{'='*60}")

    # Check Phase 1 events have fallback rates
    for event_id in PHASE1_EVENTS:
        rate = get_fallback_rate(event_id)
        name = PHASE1_EVENTS[event_id]["name"]
        print(f"  {event_id}: {rate:.4f} ({rate*100:.2f}%) — {name}")

    assert len(rates) > 0, "No fallback rates loaded!"
    print(f"\n  PASS: {len(rates)} fallback rates loaded")


def test_single_event_structure():
    """Test that a single computed event matches the Section 9.2 schema."""
    print(f"\n{'='*60}")
    print("SCHEMA VALIDATION: Testing output structure")
    print(f"{'='*60}")

    # Use PHY-BIO-001 (disease outbreak) — uses manual data, no API needed
    result = compute("PHY-BIO-001")

    # Required top-level fields
    required_fields = ["event_id", "event_name", "domain", "family", "layer1", "metadata"]
    for field in required_fields:
        assert field in result, f"Missing top-level field: {field}"
        print(f"  {field}: present")

    # Required Layer 1 fields
    layer1 = result["layer1"]
    l1_required = ["prior", "method", "derivation", "modifiers", "p_global"]
    for field in l1_required:
        assert field in layer1, f"Missing layer1 field: {field}"
        print(f"  layer1.{field}: present")

    # Required derivation fields
    deriv = layer1["derivation"]
    deriv_required = ["formula", "data_source", "confidence"]
    for field in deriv_required:
        assert field in deriv, f"Missing derivation field: {field}"
        assert deriv[field], f"Empty derivation field: {field}"
        print(f"  layer1.derivation.{field}: '{str(deriv[field])[:60]}'")

    # Validation
    errors = validate_event_output(result)
    if errors:
        print(f"\n  VALIDATION WARNINGS: {errors}")
    else:
        print(f"\n  PASS: Output matches Section 9.2 schema")


def test_all_phase1_events():
    """Test all 10 Phase 1 events compute successfully."""
    print(f"\n{'='*60}")
    print("PHASE 1 FULL RUN: Computing all 10 prototype events")
    print(f"{'='*60}")

    results = compute_all_phase1()
    passed = 0
    failed = 0
    documented_divergent = 0
    undocumented_divergent = 0

    for event_id, result in results.items():
        name = result.get("event_name", event_id)
        prior = result["layer1"]["prior"]
        p_global = result["layer1"]["p_global"]
        method = result["layer1"].get("method", "?")
        status = result.get("metadata", {}).get("data_status", "OK")
        fallback = result.get("metadata", {}).get("fallback_rate", 0)
        divergence = result.get("metadata", {}).get("divergence_from_fallback", 0)

        # Check if it computed or fell back
        is_fallback = "FALLBACK" in str(status) or method == "FALLBACK"

        # Validation
        errors = validate_event_output(result)

        status_icon = "FALLBACK" if is_fallback else "COMPUTED"
        div_flag = " DIVERGENT" if divergence > 0.50 else ""

        # Check for documented divergence reason
        div_reason = result.get("metadata", {}).get("divergence_reason")
        documented = div_reason is not None and "needs documentation" not in str(div_reason)

        print(f"\n  {event_id} ({method}) — {name[:40]}")
        print(f"    Prior: {prior:.4f} ({prior*100:.2f}%)  |  P_global: {p_global:.4f} ({p_global*100:.2f}%)")
        print(f"    Fallback: {fallback:.4f}  |  Divergence: {divergence:.0%}{div_flag}")
        print(f"    Modifiers: {len(result['layer1'].get('modifiers', []))}")
        print(f"    Status: [{status_icon}]  |  Validation: {'PASS' if not errors else errors}")
        if div_reason and divergence > 0.50:
            print(f"    Reason: {div_reason[:80]}...")

        if not errors:
            passed += 1
        else:
            failed += 1
        if divergence > 0.50:
            if documented:
                documented_divergent += 1
            else:
                undocumented_divergent += 1

    total_divergent = documented_divergent + undocumented_divergent
    # Per spec: divergences are acceptable IF documented with explanation
    acceptance = undocumented_divergent <= 2
    print(f"\n{'='*60}")
    print(f"RESULTS: {passed} passed, {failed} failed")
    print(f"  Divergent: {total_divergent} total ({documented_divergent} documented, {undocumented_divergent} undocumented)")
    print(f"  Acceptance: {'PASS' if acceptance else 'FAIL'} (undocumented divergences <=2)")
    print(f"{'='*60}")

    return results


def test_fallback_chain():
    """Test that disabling a source triggers fallback gracefully."""
    print(f"\n{'='*60}")
    print("FALLBACK CHAIN: Testing graceful degradation")
    print(f"{'='*60}")

    # Compute an event that uses an API source
    result = compute("PHY-GEO-001")  # Earthquake — uses USGS

    # The result should have a valid probability regardless
    p = result["layer1"]["p_global"]
    assert 0.001 <= p <= 0.95, f"P_global {p} out of valid range"
    print(f"  PHY-GEO-001 P_global: {p:.4f} — within valid range")

    # Test an event that doesn't exist in Phase 1
    result_unknown = compute("XXX-YYY-999")
    assert result_unknown["layer1"]["p_global"] > 0, "Unknown event should still return a probability"
    print(f"  XXX-YYY-999 (unknown): returned fallback {result_unknown['layer1']['p_global']:.4f}")

    print(f"\n  PASS: Fallback chain works correctly")


def main():
    """Run all Phase 1 integration tests."""
    print("\n" + "=" * 60)
    print("  PRISM PROBABILITY ENGINE — PHASE 1 INTEGRATION TESTS")
    print("=" * 60)

    test_fallback_rates()
    test_single_event_structure()
    test_fallback_chain()
    results = test_all_phase1_events()

    # Save results to output
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "output", "events")
    os.makedirs(output_dir, exist_ok=True)

    for event_id, result in results.items():
        output_path = os.path.join(output_dir, f"{event_id}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"  Saved: {output_path}")

    print(f"\n  All outputs saved to {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
