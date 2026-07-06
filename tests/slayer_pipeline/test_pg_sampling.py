"""DEV-1648: deterministic date-format detection from sampled values.

``detect_date_format`` picks the first ordered candidate strptime format
that round-trips ALL samples (no LLM). A ``.%f`` candidate treats the
fractional-seconds tail as OPTIONAL so a column mixing fractional and
non-fractional rows is still detected (precision preserved in the emitted
``.US`` cast). The live ``sample_pg_values`` sampler itself is exercised
only by the ``@pytest.mark.integration`` suite.
"""

from __future__ import annotations

from bird_interact_agents.slayer_pipeline import pg_sampling
from bird_interact_agents.slayer_pipeline.pg_sampling import (
    _extract_sample_query,
    _sample_query,
    detect_date_format,
    detect_fraction_pg_token,
    make_pg_sampler,
)


def test_detect_iso_date_only() -> None:
    samples = ["2025-02-19", "1999-12-31", "2000-01-01"]
    assert detect_date_format(samples, with_time=False) == "%Y-%m-%d"


def test_detect_iso_timestamp_with_microseconds() -> None:
    samples = ["2025-02-19 08:31:22.330375", "2024-01-01 00:00:00.000001"]
    assert (
        detect_date_format(samples, with_time=True) == "%Y-%m-%d %H:%M:%S.%f"
    )


def test_detect_dotted_timestamp_no_fraction() -> None:
    # match_ts: dotted, NO fractional seconds -> base format WITHOUT ".%f".
    samples = ["2025.02.19 08:31:22", "2025.03.01 12:00:00"]
    assert (
        detect_date_format(samples, with_time=True) == "%Y.%m.%d %H:%M:%S"
    )


def test_detect_mixed_fraction_and_no_fraction() -> None:
    # A single column with BOTH shapes must be detected (optional fraction).
    samples = ["2025-02-19 08:31:22.330375", "2025-02-19 08:31:22"]
    assert (
        detect_date_format(samples, with_time=True) == "%Y-%m-%d %H:%M:%S.%f"
    )


def test_detect_requires_all_samples_to_match() -> None:
    samples = ["2025-02-19", "not-a-date", "2000-01-01"]
    assert detect_date_format(samples, with_time=False) is None


def test_detect_empty_returns_none() -> None:
    assert detect_date_format([], with_time=False) is None
    assert detect_date_format([], with_time=True) is None


def test_detect_date_token_does_not_accept_datetime_only_shape() -> None:
    # A DATE token (with_time=False) must not match values that carry a
    # time component under a datetime candidate.
    samples = ["2025-02-19 08:31:22", "2025-03-01 09:00:00"]
    assert detect_date_format(samples, with_time=False) is None


def test_detect_ordering_prefers_iso_over_ambiguous() -> None:
    # Values that fit ISO must be reported as ISO, not a lower-priority
    # candidate.
    samples = ["2025-02-19", "2025-03-01"]
    assert detect_date_format(samples, with_time=False) == "%Y-%m-%d"


def test_detect_ambiguous_dmy_vs_mdy_resolved_by_order() -> None:
    # '05/06/2025' fits BOTH %d/%m/%Y and %m/%d/%Y; the ordered candidate
    # list must resolve it deterministically (day-first preferred).
    samples = ["05/06/2025", "07/06/2025"]
    assert detect_date_format(samples, with_time=False) == "%d/%m/%Y"


# ---------------------------------------------------------------------------
# Fractional-second width -> Postgres token (MS vs US)
# ---------------------------------------------------------------------------


def test_fraction_token_six_digits_is_us() -> None:
    samples = ["2025-02-19 08:31:22.330375", "2024-01-01 00:00:00.000001"]
    assert detect_fraction_pg_token(samples) == "US"


def test_fraction_token_three_digits_is_ms() -> None:
    samples = ["2025-02-19 08:31:22.330", "2024-01-01 00:00:00.001"]
    assert detect_fraction_pg_token(samples) == "MS"


def test_fraction_token_no_fraction_defaults_us() -> None:
    samples = ["2025-02-19 08:31:22", "2024-01-01 00:00:00"]
    assert detect_fraction_pg_token(samples) == "US"


def test_fraction_token_mixed_widths_defaults_us() -> None:
    samples = ["2025-02-19 08:31:22.330", "2024-01-01 00:00:00.000001"]
    assert detect_fraction_pg_token(samples) == "US"


# ---------------------------------------------------------------------------
# Sampler query builders (pure) + error handling
# ---------------------------------------------------------------------------


def test_sample_query_is_deterministic_and_quoted() -> None:
    q = _sample_query("transplant_matching", "match_ts")
    assert q == (
        'SELECT DISTINCT "match_ts" FROM "transplant_matching" '
        'WHERE "match_ts" IS NOT NULL AND "match_ts" <> \'\' '
        'ORDER BY "match_ts" LIMIT 20'
    )


def test_extract_sample_query_wraps_expression() -> None:
    extract = "jsonb_extract_path_text(\"socioeconomic\", 'event_ts')"
    q = _extract_sample_query("households", extract)
    assert extract in q
    assert q.startswith("SELECT DISTINCT ")
    assert 'FROM "households"' in q
    assert "LIMIT 20" in q
    # Deterministic: ordered by the extracted value (mirrors _sample_query).
    assert f"ORDER BY {extract} LIMIT" in q


def test_make_pg_sampler_swallows_errors_returns_empty(monkeypatch) -> None:
    def boom(_url):
        raise RuntimeError("no db")

    monkeypatch.setattr(pg_sampling, "_make_engine", boom)
    sampler = make_pg_sampler("postgresql://x@localhost:1/none")
    assert sampler("t", "c") == []
