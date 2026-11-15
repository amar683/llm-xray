import pytest

from xray.analyzers.timing import KINDS, context_ids, layer_stats, run_timing


@pytest.fixture(scope="module")
def result(tiny):
    return run_timing(tiny, contexts=[8, 16, 500], prefill_repeats=3, decode_repeats=4)


def test_contexts_beyond_model_limit_are_skipped(result):
    # The tiny model's limit is 64 positions.
    assert [c["context"] for c in result["contexts"]] == [8, 16]
    assert result["skipped_contexts"] == [500]


def test_decode_repeats_shrink_to_fit_a_short_context_window(tiny):
    # 40 prompt tokens in a 64-position model leave room for 5 repeats of 4 passes, not 24.
    r = run_timing(tiny, contexts=[40], prefill_repeats=2, decode_repeats=24)
    assert r["skipped_contexts"] == [] and r["contexts"][0]["decode_repeats"] == 5
    assert r["contexts"][0]["decode"]["layers"][0]["est_ms"] > 0


def test_context_ids_have_exact_length(tiny):
    assert context_ids(tiny, 37).shape == (1, 37)


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_estimates_add_up_to_wall_time(result, phase):
    for ctx in result["contexts"]:
        p = ctx[phase]
        total = sum(row["est_ms"] for row in p["layers"])
        total += sum(v["est_ms"] for v in p["other"].values())
        assert total == pytest.approx(p["wall_ms"], rel=1e-6)
        assert sum(p["shares"].values()) == pytest.approx(1.0, rel=1e-6)
        shares = [row["share"] for row in p["layers"]] + [v["share"] for v in p["other"].values()]
        assert sum(shares) == pytest.approx(1.0, rel=1e-6)


def test_layer_rows_are_consistent(result):
    for ctx in result["contexts"]:
        for phase in ("prefill", "decode"):
            rows = ctx[phase]["layers"]
            assert [r["layer"] for r in rows] == [0, 1, 2]
            for r in rows:
                assert r["est_ms"] > 0 and r["attn_ms"] > 0 and r["mlp_ms"] > 0
                parts = r["attn_ms"] + r["mlp_ms"] + r["other_ms"]
                assert parts == pytest.approx(r["est_ms"], rel=1e-6)
                assert r["isolated_p95_ms"] >= r["isolated_ms"] > 0


def test_speed_and_overhead_fields(result):
    ctx = result["contexts"][1]
    assert ctx["prefill"]["tokens_per_s"] == pytest.approx(16 / (ctx["prefill"]["wall_ms"] / 1e3))
    assert ctx["decode"]["tokens_per_s"] == pytest.approx(1 / (ctx["decode"]["wall_ms"] / 1e3))
    assert set(ctx["decode"]["probe_overhead_pct"]) == set(KINDS[1:])
    assert 0 < ctx["decode"]["dispatch_fraction"] <= 1.0


def test_bandwidth_and_devices(result, tiny):
    bw = result["bandwidth"]
    assert bw["peak_gb_per_s"] > 0 and bw["decode_floor_ms"] > 0
    blocks_and_head = sum(
        p.numel() * p.element_size()
        for m in [*(layer.block for layer in tiny.refs.layers), tiny.refs.lm_head]
        for p in m.parameters()
    )
    assert bw["weights_read_per_token_gb"] == pytest.approx(blocks_and_head / 1e9)
    assert result["layer_devices"] == {"main": "cpu", "offloaded_layers": []}


def test_layer_stats_rows(result):
    rows = layer_stats(result)
    # 2 contexts x 2 phases x (3 layers x 3 components + 3 other modules)
    assert len(rows) == 2 * 2 * (3 * 3 + 3)
    assert {r["metric"] for r in rows} == {"est_ms"}
    assert {r["component"] for r in rows} == {
        "block", "attn", "mlp", "embed", "final_norm", "lm_head"
    }  # fmt: skip
