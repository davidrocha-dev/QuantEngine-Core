from app.genome.gene_pool import GenePoolConfig
from app.genome.genes import ALL_GENES


def test_defaults_to_all_enabled():
    pool = GenePoolConfig()
    assert len(pool.enabled_names("signal")) == 16
    assert len(pool.enabled_names("management")) == 3


def test_set_enabled_disables_a_gene():
    pool = GenePoolConfig()
    pool.set_enabled("signal", "MACD", False)
    assert "MACD" not in pool.enabled_names("signal")
    assert not pool.is_enabled("signal", "MACD")


def test_set_enabled_unknown_gene_raises():
    pool = GenePoolConfig()
    try:
        pool.set_enabled("signal", "NOT A GENE", False)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_all_management_disabled_detection():
    pool = GenePoolConfig()
    assert not pool.all_management_disabled()
    for name in ["EOD CLOSE", "FRIDAY PROFIT", "PARTIAL TP"]:
        pool.set_enabled("management", name, False)
    assert pool.all_management_disabled()


def test_layer_counts_reflect_toggles():
    pool = GenePoolConfig()
    pool.set_enabled("filter", "RSI", False)
    counts = pool.layer_counts()
    assert counts["filter"]["total"] == 12
    assert counts["filter"]["enabled"] == 11


def test_json_round_trip_preserves_state():
    pool = GenePoolConfig()
    pool.set_enabled("signal", "MACD", False)
    pool.set_enabled("bias", "RSI", False)

    restored = GenePoolConfig.from_json(pool.to_json())

    assert restored.enabled == pool.enabled
    assert len(restored.enabled) == len(ALL_GENES)


def test_from_json_forward_compatible_with_new_genes():
    # Simulates loading a saved config from before a hypothetical new gene
    # was added — the new gene should default to enabled, not KeyError.
    pool = GenePoolConfig()
    saved = pool.to_json()
    restored = GenePoolConfig.from_json(saved)
    assert len(restored.enabled) == len(ALL_GENES)
