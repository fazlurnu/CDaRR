"""L0 equivalence: cns.reception vs. the legacy sim_models.reception_model.ReceptionModel."""
import numpy as np
import pytest

from cns.reception import sample_received
from sim_models.reception_model import ReceptionModel


@pytest.mark.fast
@pytest.mark.parametrize("p", [0.0, 0.3, 0.7, 0.99, 1.0])
@pytest.mark.parametrize("n", [1, 5, 100])
def test_sample_received_matches_legacy(p, n):
    old_rng, new_rng = np.random.default_rng(11), np.random.default_rng(11)
    old = ReceptionModel(reception_prob=p, rng=old_rng)
    old_idx = old.sample_indices(n)
    new_idx = sample_received(n, p, new_rng)
    assert np.array_equal(old_idx, new_idx)
    assert old_rng.bit_generator.state == new_rng.bit_generator.state


@pytest.mark.fast
def test_sample_received_matches_legacy_n_zero():
    old = ReceptionModel(reception_prob=0.5, rng=np.random.default_rng(0))
    assert np.array_equal(old.sample_indices(0), sample_received(0, 0.5, np.random.default_rng(0)))
    assert np.array_equal(old.sample_indices(-1), sample_received(-1, 0.5, np.random.default_rng(0)))


@pytest.mark.fast
def test_sample_received_invalid_p_raises():
    with pytest.raises(ValueError):
        ReceptionModel(reception_prob=1.5, rng=np.random.default_rng(0))
    with pytest.raises(ValueError):
        sample_received(10, 1.5, np.random.default_rng(0))
