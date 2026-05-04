import os
import random

import numpy as np
import pytest

from aaf.train.seed import seed_everything


def test_seed_everything_reproducibly_seeds_python_and_numpy() -> None:
    first_state = seed_everything(123)
    first_random = random.random()
    first_numpy = np.random.random(3)

    second_state = seed_everything(123)
    second_random = random.random()
    second_numpy = np.random.random(3)

    assert first_state == second_state
    assert first_random == second_random
    assert np.array_equal(first_numpy, second_numpy)
    assert os.environ["PYTHONHASHSEED"] == "123"


def test_seed_everything_rejects_negative_seed() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        seed_everything(-1)
