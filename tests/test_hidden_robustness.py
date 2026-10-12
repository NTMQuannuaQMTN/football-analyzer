import numpy as np

from cv.train_hidden_robustness import noisy_measurements


def test_noise_never_reveals_hidden_coordinates_and_preserves_inputs():
    positions=np.full((130,3,2),999.)
    visible=np.zeros((130,3),bool);visible[:,0]=True
    positions[:,0]=[20,30]
    before=positions.copy()
    result=noisy_measurements(positions,visible,11026)
    assert not np.isfinite(result[~visible]).any()
    assert np.array_equal(positions,before)
    # Noise is reproducible offline augmentation, not a live inference input.
    assert np.allclose(result,noisy_measurements(positions,visible,11026),equal_nan=True)
