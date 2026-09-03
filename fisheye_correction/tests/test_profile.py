import numpy as np
from fisheye_correction.profiles import PAVANALYTICS_INPUT_SIZE, correct_with_pavanalytics_profile

def test_tuned_profile_output_shape():
    width, height = PAVANALYTICS_INPUT_SIZE
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    corrected = correct_with_pavanalytics_profile(frame)
    assert corrected.shape == (720, 720, 3)
    assert np.isfinite(corrected).all()
