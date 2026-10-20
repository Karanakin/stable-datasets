import numpy as np
from PIL import Image

from stable_datasets.images import ChestXRayPneumonia

def test_chest_xray_pneumonia_dataset():
    # Load the training split.
    chest_xray = ChestXRayPneumonia(split="train")

    # Test 1: Check that the training split has the expected number of samples.
    expected_num_train_samples = 5232
    assert len(chest_xray) == expected_num_train_samples, (
        f"Expected {expected_num_train_samples} training samples, "
        f"got {len(chest_xray)}."
    )

    # Test 2: Check that each sample has the keys "image" and "label".
    sample = chest_xray[0]
    expected_keys = {"image", "label"}
    assert set(sample.keys()) == expected_keys, f"Expected keys {expected_keys}, got {set(sample.keys())}"

    # Test 3: Validate image type and format.
    image = sample["image"]
    assert isinstance(image, Image.Image), f"Image should be a PIL image, got {type(image)}."

    image_np = np.array(image)
    assert image_np.ndim == 3 and image_np.shape[2] == 3, (
        f"Image should have three RGB channels, got shape {image_np.shape}."
    )
    assert image_np.dtype == np.uint8, (
        f"Image should have dtype uint8, got {image_np.dtype}."
    )

    # Test 4: Validate label type and range.
    label = sample["label"]
    assert isinstance(label, int), f"Label should be an integer, got {type(label)}."
    assert label in (0, 1), f"Label should be 0 or 1, got {label}."

    # Test 5: Check class metadata.
    expected_classes = ["NORMAL", "PNEUMONIA"]
    class_names = chest_xray.info.features["label"].names
    assert class_names == expected_classes, (
        f"Expected classes {expected_classes}, got {class_names}."
    )

    # Test 6: Check the test split.
    chest_xray_test = ChestXRayPneumonia(split="test")
    expected_num_test_samples = 624
    assert len(chest_xray_test) == expected_num_test_samples, (
        f"Expected {expected_num_test_samples} test samples, "
        f"got {len(chest_xray_test)}."
    )

    print("All ChestXRayPneumonia dataset tests passed successfully!")




