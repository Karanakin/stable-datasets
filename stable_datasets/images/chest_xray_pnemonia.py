"""Chest X-ray Pneumonia: pediatric chest radiographs labelled NORMAL or PNEUMONIA."""

import io
import zipfile

from PIL import Image as PILImage

from stable_datasets.schema import ClassLabel, DatasetInfo, DatasetSource, DownloadInfo, Features, Version
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.utils import BaseDatasetBuilder


# Class folder names inside the archive. Label i is entry i here (0 = NORMAL, 1 = PNEUMONIA).
_CHEST_XRAY_CLASSES: list[str] = ["NORMAL", "PNEUMONIA"]

_FOLDER_TO_LABEL: dict[str, int] = {name: i for i, name in enumerate(_CHEST_XRAY_CLASSES)}

# Both splits live in the same archive; each split reads only its own folder.
# Mendeley Data record rscbjbr9sj, version 2. Version 2 ships the chest X-ray images as their own
# archive (ChestXRay2017.zip, about 1.24 GB). Version 3 bundles them with the OCT images in a much
# larger ZhangLabData.zip, so it is deliberately not used here.
_ARCHIVE = DownloadInfo(
    url="https://data.mendeley.com/datasets/rscbjbr9sj/2/files/f12eaf6d-6023-432f-acc9-80c9d7393433/ChestXRay2017.zip?dl=1",
)


class ChestXRayPneumonia(BaseDatasetBuilder):
    """Chest X-ray Pneumonia (Kermany et al.): 5,856 pediatric chest X-rays, 2 classes.

    Follows the Mendeley Data release (v2, ChestXRay2017.zip), which has a ``train`` split (5,232 images)
    and a ``test`` split (624 images). The Kaggle copy differs (5,216 train, 16 val, 624 test).
    The classes are imbalanced (about 74% PNEUMONIA). Labels are integers in [0, 2):
    0 = NORMAL, 1 = PNEUMONIA. Image sizes vary and most scans are grayscale, so all images
    are returned as RGB.
    """

    VERSION = Version("1.0.0")

    SOURCE = DatasetSource(
        homepage="https://data.mendeley.com/datasets/rscbjbr9sj/2",
        assets={
            "train": _ARCHIVE,
            "test": _ARCHIVE,
        },
        citation="""@article{kermany2018identifying,
                      title={Identifying medical diagnoses and treatable diseases by image-based deep learning},
                      author={Kermany, Daniel S and Goldbaum, Michael and Cai, Wenjia and Valentim, Carolina CS and Liang, Huiying and Baxter, Sally L and McKeown, Alex and Yang, Ge and Wu, Xiaokang and Yan, Fangbing and others},
                      journal={Cell},
                      volume={172},
                      number={5},
                      pages={1122--1131},
                      year={2018},
                      publisher={Elsevier}}""",
        license="CC BY 4.0",
    )

    def _info(self):
        return DatasetInfo(
            description="""Chest X-ray images (anterior-posterior) from pediatric patients aged one to five years,
                labelled NORMAL or PNEUMONIA (bacterial and viral pneumonia are merged into one class).""",
            features=Features(
                {
                    "image": ImageFeature(),
                    "label": ClassLabel(names=_CHEST_XRAY_CLASSES),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            citation=self.SOURCE["citation"],
            license=self.SOURCE["license"],
        )

    def _generate_examples(self, data_path, split):
        # Archive layout: chest_xray/<split>/<NORMAL|PNEUMONIA>/<file>.jpeg
        # The label comes only from the class folder name, not the file name.
        if split not in ("train", "test"):
            raise ValueError(f"Unknown ChestXRayPneumonia split: {split!r}")

        with zipfile.ZipFile(data_path) as archive:
            # Sort for a deterministic example order across runs.
            for info in sorted(archive.infolist(), key=lambda i: i.filename):
                parts = info.filename.split("/")
                if (
                    info.is_dir()
                    or "__MACOSX" in parts
                    or parts[-1].startswith("._")
                    or not parts[-1].lower().endswith((".jpeg", ".jpg"))
                ):
                    continue

                # Only read files that belong to the requested split.
                if len(parts) < 3 or parts[-3] != split:
                    continue

                label = _FOLDER_TO_LABEL[parts[-2]]  # KeyError on an unexpected folder: fail loudly

                image_bytes = archive.read(info)
                with PILImage.open(io.BytesIO(image_bytes)) as img:
                    # Keep the original JPEG bytes when already RGB; convert grayscale scans to RGB.
                    image = image_bytes if img.mode == "RGB" else img.convert("RGB")

                yield info.filename, {"image": image, "label": label}