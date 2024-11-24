from __future__ import annotations

if __name__ == '__main__':

    print('Verifying the CWD')
    import os
    from pathlib import Path
    from typing import Any

    from git.repo import Repo

    current_directory = Path.cwd()
    if current_directory.name == "000_getting_started":
        # On the assumption that, the notebook is located in
        #   ~/anomalib/notebooks/000_getting_started/
        root_directory = current_directory.parent.parent
    elif current_directory.name == "anomalib":
        # This means that the notebook is run from the main anomalib directory.
        root_directory = current_directory
    else:
        # Otherwise, we'll need to clone the anomalib repo to the `current_directory`
        repo = Repo.clone_from(url="https://github.com/openvinotoolkit/anomalib.git", to_path=current_directory)
        root_directory = current_directory / "anomalib"

    os.chdir(root_directory)


    print('Importing the modules...')
    import numpy as np
    from matplotlib import pyplot as plt
    from PIL import Image
    from pytorch_lightning import Trainer
    from torchvision.transforms import ToPILImage

    from anomalib.config import get_configurable_parameters
    from anomalib.data import get_datamodule
    from anomalib.data.utils import read_image
    from anomalib.deploy import OpenVINOInferencer
    from anomalib.models import get_model
    from anomalib.pre_processing.transforms import Denormalize
    from anomalib.utils.callbacks import LoadModelCallback, get_callbacks

    print('Choosing padim...')
    MODEL = "padim"  # 'padim', 'cflow', 'stfpm', 'ganomaly', 'dfkde', 'patchcore'
    CONFIG_PATH = root_directory / f"src/anomalib/models/{MODEL}/config.yaml"
    with open(file=CONFIG_PATH, mode="r", encoding="utf-8") as file:
        print(file.read())

    print('Getting the parameters...')
    # pass the config file to model, callbacks and datamodule
    config = get_configurable_parameters(config_path=CONFIG_PATH)

    print('Getting the datamodule...')
    datamodule = get_datamodule(config)
    datamodule.prepare_data()  # Downloads the dataset if it's not in the specified `root` directory
    datamodule.setup()  # Create train/val/test/prediction sets.

    i, data = next(enumerate(datamodule.val_dataloader()))
    print(data.keys())

    print('Checking the shape....')
    print(data["image"].shape, data["mask"].shape)

    def show_image_and_mask(sample: dict[str, Any], index: int) -> Image:
        img = ToPILImage()(Denormalize()(sample["image"][index].clone()))
        msk = ToPILImage()(sample["mask"][index]).convert("RGB")

        return Image.fromarray(np.hstack((np.array(img), np.array(msk))))


    # Visualize an image with a mask
    #show_image_and_mask(data, index=0)

    print('Setting up the model and callbacks...')
    # Set the export-mode to OpenVINO to create the OpenVINO IR model.
    config.optimization.export_mode = "openvino"

    # Get the model and callbacks
    model = get_model(config)
    callbacks = get_callbacks(config)

    print('Starting training...')
    # start training
    trainer = Trainer(**config.trainer, callbacks=callbacks)
    trainer.fit(model=model, datamodule=datamodule)