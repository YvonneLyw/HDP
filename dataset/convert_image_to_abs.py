if __name__ == "__main__":
    import sys
    import pathlib
    ROOT_DIR = str(pathlib.Path(__file__).resolve().parent.parent)
    sys.path.append(ROOT_DIR)

import shutil
import pathlib
import h5py
from tqdm import tqdm

from hiera_diffusion_policy.common.robomimic_util import RobomimicAbsoluteActionConverter


def main():
    input_path = pathlib.Path("data/robomimic/datasets/can/ph/image.hdf5")
    output_path = pathlib.Path("data/robomimic/datasets/can/ph/image_abs.hdf5")

    assert input_path.is_file(), f"Input file not found: {input_path}"
    assert output_path.parent.is_dir(), f"Output dir not found: {output_path.parent}"

    print(f"Converting:\n  {input_path}\n-> {output_path}")

    converter = RobomimicAbsoluteActionConverter(str(input_path))
    print(f"Number of demos: {len(converter)}")

    if output_path.exists():
        output_path.unlink()
        print(f"Removed existing file: {output_path}")

    print("Copying original file...")
    shutil.copy(str(input_path), str(output_path))

    print("Writing absolute actions...")
    with h5py.File(output_path, "r+") as f:
        for i in tqdm(range(len(converter)), desc="Converting demos"):
            abs_actions = converter.convert_idx(i)
            f[f"data/demo_{i}/actions"][:] = abs_actions

    print("Done.")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()