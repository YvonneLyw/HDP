if __name__ == "__main__":
    import sys
    import pathlib
    ROOT_DIR = str(pathlib.Path(__file__).resolve().parent.parent)
    sys.path.append(ROOT_DIR)

import shutil
import pathlib
import h5py
from tqdm import tqdm


def main():
    src_pcd_path = pathlib.Path("data/robomimic/datasets/can/ph/low_dim_abs_pcd.hdf5")
    src_img_abs_path = pathlib.Path("data/robomimic/datasets/can/ph/image_abs.hdf5")
    dst_path = pathlib.Path("data/robomimic/datasets/can/ph/image_abs_pcd.hdf5")

    assert src_pcd_path.is_file(), f"PCD source not found: {src_pcd_path}"
    assert src_img_abs_path.is_file(), f"Image abs source not found: {src_img_abs_path}"
    assert dst_path.parent.is_dir(), f"Output dir not found: {dst_path.parent}"

    print(f"Copying:\n  {src_img_abs_path}\n-> {dst_path}")
    if dst_path.exists():
        dst_path.unlink()
        print(f"Removed existing file: {dst_path}")
    shutil.copyfile(src_img_abs_path, dst_path)

    with h5py.File(src_pcd_path, "r") as f_pcd:
        demos_pcd = f_pcd["data"]
        demo0 = demos_pcd["demo_0"]

        scene_pcd = demo0["scene_pcd"][:]
        object_pcd = demo0["object_pcd"][:]
        goal = demo0["goal"][:]

        print("Loaded from low_dim_abs_pcd.hdf5:")
        print("  scene_pcd shape:", scene_pcd.shape)
        print("  object_pcd shape:", object_pcd.shape)
        print("  goal shape:", goal.shape)

    with h5py.File(dst_path, "r+") as f_out:
        demos_out = f_out["data"]
        demo_keys = list(demos_out.keys())

        for k in tqdm(demo_keys, desc="Writing pcd/meta"):
            demo = demos_out[k]

            if "scene_pcd" in demo:
                del demo["scene_pcd"]
            if "object_pcd" in demo:
                del demo["object_pcd"]
            if "goal" in demo:
                del demo["goal"]

            demo.create_dataset("scene_pcd", data=scene_pcd)
            demo.create_dataset("object_pcd", data=object_pcd)
            demo.create_dataset("goal", data=goal)

    print("Done.")
    print(f"Saved to: {dst_path}")


if __name__ == "__main__":
    main()