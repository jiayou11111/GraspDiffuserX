import h5py

# 文件路径
hdf5_file_path = "data/robomimic/datasets/lift/mh/pink_weita_delete0data_initdofchange_camerachange_BGR_240320.hdf5"
# hdf5_file_path = "real_data/dataset.hdf5"

def print_hdf5_structure(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"Dataset: {name}, shape: {obj.shape}, dtype: {obj.dtype}")
    elif isinstance(obj, h5py.Group):
        print(f"Group: {name}")

with h5py.File(hdf5_file_path, "r") as f:
    f.visititems(print_hdf5_structure)

