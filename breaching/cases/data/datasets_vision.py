"""Additional torchvision-like datasets."""

import torch
import torchvision

import os
import glob
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from torchvision.datasets.utils import download_and_extract_archive
import hashlib

import concurrent.futures

import csv
from torchvision import datasets
import nibabel as nib
from PIL import Image
import shutil

import torch
import torch.utils.data

import torchvision.datasets
import os
import torchio as tio
import torchvision.io as io
import os
import torch
import torchio
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader
import os
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset, DataLoader
import os
import shutil
import nibabel as nib
from PIL import Image
import numpy as np

def _build_dataset_vision(cfg_data, split, can_download=True):
    _default_t = torchvision.transforms.ToTensor()
    cfg_data.path = os.path.expanduser(cfg_data.path)

    if cfg_data.name == "BrainTumor":
    
        #dataset = BrainTumor(root=cfg_data.path,split=split, transform=_default_t)
       
        #dataset.lookup = dict(zip(list(range(len(dataset))), dataset.labels))
        dataset = torchvision.datasets.ImageFolder(root='/home/ubuntu/data/imagenet/BrainTumor/train',transform=_default_t)
        dataset_loader = torch.utils.data.DataLoader(dataset,batch_size=32,shuffle=True,num_workers=4)
        dataset.lookup = dict(zip(list(range(len(dataset))), dataset.targets))

    elif cfg_data.name == "MosMedData":
        dataset = torchvision.datasets.ImageFolder(root='/home/ubuntu/data/imagenet/MosMedData/train',transform=_default_t)
        dataset_loader = torch.utils.data.DataLoader(dataset,batch_size=32,shuffle=True,num_workers=4)
        dataset.lookup = dict(zip(list(range(len(dataset))), dataset.targets))
        




    elif cfg_data.name == "MOSMEDORIGINAL":
        

        class CustomDataset(Dataset):
            def __init__(self, root_dir, transform=None):
                self.root_dir = root_dir
                self.transform = transform
                self.samples = []  # List to store (image_path, label) tuples
                self.classes = []  # List to store class labels
                
                # Iterate through subdirectories to get image paths and labels
                for label, class_dir in enumerate(os.listdir(root_dir)):
                    class_path = os.path.join(root_dir, class_dir)
                    if os.path.isdir(class_path):
                        self.classes.append(class_dir)  # Store class label
                        for image_file in os.listdir(class_path):
                            if image_file.endswith('.nii.gz'):
                                image_path = os.path.join(class_path, image_file)
                                self.samples.append((image_path, label))

                # Create a lookup dictionary mapping index to target
                self.lookup = dict(zip(range(len(self.samples)), [sample[1] for sample in self.samples]))

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                image_path, label = self.samples[idx]
                sample = self.read_3d_image(image_path)

                if self.transform:
                    sample = self.transform(sample)

                return sample, label

            def read_3d_image(self, image_path):
                image = sitk.ReadImage(image_path)
                image_array = sitk.GetArrayFromImage(image)
                # You may need to preprocess the image array here
                
                return image_array

        # Example usage:
        root_dir = '/home/ubuntu/data/imagenet/MOSMEDORIGINAL/train'
        dataset = CustomDataset(root_dir=root_dir, transform=_default_t)
       
        dataset_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
        dataset.lookup = dict(zip(range(len(dataset)), [sample[1] for sample in dataset.samples]))
       






    elif cfg_data.name == "MosMedData_3D":
        data_root = '/home/ubuntu/data/imagenet/MosMedData_3D/train'

        # Convert NIfTI files to JPEG slices
        convert_nifty_to_slices(data_root)
        # Create the ImageFolder dataset
        dataset = datasets.ImageFolder(root=os.path.join(data_root), transform=_default_t)
        dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

        # Create a lookup dictionary mapping index to target
        dataset.lookup = dict(zip(list(range(len(dataset))), dataset.targets))

    elif cfg_data.name == "Healthy_FullBody_3D":
        data_root = '/home/ubuntu/data/imagenet/Healthy_FullBody_3D/train'

        # Convert NIfTI files to JPEG slices
        convert_nifty_to_slices(data_root)
        # Create the ImageFolder dataset
        dataset = datasets.ImageFolder(root=os.path.join(data_root), transform=_default_t)
        dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

        # Create a lookup dictionary mapping index to target
        dataset.lookup = dict(zip(list(range(len(dataset))), dataset.targets))


    elif cfg_data.name == "CIFAR10":
        dataset = torchvision.datasets.CIFAR10(
            root=cfg_data.path, train=split == "training", download=can_download, transform=_default_t,
        )
        dataset.lookup = dict(zip(list(range(len(dataset))), dataset.targets))
        
        
    elif cfg_data.name == "CIFAR100":
        dataset = torchvision.datasets.CIFAR100(
            root=cfg_data.path, train=split == "training", download=can_download, transform=_default_t,
        )
        dataset.lookup = dict(zip(list(range(len(dataset))), dataset.targets))
    elif cfg_data.name == "ImageNet":
        dataset = torchvision.datasets.ImageNet(
            root=cfg_data.path, split="train" if "train" in split else "val", transform=_default_t,
        )
        dataset.lookup = dict(zip(list(range(len(dataset))), [label for (_, label) in dataset.samples]))
    elif cfg_data.name == "ImageNetAnimals":
        dataset = torchvision.datasets.ImageNet(
            root=cfg_data.path, split="train" if "train" in split else "val", transform=_default_t,
        )
        dataset.lookup = dict(zip(list(range(len(dataset))), [label for (_, label) in dataset.samples]))
        indices = [idx for (idx, label) in dataset.lookup.items() if label < 397]
        dataset.classes = dataset.classes[:397]
        dataset.samples = [dataset.samples[i] for i in indices]  # Manually remove samples instead of using a Subset
        dataset.lookup = dict(zip(list(range(len(dataset))), [label for (_, label) in dataset.samples]))
    elif cfg_data.name == "TinyImageNet":
        dataset = TinyImageNet(
            root=cfg_data.path, split=split, download=can_download, transform=_default_t, cached=True,
        )
        dataset.lookup = dict(zip(list(range(len(dataset))), dataset.targets))
       
    elif cfg_data.name == "Birdsnap":
        dataset = Birdsnap(root=cfg_data.path, split=split, download=can_download, transform=_default_t)
        dataset.lookup = dict(zip(list(range(len(dataset))), dataset.labels))
       
    else:
        raise ValueError(f"Invalid dataset {cfg_data.name} provided.")

    if cfg_data.mean is None and cfg_data.normalize:
        data_mean, data_std = _get_meanstd(dataset)
        cfg_data.mean = data_mean
        cfg_data.std = data_std

    transforms = _parse_data_augmentations(cfg_data, split)

    # Apply transformations
    dataset.transform = transforms if transforms is not None else None

    # Save data mean and data std for easy access:
    if cfg_data.normalize:
        dataset.mean = cfg_data.mean
        dataset.std = cfg_data.std
    else:
        dataset.mean = [0]
        dataset.std = [1]

    # Reduce train dataset according to cfg_data.size:
    if cfg_data.size < len(dataset):
        dataset = Subset(dataset, torch.arange(0, cfg_data.size))

    collate_fn = _torchvision_collate
    return dataset, collate_fn




def convert_nifty_to_slices(directory):
    # Create a copy of the original directory if it doesn't already exist
    directory_copy = directory + "_copy"
    if not os.path.exists(directory_copy):
        shutil.copytree(directory, directory_copy)

    def process_nifti_file(nifty_path, output_directory):
        try:
            img = nib.load(nifty_path)
        except Exception as e:
            print(f"Error loading {nifty_path}: {str(e)}")
            return

        # Get the data array from the NIfTI file
        data = img.get_fdata()
        data_normalized = ((data - np.min(data)) / (np.max(data) - np.min(data))) * 255.0  # Normalize data

        # Extract original file name (without extension)
        file_name = os.path.splitext(os.path.basename(nifty_path))[0]

        # Determine the output subdirectory based on the NIfTI file path and ensure it exists
        output_subdir = os.path.join(output_directory, os.path.relpath(os.path.dirname(nifty_path), directory_copy))
        os.makedirs(output_subdir, exist_ok=True)

        # Save each slice as a TIFF file in the corresponding subfolder
        for i in range(data.shape[2]):
           
            slice_img = Image.fromarray(data_normalized[:, :, i].astype('uint8'))
            slice_path = os.path.join(output_subdir, f"{file_name}_slice_{i + 1}.tif")
            slice_img.resize((1024, 1024), Image.BICUBIC).save(slice_path, format='TIFF')

        # Note: The original NIfTI file is not removed in this version, to preserve data in the copy

    # Loop through all subdirectories and files in the copied directory
    for root, dirs, files in os.walk(directory_copy):
        for file in files:
            if file.lower().endswith(".nii.gz") or file.lower().endswith(".nii"):
                nifty_path = os.path.join(root, file)
                # Process each file, saving output in the original directory structure
                process_nifti_file(nifty_path, directory)

        print(f"Slices processed for {root}")

# Example usage
# convert_nifti_to_tiff('/path/to/your/directory')




# def convert_nifty_to_slices(directory):
#     # Create a copy of the original directory if it doesn't already exist
#     directory_copy = directory + "_copy"
#     if not os.path.exists(directory_copy):
#         shutil.copytree(directory, directory_copy)

#     # Function to read NIfTI files, save slices, and remove the original NIfTI files
#     def process_nifti_file(nifty_path):
#         try:
#             img = nib.load(nifty_path)
#         except Exception as e:
#             print(f"Error loading {nifty_path}: {str(e)}")
#             return

#         # Get the data array from the NIfTI file
#         data = img.get_fdata()

#         # Extract original file name (without extension)
#         file_name = os.path.splitext(os.path.basename(nifty_path))[0]

#         # Save each slice as a JPG file in the corresponding subfolder
#         for i in range(data.shape[2]):
#             slice_img = Image.fromarray(data[:, :, i].astype('uint8'))
#             slice_path = os.path.join(os.path.dirname(nifty_path), f"{file_name}_slice_{i + 1}.jpg")
#             slice_img.resize((1024, 1024)).save(slice_path, "JPEG")

#         # Remove the original NIfTI file
#         os.remove(nifty_path)

#     # Loop through all subdirectories and files in the copy
#     for root, dirs, files in os.walk(directory):
#         for file in files:
#             if file.lower().endswith(".nii.gz") or file.lower().endswith(".nii"):
#                 nifty_path = os.path.join(root, file)
#                 process_nifti_file(nifty_path)

#         print(f"Slices received for {root}")

def _split_dataset_vision(dataset, cfg_data, user_idx=None, return_full_dataset=False):
    if not return_full_dataset:
        if user_idx is None:
            user_idx = torch.randint(0, cfg_data.default_clients, (1,))
        else:
            if user_idx > cfg_data.default_clients:
                raise ValueError("This user index exceeds the maximal number of clients.")

        # Create a synthetic split of the dataset over all possible users if no natural split is given
       
        if cfg_data.partition == "balanced":
            
            #print(f"{len(dataset.classes)} len(dataset.classes)")
            
            data_per_class_per_user = len(dataset) // len(dataset.classes) // cfg_data.default_clients
           
            #print(f"{len(dataset.classes)} after len(dataset.classes)")
           
            if data_per_class_per_user < 1:
                raise ValueError("Too many clients for a balanced dataset.")
            data_ids = []
            for class_idx, _ in enumerate(dataset.classes):
                data_with_class = [idx for (idx, label) in dataset.lookup.items() if label == class_idx]
                data_ids += data_with_class[
                    user_idx * data_per_class_per_user : data_per_class_per_user * (user_idx + 1)
                ]
            dataset = Subset(dataset, data_ids)
        elif cfg_data.partition == "unique-class":
            data_ids = [idx for (idx, label) in dataset.lookup.items() if label == user_idx]
            dataset = Subset(dataset, data_ids)
        elif cfg_data.partition == "mixup":
            if "mixup_freq" in cfg_data:
                mixup_freq = cfg_data.mixup_freq
            else:
                # use default mixup_freq=2
                mixup_freq = 2
            data_per_user = len(dataset) // cfg_data.default_clients
            last_id = len(dataset) - 1
            data_ids = []
            for i in range(data_per_user):
                data_ids.append(user_idx * data_per_user + i)
                data_ids.append(last_id - user_idx * data_per_user - i)
            dataset = Subset(dataset, data_ids)
        elif cfg_data.partition == "feat_est":
            if "num_data_points" in cfg_data:
                num_data_points = cfg_data.num_data_points
            else:
                num_data_points = 1

            if "target_label" in cfg_data:
                target_label = cfg_data.target_label
            else:
                target_label = 0

            data_ids = [idx for (idx, label) in dataset.lookup.items() if label == target_label]
            data_ids = data_ids[user_idx * num_data_points : (user_idx + 1) * num_data_points]
            dataset = Subset(dataset, data_ids)
        elif cfg_data.partition == "random-full":  # Data might be repeated across users (e.g. meme images)
            data_per_user = len(dataset) // cfg_data.default_clients
            data_ids = torch.randperm(len(dataset))[:data_per_user]
            dataset = Subset(dataset, data_ids)
        elif cfg_data.partition == "random":  # Data not replicated across users. Split is deterministic over reruns!
            data_per_user = len(dataset) // cfg_data.default_clients
            generator = torch.Generator()
            generator.manual_seed(233)
            data_ids = torch.randperm(len(dataset))[user_idx * data_per_user : data_per_user * (user_idx + 1)]
            dataset = Subset(dataset, data_ids)
        elif cfg_data.partition == "none":  # Replicate on all users for a sanity check!
            pass
        else:
            raise ValueError(f"Partition scheme {cfg_data.partition} not implemented.")
    return dataset


def _torchvision_collate(batch):
    """Small hack around the pytorch default collator to return a dictionary"""
    transposed = list(zip(*batch))

    def _stack_tensor(tensor_list):
        elem = tensor_list[0]
        elem_type = type(elem)
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in tensor_list)
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(tensor_list, 0, out=out)

    return dict(inputs=_stack_tensor(transposed[0]), labels=torch.tensor(transposed[1]))


class Subset(torch.utils.data.Subset):
    """Overwrite subset class to provide class methods of main class."""

    def __getattr__(self, name):
        """Call this only if all attributes of Subset are exhausted."""
        return getattr(self.dataset, name)


def _get_meanstd(dataset):
    print("Computing dataset mean and std manually ... ")
    # Run parallelized Wellford:
    current_mean = 0
    current_M2 = 0
    n = 0
    for data, _ in dataset:
        datapoint = data.view(3, -1)
        ds, dm = torch.std_mean(datapoint, dim=1)
        n_a, n_b = n, datapoint.shape[1]
        n += n_b
        delta = dm.to(dtype=torch.double) - current_mean
        current_mean += delta * n_b / n
        current_M2 += ds.to(dtype=torch.double) / (n_b - 1) + delta ** 2 * n_a * n_b / n
        # print(current_mean, (current_M2 / (n - 1)).sqrt())

    data_mean = current_mean.tolist()
    data_std = (current_M2 / (n - 1)).sqrt().tolist()
    print(f"Mean: {data_mean}. Standard deviation: {data_std}")
    return data_mean, data_std


def _parse_data_augmentations(cfg_data, split, PIL_only=False):
    def _parse_cfg_dict(cfg_dict):
        list_of_transforms = []
        if hasattr(cfg_dict, "keys"):
            for key in cfg_dict.keys():
                try:  # ducktype iterable
                    transform = getattr(torchvision.transforms, key)(*cfg_dict[key])
                except TypeError:
                    transform = getattr(torchvision.transforms, key)(cfg_dict[key])
                list_of_transforms.append(transform)
        return list_of_transforms

    if split == "train":
        transforms = _parse_cfg_dict(cfg_data.augmentations_train)
    else:
        transforms = _parse_cfg_dict(cfg_data.augmentations_val)

    if not PIL_only:
        transforms.append(torchvision.transforms.ToTensor())
        if cfg_data.normalize:
            transforms.append(torchvision.transforms.Normalize(cfg_data.mean, cfg_data.std))
    return torchvision.transforms.Compose(transforms)


class TinyImageNet(torch.utils.data.Dataset):
    """Tiny ImageNet data set available from `http://cs231n.stanford.edu/tiny-imagenet-200.zip`.

    This is a TinyImageNet variant to the code of Meng Lee, mnicnc404 / Date: 2018/06/04
    References:
        - https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel.html
    Parameters
    ----------
    root: string
        Root directory including `train`, `test` and `val` subdirectories.
    split: string
        Indicating which split to return as a data set.
        Valid option: [`train`, `test`, `val`]
    transform: torchvision.transforms
        A (series) of valid transformation(s).
    cached: bool
        Set to True if there is enough memory (about 5G) and want to minimize disk IO overhead.
    download: bool
        Set to true to automatically download the dataset in to the root folder.
    """

    EXTENSION = "JPEG"
    NUM_IMAGES_PER_CLASS = 500
    CLASS_LIST_FILE = "wnids.txt"
    VAL_ANNOTATION_FILE = "val_annotations.txt"
    CLASSES = "words.txt"

    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    archive = "tiny-imagenet-200.zip"
    folder = "tiny-imagenet-200"
    train_md5 = "c77c61d662a966d2fcae894d82df79e4"
    val_md5 = "cef44e3f1facea2ea8cd5e5a7a46886c"
    test_md5 = "bc72ebd5334b12e3a7ba65506c0f8bc0"

    def __init__(self, root, split="train", transform=None, target_transform=None, cached=True, download=True):
        """Init with split, transform, target_transform."""
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.cached = cached

        self.split_dir = os.path.join(root, self.folder, self.split)
        self.image_paths = sorted(
            glob.iglob(os.path.join(self.split_dir, "**", "*.%s" % self.EXTENSION), recursive=True)
        )
        self.labels = {}  # fname - label number mapping

        if download:
            self.download()

        self._parse_labels()

        if self.cached:
            self._build_cache()

    def _check_integrity(self):
        """This only checks if all files are there."""
        string_rep = "".join(self.image_paths).encode("utf-8")
        hash = hashlib.md5(string_rep)
        if self.split == "train":
            return hash.hexdigest() == self.train_md5
        elif self.split == "val":
            return hash.hexdigest() == self.val_md5
        else:
            return hash.hexdigest() == self.test_md5

    def download(self):
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_and_extract_archive(self.url, self.root, filename=self.archive)

    def _parse_labels(self):
        with open(os.path.join(self.root, self.folder, self.CLASS_LIST_FILE), "r") as fp:
            self.label_texts = sorted([text.strip() for text in fp.readlines()])
        self.label_text_to_number = {text: i for i, text in enumerate(self.label_texts)}

        if self.split == "train":
            for label_text, i in self.label_text_to_number.items():
                for cnt in range(self.NUM_IMAGES_PER_CLASS):
                    self.labels["%s_%d.%s" % (label_text, cnt, self.EXTENSION)] = i
        elif self.split == "val":
            with open(os.path.join(self.split_dir, self.VAL_ANNOTATION_FILE), "r") as fp:
                for line in fp.readlines():
                    terms = line.split("\t")
                    file_name, label_text = terms[0], terms[1]
                    self.labels[file_name] = self.label_text_to_number[label_text]

        # Build class names
        label_text_to_word = dict()
        with open(os.path.join(self.root, self.folder, self.CLASSES), "r") as file:
            for line in file:
                label_text, word = line.split("\t")
                label_text_to_word[label_text] = word.split(",")[0].rstrip("\n")
        self.classes = [label_text_to_word[label] for label in self.label_texts]

        # Prepare index - label mapping
        self.targets = [self.labels[os.path.basename(file_path)] for file_path in self.image_paths]

    def _build_cache(self):
        """Cache images in RAM."""
        self.cache = []
        for index in range(len(self)):
            img = Image.open(self.image_paths[index])
            img = img.convert("RGB")
            self.cache.append(img)

    def __len__(self):
        """Return length via image paths."""
        return len(self.image_paths)

    def __getitem__(self, index):
        """Return image, label."""
        if self.cached:
            img = self.cache[index]
        else:
            img = Image.open(self.image_paths[index])
            img = img.convert("RGB")
        target = self.targets[index]

        img = self.transform(img) if self.transform else img
        target = self.target_transform(target) if self.target_transform else target
        if self.split == "test":
            return img, None
        else:
            return img, target


class Birdsnap(torch.utils.data.Dataset):
    """This is the BirdSnap dataset presented in
    - Berg et al., "Birdsnap: Large-scale Fine-grained Visual Categorization of Birds"
    It contains a lot of classes of birds and can be used as a replacement for ImageNet validation images
    with similar image fidelity but less of the baggage, given that all subjects are in fact birds.

    This is too small to train on though and hence not even partitioned into train/test.
    Several images are missing from flickr (in 2021), these will be discarded automatically.
    """

    METADATA_URL = "http://thomasberg.org/datasets/birdsnap/1.1/birdsnap.tgz"
    METADATA_ARCHIVE = "birdsnap.tgz"
    META_MD5 = "1788158175f6ae794aebf27bcd7a3f5d"
    BASE_FOLDER = "birdsnap"

    def __init__(self, root, split="train", transform=None, target_transform=None, download=True, crop_to_bbx=False):
        """Init with split, transform, target_transform."""
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        self.crop_to_bbx = crop_to_bbx  # Crop to dataset default bounding boxes

        if download:
            self.download()
        if not self.check_integrity():
            raise ValueError("Dataset Birdsnap not downloaded completely or possibly corrupted.")

        self._purge_missing_data()

    def _check_integrity_of_metadata(self, chunk_size=8192):
        """This only checks if all files are there."""
        try:
            with open(os.path.join(self.root, self.METADATA_ARCHIVE), "rb") as f:
                archive_hash = hashlib.md5()
                while chunk := f.read(chunk_size):
                    archive_hash.update(chunk)
            return self.META_MD5 == archive_hash.hexdigest()
        except FileNotFoundError:
            return False

    def check_integrity(self):
        """Full integrity check."""
        if not self._check_integrity_of_metadata():
            return False
        else:
            self._parse_metadata()
            missing_images = 0
            for idx, file in enumerate(self.meta):
                if not self._verify_image(idx):
                    missing_images += 1
            if missing_images > 0:
                print(f"{missing_images} images could not be downloaded.")
            return True

    def download(self):
        # Metadata:
        if self._check_integrity_of_metadata():
            print("Metadata already downloaded and verified")
        else:
            download_and_extract_archive(self.METADATA_URL, self.root, filename=self.METADATA_ARCHIVE)
        # Actual files:
        self._parse_metadata()

        missing_ids = []
        for idx, file in enumerate(self.meta):
            if not self._verify_image(idx):
                missing_ids.append(idx)
        if len(missing_ids) > 0:
            print(f"Downloading {len(missing_ids)} missing files now...")
            self.scrape_images(missing_ids)

    def __len__(self):
        """Return length via metainfo."""
        return len(self.meta)

    def __getitem__(self, index):
        """Return image, label."""
        img = Image.open(self.paths[index])
        if self.crop_to_bbx:
            img = img.crop(
                (
                    self.meta[index]["bb_x1"],
                    self.meta[index]["bb_y1"],
                    self.meta[index]["bb_x2"],
                    self.meta[index]["bb_y2"],
                )
            )
        img = img.convert("RGB")
        label = self.labels[index]

        img = self.transform(img) if self.transform else img
        label = self.target_transform(label) if self.target_transform else label
        return img, label

    def _parse_metadata(self):
        """Metadata keys are
        dict_keys(['url', 'md5', 'path', 'species_id', 'bb_x1', 'bb_y1', 'bb_x2', 'bb_y2', 'back_x', 'back_y', 'beak_x',
        'beak_y', 'belly_x', 'belly_y', 'breast_x', 'breast_y', 'crown_x', 'crown_y', 'forehead_x', 'forehead_y',
        'left_cheek_x', 'left_cheek_y', 'left_eye_x', 'left_eye_y', 'left_leg_x', 'left_leg_y', 'left_wing_x',
        'left_wing_y', 'nape_x', 'nape_y', 'right_cheek_x', 'right_cheek_y', 'right_eye_x', 'right_eye_y',
        'right_leg_x', 'right_leg_y', 'right_wing_x', 'right_wing_y', 'tail_x', 'tail_y', 'throat_x', 'throat_y']
        """
        with open(os.path.join(self.root, self.BASE_FOLDER, "images.txt"), "r") as f:
            reader = csv.DictReader(f, delimiter="\t")
            self.meta = list(reader)  # List of dictionaries.
        self.labels = [int(entry["species_id"]) for entry in self.meta]
        self.paths = [os.path.join(self.root, self.BASE_FOLDER, entry["path"]) for entry in self.meta]
        with open(os.path.join(self.root, self.BASE_FOLDER, "species.txt"), "r") as f:
            reader = csv.DictReader(f, delimiter="\t")
            self.classes_metadata = list(reader)
        self.classes = [str(entry["common"]) for entry in self.classes_metadata]

    def _verify_image(self, idx):
        try:
            # Do this if you want to check in detail:
            # with open(os.path.join(self.root, self.BASE_FOLDER, self.meta[idx]['path']), 'rb') as fin:
            #     return (hashlib.md5(fin.read()).hexdigest() == self.meta[idx]['md5'])
            # In the mean time, just check if everything is there:
            return os.path.exists(os.path.join(self.root, self.BASE_FOLDER, self.meta[idx]["path"]))
        except FileNotFoundError:
            return False

    def scrape_images(self, missing_ids, chunk_size=8196):
        """Scrape images using the python default ThreadPool example."""
        import requests

        def _load_url_and_save_image(idx, timeout):
            full_path = os.path.join(self.root, self.BASE_FOLDER, self.meta[idx]["path"])
            os.makedirs(os.path.split(full_path)[0], exist_ok=True)
            r = requests.get(self.meta[idx]["url"], stream=True)
            with open(full_path, "wb") as write_file:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    write_file.write(chunk)
            return True

        # We can use a with statement to ensure threads are cleaned up promptly
        with concurrent.futures.ThreadPoolExecutor(max_workers=None) as executor:  # Choose max_workers dynamically
            # Start the load operations and mark each future with its URL
            future_to_url = {
                executor.submit(_load_url_and_save_image, idx, 600): self.meta[idx]["url"] for idx in missing_ids
            }
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    data = future.result()
                except Exception as exc:
                    print(f"{url} generated exception: {exc}")
                else:
                    print(f"{url} downloaded successfully.")

    def _purge_missing_data(self):
        """Iterate over all data and throw out missing images."""
        JPG = b"\xff\xd8\xff"

        clean_meta = []
        invalid_files = 0
        for entry in self.meta:
            full_path = os.path.join(self.root, self.BASE_FOLDER, entry["path"])
            with open(full_path, "rb") as file_handle:
                if file_handle.read(3) == JPG:
                    clean_meta.append(entry)
                else:
                    invalid_files += 1
        print(f"Discarded {invalid_files} invalid files.")
        self.meta = clean_meta

        self.labels = [int(entry["species_id"]) for entry in self.meta]
        self.paths = [os.path.join(self.root, self.BASE_FOLDER, entry["path"]) for entry in self.meta]

class BrainTumor(torch.utils.data.Dataset):

    EXTENSION = "jpg"  # Update with the actual extension of your images

    dataset_classes = ["glioma", "meningioma", "notumor", "pituitary"]

    def __init__(self, root, split="train", transform=None, target_transform=None, cached=True):
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.cached = cached

        # Assuming cfg_data.path() is a method call that returns the root path
        
        self.split_dir = os.path.join(root, 'BrainTumor', split)

        self.subdirectories = sorted([d for d in os.listdir(self.split_dir) if os.path.isdir(os.path.join(self.split_dir, d))])
        self.image_paths = []

        for subdirectory in self.subdirectories:
            subdirectory_path = os.path.join(self.split_dir, subdirectory)
            images_in_subdirectory = sorted(glob.glob(os.path.join(subdirectory_path, f"*.{self.EXTENSION}")))

            self.image_paths.extend(images_in_subdirectory)

        # Prepare index - label mapping
        self.labels = {os.path.basename(subdirectory): i for i, subdirectory in enumerate(self.subdirectories)}
        self.targets = [self.labels[os.path.basename(os.path.dirname(file_path))] for file_path in self.image_paths]

        # Attach dataset.classes
        self.classes = self.dataset_classes

        if self.cached:
            self._build_cache()

        # Generate lookup dictionary
        self.lookup = dict(zip(list(range(len(self))), self.targets))

    def _build_cache(self):
        """Cache images in RAM."""
        self.cache = []
        for index in range(len(self)):
            img = Image.open(self.image_paths[index])
            img = img.convert("RGB")
            self.cache.append(img)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        if self.cached:
            img = self.cache[index]
        else:
            img = Image.open(self.image_paths[index])
            img = img.convert("RGB")
        target = self.targets[index]

        img = self.transform(img) if self.transform else img
        target = self.target_transform(target) if self.target_transform else target

        # Add index to the output
        return index, img, target
