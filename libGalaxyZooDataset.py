import os
from PIL import Image
import matplotlib.image as mpimg
import torch
from torch.utils.data.dataloader import default_collate

class GalaxyZooDataset(torch.utils.data.Dataset):
    """
        A custom dataset class for loading Galaxy Zoo images with optional
        transformations.

        Args:
            - df (pd.DataFrame): A DataFrame containing metadata for the images.
                It should have at least one column named 'asset_id', which
                corresponds to the image filename (without the extension).
            - data_dir (str): Path to the directory containing the image files.
                            The images are assumed to be located in
                            a subdirectory named 'images' within this directory.
            - transform (callable, optional): A function or transform to be
                            applied to each image (e.g., a torchvision transform).
                            Default is None, meaning no transformation will be applied.

        Methods:
            __len__(): Returns the total number of images in the dataset.
            __getitem__(idx): Given an index, loads the corresponding image and
                                applies the optional transformation.

        Returns:
            A tuple (image, img_name), where:
                - image (PIL.Image or transformed image): The loaded image,
                        possibly transformed.
                - img_name (str): The file path of the image.

        Example:
            dataset = GalaxyZooDataset(df=my_dataframe, data_dir='/path/to/data',
                                        transform=my_transform)
            image, img_name = dataset[0]  # Access the first image
        """
    def __init__(self, df, data_dir, transform=None):
        self.df = df
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, 'images',
                                str(self.df.iloc[idx]['asset_id']) + '.jpg')
        img_type = str(self.df.iloc[idx]['gz2_class'])
        try:
            #image = mpimg.imread(img_name)
            #image = Image.fromarray(image)
            image = Image.open(img_name).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, img_name, img_type
        except FileNotFoundError:
            return None  


def custom_collate(batch):
    """
    A custom collate function that filters out None values from the batch

    Args:
        - batch (list): A list of individual data samples returned
            by the dataset's __getitem__ method.

    Returns:
        - A batch of valid data samples after filtering out the None values.
    """
    batch = [b for b in batch if b is not None]

    if len(batch) == 0:
        return None
    return default_collate(batch)