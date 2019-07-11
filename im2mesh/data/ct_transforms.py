import torch
import torch.nn.functional as tf
import numpy as np


class NaiveRescale(object):
    """Rescale the ndarray to a given size by taking every n_i-th pixel of the i-th axis,
    where n_i = image.shape[i]/output_size[i].
    Note that this requires that image.shape[i] is dividable by output_size[i]

    Args:
        output_size (tuple or int): Desired output size.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (tuple))
        self.output_size = output_size

    def __call__(self, image):
        """
        Expect 3 dim ndarray. Note the requirements mentions above
        """
        assert isinstance(image, np.ndarray)
        x_scalefactor = image.shape[0]//self.output_size[0]
        y_scalefactor = image.shape[1]//self.output_size[1]
        z_scalefactor = image.shape[2]//self.output_size[2]

        image = image[0::x_scalefactor, 0::y_scalefactor, 0::z_scalefactor]

        return image


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image):
        # swap depth axis because
        # numpy image: H x W x D
        # torch image: C x D x H x W
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image.astype('int32'))
        print(image)
        return image


class ReplicateBorderPadding3D(object):
    """Pad the ndarray in a sample to a given size.

    Args:
        output_size (3-tuple): output size of the padded image
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (tuple))
        self.output_size = output_size

    def __call__(self, image):
        """
        expect 3 dim ndarray in H x W x D
        """
        assert isinstance(image, np.ndarray)

        # pad image to the given size
        x_pad = self.output_size[0] - image.shape[0]
        y_pad = self.output_size[1] - image.shape[1]
        z_pad = self.output_size[2] - image.shape[2]
        if x_pad < 0 or y_pad < 0 or z_pad < 0:
            raise ("Image dimension is bigger than the output size")

        # determine 'before' and 'after' for each dimension
        x_pad_bf = (x_pad // 2)
        x_pad_af = x_pad - x_pad_bf
        y_pad_bf = (y_pad // 2)
        y_pad_af = y_pad - y_pad_bf
        z_pad_bf = (z_pad // 2)
        z_pad_af = z_pad - z_pad_bf

        image = np.pad(image, ((x_pad_bf, x_pad_af), (y_pad_bf, y_pad_af), (z_pad_bf, z_pad_af)), 'edge')
        return image


class CropZCenter3D(object):
    """Crop the center pixel of the Z dimension

    Args:
        center_size : center_size in pixels
    """

    def __init__(self, center_size):
        assert isinstance(center_size, (int))
        self.center_size = center_size

    def __call__(self, image):
        """
        expect 3 dim ndarray in H x W x D
        """
        assert isinstance(image, np.ndarray)

        # find out diff between actual size and center_size
        z_diff = image.shape[2] - self.center_size
        if z_diff < 0:
            raise ValueError("Image dimension is smaller than the center size")

        # crop 'center' X x Y x center_size pixels of the image.
        image = image[:, :, z_diff:z_diff + self.center_size]

        return image
