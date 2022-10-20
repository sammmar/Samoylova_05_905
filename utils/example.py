import torch

def n(ii):
    ii = ii.float() / 255.0
    return ii

def _normalize(input_image):
    """ Normalise input image
    Args:
        input_image (torch.Tensor): The input image

    Returns:
        input_image (torch.Tensor): The normalized input image
    """
    input_image = input_image.float() / 255.0
    return input_image