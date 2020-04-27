import torch

def get_sample_dataloader(data_loader):
    dataiter = iter(data_loader)
    batch_data = dataiter.next()
    return batch_data[0], batch_data[1]

class AddNoise(object):
    def __init__(self, noise_factor=0.05):
        self.noise_factor = noise_factor

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        noisy_imgs = pic + self.noise_factor * torch.randn(*pic.shape)
        noisy_imgs = torch.clamp(noisy_imgs, 0., 1.)
        return noisy_imgs

    def __repr__(self):
        return self.__class__.__name__ + '()'
