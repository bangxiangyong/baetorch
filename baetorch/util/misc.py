import torch
import os
import pickle

def create_dir(folder="plots"):
    if os.path.exists(folder) == False:
        os.mkdir(folder)

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

def save_bae_model(model,folder="pickles/"):
    create_dir(folder)
    pickle.dump(model, open(folder+model.model_name+".p", "wb"))

def load_bae_model(model_name, folder="pickles/"):
    bae_model = pickle.load( open(folder+model_name+".p", "rb"))
    return bae_model
