import torch
import os
import pickle
import copy

def create_dir(folder="plots"):
    if os.path.exists(folder) == False:
        os.mkdir(folder)

def get_sample_dataloader(data_loader):
    dataiter = iter(data_loader)
    batch_data = dataiter.next()
    return batch_data[0], batch_data[1]

def parse_activation(activation="relu"):
    if isinstance(activation, str):
        if activation == 'sigmoid':
            activation = torch.nn.Sigmoid()
        elif activation == 'tanh':
            activation = torch.nn.Tanh()
        elif activation == 'relu':
            activation = torch.nn.ReLU()
        elif activation == 'leakyrelu':
            activation = torch.nn.LeakyReLU()
        elif activation == 'none':
            activation = None
    else:
        activation = copy.deepcopy(activation)
    return activation

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
    #add '.p' to filename if not already
    if ".p" not in model_name:
        model_name = model_name+".p"
    #add "/" to folder path, if not already
    if folder[-1] != '/':
        folder = folder+"/"
    #load pickled model
    bae_model = pickle.load( open(folder+model_name, "rb"))
    return bae_model

def save_csv_pd(results_pd,folder="results",train_set_name="FashionMNIST",title="auroc"):
    create_dir(folder)
    save_path = folder+"/"+train_set_name+"_"+title+".csv"
    csv_exists = os.path.exists(save_path)
    csv_mode = 'a' if csv_exists else 'w'
    header_mode = False if csv_exists else True
    results_pd.to_csv(save_path, mode=csv_mode, header=header_mode, index=False)

