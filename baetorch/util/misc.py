import time

import torch
import os
import copy


def create_dir(folder="plots"):
    if os.path.exists(folder) == False:
        os.mkdir(folder)


def get_sample_dataloader(data_loader):
    dataiter = iter(data_loader)
    batch_data = dataiter.next()
    return batch_data[0], batch_data[1]


def convert_int_to_list(int_param, num_replicate):
    """
    To handle integer passed as param, creates replicate of list
    """
    if isinstance(int_param, int):
        return [int_param] * num_replicate
    else:
        return int_param


def parse_activation(activation="relu"):
    if isinstance(activation, str):
        if activation == "sigmoid":
            activation = torch.nn.Sigmoid()
        elif activation == "tanh":
            activation = torch.nn.Tanh()
        elif activation == "relu":
            activation = torch.nn.ReLU()
        elif activation == "leakyrelu":
            activation = torch.nn.LeakyReLU(0.01)
        elif activation == "silu":
            activation = torch.nn.SiLU()
        elif activation == "selu":
            activation = torch.nn.SELU()
        elif activation == "softplus":
            activation = torch.nn.Softplus()
        elif activation == "gelu":
            activation = torch.nn.GELU()
        elif activation == "elu":
            activation = torch.nn.ELU()
        elif activation == "none":
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
        noisy_imgs = torch.clamp(noisy_imgs, 0.0, 1.0)
        return noisy_imgs

    def __repr__(self):
        return self.__class__.__name__ + "()"


def save_bae_model(model, folder="trained_models/"):
    create_dir(folder)
    model_path = folder + model.model_name + ".p"
    # pickle.dump(model, open(folder+model.model_name+".p", "wb"))
    # model.set_cuda(False)
    torch.save(model, model_path)


def load_bae_model(model_name, folder="pickles/"):
    # add '.p' to filename if not already
    if ".p" not in model_name:
        model_name = model_name + ".p"
    # add "/" to folder path, if not already
    if folder[-1] != "/":
        folder = folder + "/"
    # load pickled model
    # bae_model = pickle.load( open(folder+model_name, "rb"))
    model_path = folder + model_name
    bae_model = torch.load(model_path, map_location=torch.device("cpu"))
    return bae_model


def save_csv_pd(
    results_pd, folder="results", train_set_name="FashionMNIST", title="auroc"
):
    create_dir(folder)
    save_path = folder + "/" + train_set_name + "_" + title + ".csv"
    csv_exists = os.path.exists(save_path)
    csv_mode = "a" if csv_exists else "w"
    header_mode = False if csv_exists else True
    results_pd.to_csv(save_path, mode=csv_mode, header=header_mode, index=False)


def time_method(func, *args, **func_params):
    start = time.time()
    res = func(*args, **func_params)
    end = time.time()
    time_taken = end - start
    print("FIT TIME TAKEN: {:.5f}".format(time_taken))
    return res, time_taken
