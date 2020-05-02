from scipy.signal import find_peaks
from math import log10, floor
import matplotlib.pyplot as plt
import numpy as np
import copy
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from tqdm import tqdm

def plot_learning_rate_iterations(train_batch_number, lr_list):
    plt.figure()
    plt.plot(np.arange(train_batch_number),lr_list)
    plt.title("Learning rate ranges")
    plt.ylabel("Learning rate")
    plt.xlabel("Iterations")

def plot_learning_rate_finder(X, y, gp_mean,negative_peaks,minimum_lr,maximum_lr):
    plt.figure()
    plt.axvspan(minimum_lr, maximum_lr, color='green', alpha=0.3)
    plt.plot(X,y)
    plt.plot(X,gp_mean)
    plt.plot(X[negative_peaks], gp_mean[negative_peaks], "x", c="red")
    plt.xscale('log')

    #min and max vertical lines
    min_max_lr_text = "Min lr:{} , Max lr: {}".format(minimum_lr,maximum_lr)
    ymin_plot = np.min(gp_mean)
    ymax_plot = np.max(gp_mean)
    plt.vlines(maximum_lr,ymin_plot,ymax_plot)
    plt.vlines(minimum_lr,ymin_plot,ymax_plot)

    plt.xlabel("Learning rate (log-scale)")
    plt.ylabel("Loss (scaled)")
    plt.title(min_max_lr_text)


def run_auto_lr_range(train_loader, bae_model, mode="mu", sigma_train="separate",
                      min_lr_range=0.0000001, max_lr_range=10,
                      reset_params=False, plot=True, verbose=True, save_mecha="copy", run_full=False):
    #helper function
    def round_sig(x, sig=2):
        return round(x, sig-int(floor(log10(abs(x))))-1)

    #get number of iterations for a half cycle based on train loader
    total_iterations = len(train_loader)
    half_iterations = int(total_iterations/2)

    #save temporary model state
    #depending on chosen mechanism
    if save_mecha == "file":
        bae_model.save_model_state()
    elif save_mecha == "copy":
        temp_autoencoder = copy.deepcopy(bae_model.autoencoder)

    #reset it before anything
    if reset_params:
        bae_model.reset_parameters()

    bae_model.scheduler_enabled = False
    #learning range list
    lr_list = []
    train_batch_number = len(train_loader) #num iterations
    for i in range(train_batch_number):
        q= (max_lr_range/min_lr_range)**(1/train_batch_number)
        lr_i = min_lr_range*(q**i)
        lr_list.append(lr_i)

    #calculate smoothened loss
    window_size = 10

    #forward propagate model to get loss vs learning rate
    sigma_train = sigma_train
    mode = mode
    loss_list = []
    current_minimum_loss = 0
    smoothen_loss_list = []

    if verbose:
        print("Starting auto learning rate range finder")

    for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
        bae_model.learning_rate = lr_list[batch_idx]
        bae_model.learning_rate_sig = lr_list[batch_idx]
        bae_model.set_optimisers(bae_model.autoencoder, mode=mode,sigma_train=sigma_train)
        loss = bae_model.fit_one(x=data,y=data, mode=mode)
        loss_list.append(loss)
        if (batch_idx+1)>=window_size:
            #first time, fill up with mean
            if len(smoothen_loss_list) == 0:
                smoothen_loss_list.append(np.mean(copy.copy(loss_list[0:window_size])))
                current_minimum_loss = copy.copy(smoothen_loss_list[0])
            else:
            #calculate exponential ma
                k = 2/(window_size+1)
                smoothen_loss = (loss * k) + smoothen_loss_list[-1]*(1-k)
                if smoothen_loss <= current_minimum_loss:
                    current_minimum_loss = smoothen_loss

                if verbose:
                    print("LRTest-Loss:"+str(smoothen_loss))

                #break if loss is nan
                if np.isnan(smoothen_loss):
                    break

                #append to list
                smoothen_loss_list.append(smoothen_loss)

                #stopping criteria
                if run_full == False:
                    if ((np.abs(smoothen_loss) >= np.abs(current_minimum_loss)*4) or (smoothen_loss>loss_list[0]*1.25) and (batch_idx+1)>=(window_size+10)):
                        break

        #prevent nan
        if loss >= 1000:
            break

    smoothen_loss_list_scaled = (np.array(smoothen_loss_list)-np.min(smoothen_loss_list))/(smoothen_loss_list[0]-np.min(smoothen_loss_list))
    smoothen_loss_list_scaled = np.clip((smoothen_loss_list_scaled),a_min=-100,a_max=2)
    lr_list_plot = (lr_list)[window_size-1:(len(smoothen_loss_list_scaled)+window_size-1)]

    #fit gaussian process to the loss/lr to get a smoothen shape
    X, y = np.array(lr_list_plot).reshape(-1,1), np.array(smoothen_loss_list_scaled).reshape(-1,1)
    X_log10 = np.log10(X)
    kernel = RBF(10, (0.5, 10))
    gpr = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(X_log10,y)
    gpr.score(X_log10,y)

    gp_mean, gp_sigma = gpr.predict(X_log10, return_std=True)
    gp_mean = gp_mean.flatten()
    gp_sigma = gp_sigma.flatten()
    negative_peaks, _ = find_peaks(-gp_mean)

    #get minimum lr
    residuals = np.abs(gp_mean[negative_peaks] - gp_mean[negative_peaks].min())
    minimum_loss_arg = np.argwhere(residuals<=0.05).flatten()[0]
    minimum_loss_arg = negative_peaks[minimum_loss_arg]
    minimum_loss = gp_mean[minimum_loss_arg]
    maximum_lr = lr_list[minimum_loss_arg+(window_size-1)]
    minimum_lr = lr_list[np.argwhere(gp_mean<=0.9)[0][0]+(window_size-1)]/2

    #round up to 3 significant figures
    maximum_lr = round_sig(maximum_lr,3)
    minimum_lr = round_sig(minimum_lr,3)
    min_max_lr_text = "Min lr:{} , Max lr: {}".format(minimum_lr,maximum_lr)

    if verbose:
        print(min_max_lr_text)
    if plot:
        plot_learning_rate_finder(X,y,gp_mean,negative_peaks, minimum_lr,maximum_lr)

    #reset the model again after training
    if reset_params:
        bae_model.reset_parameters()

    #set parameters necessary for the scheduler
    bae_model.init_scheduler(half_iterations,minimum_lr,maximum_lr)
    bae_model.scheduler_enabled = True

    #load model state
    if save_mecha == "file":
        bae_model.load_model_state()
    if save_mecha == "copy":
        bae_model.autoencoder = temp_autoencoder
    return minimum_lr, maximum_lr, half_iterations
