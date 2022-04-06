from baetorch.baetorch.lr_range_finder import run_auto_lr_range_v4
import numpy as np
from baetorch.baetorch.util.convert_dataloader import convert_dataloader

# Coalitional BAE
class CoalitionalBAE:
    def __init__(self, bae_class, **params):
        self.bae_class = bae_class
        self.params = params
        self.coalitional_members = []
        self.params["chain_params"][0]["conv_channels"][0] = 1

    def fit(self, x_train, use_auto_lr=True, auto_lr_save_mecha="file", **kwargs):
        # assume sensor data of
        self.num_sensors = x_train.shape[1]
        self.coalitional_members = []

        for sensor_i in range(self.num_sensors):
            new_member = self.bae_class(**self.params)

            subset_data = np.expand_dims(x_train[:, sensor_i], 1)

            x_id_train_loader = convert_dataloader(
                subset_data,
                batch_size=len(subset_data) // 5,
                shuffle=True,
                drop_last=True,
            )

            if use_auto_lr:
                min_lr, max_lr, half_iter = run_auto_lr_range_v4(
                    x_id_train_loader,
                    new_member,
                    window_size=1,
                    num_epochs=10,
                    run_full=False,
                    plot=False,
                    verbose=False,
                    save_mecha=auto_lr_save_mecha,
                )

            new_member.fit(x_id_train_loader, **kwargs)
            self.coalitional_members.append(new_member)

    def predict_nll(self, x_test):
        predictions = []
        for member_i, member in enumerate(self.coalitional_members):
            predict_res = member.predict(
                np.expand_dims(x_test[:, member_i], 1), select_keys=["nll"]
            )["nll"]

            predictions.append(predict_res)
        return np.moveaxis(np.array(predictions).sum(3), 0, 2)
