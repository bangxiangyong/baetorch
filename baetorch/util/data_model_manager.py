import base64
import os
import pandas as pd
import pickle
import random
import hashlib

class DataModelManager:
    def __init__(self, lookup_table_file="data_model_table.csv", lookup_table_folder="data_models", random_seed=1000):
        self.lookup_table_folder = lookup_table_folder
        self.lookup_table_file = os.path.join(self.lookup_table_folder, lookup_table_file)
        self.random_seed = random_seed

    def encode(self, datatype="model", return_pickle=True, return_compact=False, *args, **kwargs):
        if len(args) > 0:
            data_params = datatype + "_" + str(args) + str(kwargs)
        else:
            data_params = datatype + "_" + str(kwargs)
        # encode_data_name = str(base64.b64encode(str(data_params).encode('ascii')))
        # encode_data_name = str(self.cipher_suite.encrypt(str.encode(data_params)))
        # encode_data_name = str(self.cipher_suite.encrypt(base64.b64encode(str(data_params).encode('ascii'))))
        encoded_message = hashlib.sha1(str.encode(data_params))
        encode_data_name = encoded_message.hexdigest()
        encode_data_name = encode_data_name.replace('b', '')
        encode_data_name = encode_data_name.replace("'", '')

        # perform shuffling
        random.seed(self.random_seed)
        encode_data_name = list(encode_data_name)
        random.shuffle(encode_data_name)
        encode_data_name = ''.join(encode_data_name)
        if return_compact:
            return encode_data_name[:10]+encode_data_name[-10:] + (".p" if return_pickle else "")
        else:
            return encode_data_name + (".p" if return_pickle else "")

    def update_csv(self, datatype="model", **kwargs):
        encode_data_name = self.encode(datatype, return_pickle=True, return_compact=False, **kwargs)
        decode_data = datatype + str(kwargs)
        if not os.path.exists(self.lookup_table_folder):
            os.mkdir(self.lookup_table_folder)

        if not os.path.exists(self.lookup_table_file):
            next_file_name = str(1).zfill(5) + '.p'
            new_df = pd.DataFrame(
                {"filename": [next_file_name], "encoded": [encode_data_name], "decoded": [decode_data]})
            new_df.to_csv(self.lookup_table_file, index=False)
        else:
            csv_df = pd.read_csv(self.lookup_table_file)
            current_file_number = int(csv_df['filename'].iloc[-1].split('.')[0])
            next_file_name = str(current_file_number + 1).zfill(5) + '.p'
            new_df = pd.DataFrame(
                {"filename": [next_file_name], "encoded": [encode_data_name], "decoded": [decode_data]})
            csv_df = csv_df.append(new_df)
            csv_df.to_csv(self.lookup_table_file, index=False)

    def load_csv(self):
        if not os.path.exists(self.lookup_table_file):
            print("No csv table exist yet for data model.")
            return -1
        else:
            return pd.read_csv(self.lookup_table_file)

    def get_pickle_name(self, datatype="model", **kwargs):
        csv_df = self.load_csv()
        encode_data_name = self.encode(datatype, return_pickle=True, return_compact=False,  **kwargs)

        try:
            pickled_file_name = csv_df.loc[csv_df['encoded'] == encode_data_name]['filename'].values[0]
            return pickled_file_name
        except Exception as e:
            print(e)
            return -1

    def load_model(self, datatype="model", **kwargs):
        return pickle.load(
            open(os.path.join(self.lookup_table_folder, self.get_pickle_name(datatype, **kwargs)), mode="rb"))

    def save_model(self, model, datatype="model", **kwargs):
        self.update_csv(datatype, **kwargs)
        pickle.dump(model,
                    open(os.path.join(self.lookup_table_folder, self.get_pickle_name(datatype, **kwargs)), mode="wb"))

    def exist_model(self, datatype="model", **kwargs):
        if self.get_pickle_name(datatype, **kwargs) == -1:
            return False
        else:
            return True

    def wrap(self, method, datatype="data", *args, **data_params):
        if self.exist_model(datatype, **data_params):
            print("Data model existed, loading from pickle...")
            x = self.load_model(datatype=datatype, **data_params)
        else:
            x = method(*args, **data_params)
            print("Saving data model...")
            self.save_model(x, datatype=datatype, **data_params)
        return x
