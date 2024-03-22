import torch
import torch.nn as nn

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from data_preprocessor import load_data, make_maps, get_encoder_decoder


class EnToPorDataset(Dataset):
    def __init__(self, dataset_path):
        super().__init__()
        self.por_data, self.en_data = load_data(dataset_path)

        assert len(self.por_data) == len(self.en_data), "Pairs mismatch !"

        self.por_stoi, self.por_itos = make_maps(self.por_data)
        self.en_stoi, self.en_itos = make_maps(self.en_data)

        self.por_encoder, self.por_decoder = get_encoder_decoder(
            self.por_stoi, self.por_itos
        )

        self.en_encoder, self.en_decoder = get_encoder_decoder(
            self.en_stoi, self.en_itos
        )

        self.get_max_lengh()

    def __len__(self):
        return len(self.por_data)

    def __getitem__(self, index):

        context = torch.tensor(list(map(self.en_encoder, self.en_data[index].split())))
        target = torch.tensor(list(map(self.por_encoder, self.por_data[index].split())))

        return context, target

    def get_max_lengh(self):
        data_list = []
        max_length_context = float("-inf")
        for i in range(self.__len__()):
            context = torch.tensor(list(map(self.en_encoder, self.en_data[i].split())))
            data_list.append(context)
            if max_length_context < len(context):
                max_length_context = len(context)

        self.en_data = pad_sequence(data_list)

        data_list = []
        max_length_context = float("-inf")
        for i in range(self.__len__()):
            context = torch.tensor(
                list(map(self.por_encoder, self.por_data[i].split()))
            )
            data_list.append(context)
            if max_length_context < len(context):
                max_length_context = len(context)

        self.por_data = pad_sequence(data_list)


dt = EnToPorDataset("data/por-eng/por.txt")
