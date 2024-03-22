import torch

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from .data_preprocessor import load_data, make_maps, get_encoder_decoder


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

        context = self.en_data[index]
        target_in = self.por_data[index][:-1]
        target_out = self.por_data[index][1:]

        return context, target_in, target_out

    def get_max_lengh(self):
        data_list = []
        max_length_context = float("-inf")
        for i in range(self.__len__()):
            context = torch.tensor(list(map(self.en_encoder, self.en_data[i].split())))
            data_list.append(context)
            if max_length_context < len(context):
                max_length_context = len(context)

        self.en_data = pad_sequence(data_list)
        self.en_data = torch.permute(self.en_data, (1, 0))

        self.en_data_length = max_length_context

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
        self.por_data = torch.permute(self.por_data, (1, 0))
        self.por_data_length = max_length_context
