import os
import pickle

class DLNNDataLoader(list):
    def __init__(self, dir):
        super().__init__()
        self._dir = dir
        self._games = os.listdir(dir)
    
    def __getitem__(self, idx):
        assert 0 <= idx <= len(self._games)
        game_file = f'{self._dir}/mario-v0-offline-game_{idx}.pkl'
        game_data = pickle.load(open(game_file, "rb"))
        return game_data

    def __len__(self) -> int:
        return self._games.__len__()
