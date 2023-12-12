import mido 
from mido import MidiTrack
import os

from sklearn import mean_absolute_error

import numpy as np
import matplotlib.pyplot as plt

def plot_and_save_experiment_results(notes, pred_notes_exp,pred_notes_prox,pred_notes_depend, dataset_name) -> None:
    path = f'results/{dataset_name}'
    os.makedirs(path, exist_ok=True)
    plt.figure(figsize=(18,10))
    indices = np.arange(0, len(notes))
    plt.plot(indices, notes, marker='.', label='Ground  truth')
    plt.plot(indices, pred_notes_exp, marker='.', label = 'ExponentialSmoothing')
    plt.plot(indices, pred_notes_prox, marker='.', label='ProximityEstimator')
    plt.plot(indices, pred_notes_depend, marker='.', label='Dependency retrieval')

    plt.grid(True)

    plt.legend()

    plt.ylabel('Note')
    # plt.title('')

    plt.savefig(f'{path}/{dataset_name}_all_ts.png')

    plt.show()

    not_nan_ind = np.where(~np.isnan(pred_notes_depend))[0]

    print('Exponential smoothing: ', mean_absolute_error(notes[5:], pred_notes_exp[5:]))
    print('Proximity Estimator: ', mean_absolute_error(notes[5:], pred_notes_prox[5:]))


    print('Exponential smoothing without nans: ', mean_absolute_error(notes[not_nan_ind], pred_notes_exp[not_nan_ind]))
    print('Proximity Estimator without nans: ', mean_absolute_error(notes[not_nan_ind], pred_notes_prox[not_nan_ind]))
    print('Dependency retrieval without nans: ', mean_absolute_error(notes[not_nan_ind], pred_notes_depend[not_nan_ind]))

    def tmp_func(indices, notes, pred_notes, dataset_name, estimator_name):
        plt.plot(indices, notes, marker='.')
        plt.plot(indices, pred_notes, marker='.')

        plt.ylabel("Note")
        plt.grid(True)

        plt.title(f"{estimator_name} estimation")

        plt.savefig(f"{path}/{dataset_name}_{estimator_name}.png")
        plt.show()

    tmp_func(indices, notes, pred_notes_exp, dataset_name, 'Exponential_smoothing')
    tmp_func(indices, notes, pred_notes_prox, dataset_name, 'Proximity')
    tmp_func(indices, notes, pred_notes_depend, dataset_name, 'Dependency_retrieval')



class AudioFile:
    '''для загрузки более сложных мелодий, где используется несколько инструментов'''

    def __init__(self, author: str, track: str) -> None:

        self.file = mido.MidiFile(os.path.join('data', author, track + '.mid'))
        self.merged_file = mido.merge_tracks(self.file.tracks)


    def extract_notes(self, miditrack: MidiTrack=None) -> list:
        notes = []
        if miditrack is None:
            miditrack = self.merged_file
        for msg in miditrack:
            if msg.type == 'note_on':
                notes.append(msg.note)
        return notes

    def extract_channels(self):
        channels = []
        
        for msg in self.merged_file:
            if msg.type == 'note_on':
                channels.append(msg.channel)
        return channels    
    
    
    def plot(self, instrument: str=None, figsize=(18,10), scatter: bool=True) -> None:

        if instrument is None:
            notes = self.extract_notes()
            channels = self.extract_channels()
            indices = np.arange(0, len(notes))

        plt.figure(figsize=figsize)

        if scatter:
            plt.scatter(indices, notes, c=channels)
        else:
            plt.plot(indices, notes, marker='.')

        plt.grid(True)

        plt.show()




