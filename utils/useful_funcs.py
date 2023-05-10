'''useful functions
'''

import pandas as pd

import torch
import os
from typing import Tuple

from sklearn.metrics import classification_report


def walk_through_dir(target_dir: str) -> None:
    counter, total_dir, total_img = 1, 0, 0
    
    for dirpath, dirnames, filenames in os.walk(target_dir):
        dirpath = dirpath.replace('\\', '/')
        label = dirpath.split('/')[-1]
        
        if label in target_dir:
            space = ''
            
            if dirpath != target_dir:
                total_dir += len(dirnames)
                space = '\n'
                
            print(f'{space}Путь: {dirpath} -> Каталогов: {len(dirnames)}\n')
            
        else:
            space, end = '├── ', ''
            
            if counter == total_dir:
                space, end = '└── ', '\n'
            
            print(f'\t{space}Класс {label} -> Изображений: {len(filenames)}{end}')
            counter += 1
        
        total_img += len(filenames)


def iterate_dataloader(dataloader: torch.utils.data.dataloader.DataLoader,
                       return_iterator: bool = False) -> None or Tuple[torch.Tensor, torch.Tensor]:
    
    imgs, labels = next(iter(dataloader))
    
    print(f'Размерность изображения: {imgs.shape} -> [batch_size, color_channels, height, width]')
    print(f'Размерность класса: {labels.shape}')
    
    if return_iterator:
        return imgs, labels


def check_time(start: float, end: float, device: torch.device = None) -> None:
    total_time = end - start
    print(f'\nTotal train time on {device}: {total_time:.3f} seconds')


def get_classification_report(targets: list,
                              predictions: list,
                              class_names: list) -> pd.DataFrame:
    
    return pd.DataFrame(
        classification_report(
            targets, 
            predictions,
            target_names=class_names,
            output_dict=True
        )).T.sort_values(by=['precision', 'recall', 'f1-score'], ascending=False)