'''plot charts functions
'''

import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

from typing import List
import random
import pathlib

import torch
import torchvision

from torchmetrics import ConfusionMatrix


def plot_random_image(target_dir: str,
                      seed: int = None,
                      depth: str = '*/*/*') -> None:
    
    random.seed(seed)

    image_paths = list(pathlib.Path(target_dir).glob(f'{depth}.jpg'))
    image_random = random.choice(image_paths)
    image_class = image_random.parent.stem
    
    img = Image.open(image_random)
    
    print('Путь к изображению:', image_random)
    print('Класс изображения:', image_class)
    print(f'Высота: {img.height} | Ширина: {img.width}')
    
    img = np.array(img)
    plt.imshow(img)
    
    print(f'Размерность изображения {img.shape} -> [height, width, color_channels]')


def plot_transformed_images(transforms: torchvision.transforms.transforms.Compose, 
                            n_images: int, 
                            target_dir: str or List[str],
                            seed: int = None,
                            depth: str = '*/*/*') -> None:
    
    font_s = 12
    random.seed(seed)
    
    image_paths = list(pathlib.Path(target_dir).glob(f'{depth}.jpg')) 
    image_random = random.sample(image_paths, k=n_images)
    
    for image in image_random:
        with Image.open(image) as file:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(file)
            ax[0].set_title(f'Исходный: \n{np.array(file).shape}')
            ax[0].axis('off')

            # PyTorch default shape is [C, H, W] but Matplotlib is [H, W, C]
            image_transformed = transforms(file).permute(1, 2, 0) 
            ax[1].imshow(image_transformed)
            ax[1].set_title(f'Трансформированный: \n{image_transformed.numpy().shape}')
            ax[1].axis('off')
            
        fig.suptitle(f'Класс {image.parent.stem}', fontsize=font_s+4)


def print_image_data(data: torchvision.datasets.folder.ImageFolder,
                     index: int) -> None:
    
    img, label = data[index][0], data[index][1]

    print(f'\nImage tensor:\n{img}')
    print(f'Image shape: {img.shape}')
    print(f'Image datatype: {img.dtype}')
    print(f'Image label: {label} ({data.classes[label]})')
    print(f'Label datatype: {type(label)}')


def plot_permuted_image(data: torchvision.datasets.folder.ImageFolder,
                        index: int,
                        target_dir: str,
                        depth: str = '*/*/*') -> None:

    font_s = 12
    img, label = data[index][0], data[index][1]

    image_paths = list(pathlib.Path(target_dir).glob(f'{depth}.jpg'))
    image_random = random.choice(image_paths)   

    # PyTorch default shape is [C, H, W] but Matplotlib is [H, W, C]
    img_permute = img.permute(1, 2, 0)

    print(f'Исходная размерность: {img.shape} -> [color_channels, height, width]')
    print(f'Изменённая размерность: {img_permute.shape} -> [height, width, color_channels]')
    
    plt.imshow(img_permute)
    plt.title(data.classes[label], fontsize=font_s+4)


def plot_random_dataset_images(data: torch.utils.data.dataset.Dataset,
                               n_images: int = 8,
                               seed: int = None) -> None:
    
    font_s = 12
    random.seed(seed)
    
    cols = 8
    
    if n_images % cols != 0:
        if n_images < cols:
            n_images = cols
        else:
            n_images = (n_images // cols) * cols 
            
        print('Для корректной взуализации значение n_images было установлено как кратное 8')
    
    fig = plt.figure(figsize=(16, n_images / 2))
    
    image_idxs = random.sample(range(len(data)), k=n_images)
        
    cols = 8
    rows = n_images // cols + int((n_images % cols) / 10)

    if rows == 0:
        rows = 1
    
    for i, image in enumerate(image_idxs):
        fig.add_subplot(rows, cols, i+1)
        
        img, label = data[image][0], data[image][1]
        
        # PyTorch default shape is [C, H, W] but Matplotlib is [H, W, C]
        img_permute = img.permute(1, 2, 0)
        
        plt.imshow(img_permute)
        plt.axis('off')
        
        plt.title(f'Класс: {data.classes[label]} \nРазмерность: \n{img_permute.shape}', 
                  fontsize=font_s-2)
        
        plt.tight_layout()


def plot_loss_curves(loss: List[float],
                     accuracy: List[float],
                     loss_val: List[float] = None,
                     accuracy_val: List[float] = None) -> None:
    
    font_s = 12
    titles = ['Loss', 'Accuracy']
    
    epochs = range(len(loss))
    n_cols = len(titles)
    
    fig, ax = plt.subplots(1, n_cols, figsize=(16,5))
    
    for i in range(n_cols):
        key = n_cols * i
        
        ax[i].plot(epochs, [loss, accuracy][i], label=f'{titles[i].lower()}_train')
        
        if loss_val is not None and accuracy_val is not None:
            ax[i].plot(epochs, [loss_val, accuracy_val][i], label=f'{titles[i].lower()}_val')
        
        ax[i].set_title(titles[i], fontsize=font_s+4)
        ax[i].set_xlabel('Epochs', fontsize=font_s)
        
        ax[i].legend(loc='upper right')
        ax[i].grid()
        
        i += 1


def plot_loss_curves_comparison(*args) -> None:
    font_s = 12
    titles = ['Loss', 'Accuracy']
    
    models = {}
    
    for i in range(len(args)):
        models[f'model_{i}'] = args[i]
    
    epochs = range(len(list(models.values())[0]))
    n_cols = len(titles)
    
    fig, ax = plt.subplots(1, n_cols, figsize=(16,5))
    
    for i in range(n_cols):
        key = n_cols * i
        
        for j in range(len(models)):
            ax[i].plot(epochs, models[list(models.keys())[j]][:, i], label=list(models.keys())[j])
        
        ax[i].set_title(titles[i], fontsize=font_s+4)
        ax[i].set_xlabel('Epochs', fontsize=font_s)
        
        ax[i].legend(loc='upper right')
        ax[i].grid()
        
        i += 1