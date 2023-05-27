'''plot charts functions
'''

import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
import contextlib

from typing import List
import random
import pathlib
import glob

import torch
import torchvision

from torchmetrics import ConfusionMatrix
from torchvision.utils import make_grid

from mlxtend.plotting import plot_confusion_matrix
from torchmetrics import ConfusionMatrix

from utils.useful_funcs import predict_test


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
        
        
def plot_loss(loss_dis: list, loss_gen: list) -> None:
    font_s = 12
    plt.figure(figsize=(8,5))
    
    plt.title('Значения функции потерь на \nдискриминаторе и генераторе\n', fontsize=font_s+4)
    
    plt.plot(loss_gen, label='Generator', alpha=0.7)
    plt.plot(loss_dis, label='Discriminator', alpha=0.7)
    
    plt.xlabel('epochs', fontsize=font_s)
    plt.ylabel('loss', fontsize=font_s)
    
    plt.legend(loc='upper right')
    
    plt.grid()
    plt.show()
    
    
def save_image(epoch: int, 
               title: str,
               images: torch.Tensor, 
               n_col: int,
               path: str = 'visualization/') -> None:
    
    font_s = 12
    
    # PyTorch default shape is [C, H, W] but Matplotlib is [H, W, C]
    image_grid = images.permute(1, 2, 0)

    plt.figure(figsize=(8,8), facecolor='white')
    plt.imshow(image_grid)
    plt.title(f'Epoch {epoch}', fontsize=font_s+8)
    
    plt.xticks([])
    plt.yticks([])
    
    plt.savefig(f'{path}images/{title}_{epoch:05d}.jpg')
    plt.close()
    
    
def create_gif(title: str, path: str = 'visualization/') -> None:
    
    images_path = f'{path}images/{title}_*.jpg'
    gif_path = f'{path}gifs/{title}.gif'

    with contextlib.ExitStack() as stack:
        images_stack = (stack.enter_context(Image.open(image))
                        for image in sorted(glob.glob(images_path)))

        image = next(images_stack)

        image.save(fp=gif_path, 
                   format='GIF', 
                   append_images=images_stack,
                   save_all=True, 
                   duration=250, 
                   disposal=2,
                   loop=0)
        
        
def plot_comparison_real_fake(dataloader: torch.utils.data.dataloader.DataLoader,
                              fake_images: list,
                              device: str) -> None:
    
    font_s=12
    
    real_batch = next(iter(dataloader))

    plt.figure(figsize=(16,15))
    plt.subplot(1,2,1)
    plt.axis('off')
    plt.title('Real Images\n', fontsize=font_s+4)
    plt.imshow(np.transpose(
        make_grid(real_batch[0].to(device), 4, padding=2, normalize=True).cpu(),(1,2,0))
    )

    plt.subplot(1,2,2)
    plt.axis('off')
    plt.title('Fake Images\n', fontsize=font_s+4)
    plt.imshow(np.transpose(fake_images[-1],(1,2,0)))
    plt.show()
    
    
def display_confusion_matrix(predictions: List[int],
                             data: torch.utils.data.dataloader.DataLoader,
                             class_names: List[str]) -> None:
    
    confmat = ConfusionMatrix(num_classes=len(data.classes), task='multiclass')
    confmat_tensor = confmat(
        preds=torch.IntTensor(predictions), 
        target=torch.IntTensor(data.targets)
    )
    
    fig, ax = plot_confusion_matrix(
        conf_mat=confmat_tensor.numpy(),
        class_names=class_names,
        figsize=(10,10)
    )
    
    
def plot_image_predictions(model: torch.nn.Module, 
                           device: torch.device,
                           target_dir: str,
                           class_names: List[str] = None, 
                           transform: torchvision.transforms.transforms.Compose = None,
                           n_images: int = 6,
                           depth: str = '*/*') -> None:
    
    font_s = 12
    cols = 8
    
    if n_images % cols != 0:
        if n_images < cols:
            n_images = cols
        else:
            n_images = (n_images // cols) * cols 
            
        print(f'Для корректной взуализации значение n_images было установлено как кратное {cols}')
        
    image_paths = list(pathlib.Path(target_dir).glob(f'{depth}.jpg'))
    images = random.sample(image_paths, n_images)
    
    true_labels = [img.parent.stem for img in images]
    
    fig = plt.figure(figsize=(16, n_images / 2))
        
    cols = 8
    rows = n_images // cols + int((n_images % cols) / 10)
    
    if rows == 0:
        rows = 1
    
    for i, image in enumerate(images):
        fig.add_subplot(rows, cols, i+1)
        
        img, probs, label = predict_test(image, model, device, transform)
        
        # PyTorch default shape is [C, H, W] but Matplotlib is [H, W, C]
        img_permute = img.squeeze().permute(1, 2, 0)
        
        plt.imshow(img_permute)
        plt.axis('off')
        
        plt.title(f'Прогноз: {class_names[label.cpu()]} \nКласс: {true_labels[i]} \n\nВероятность: {probs.max().cpu():.3f}', 
                  fontsize=font_s-2)
        
        plt.tight_layout()