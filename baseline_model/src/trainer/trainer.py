import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim import AdamW
from tqdm import tqdm
from torchmetrics.image import StructuralSimilarityIndexMeasure
import matplotlib.pyplot as plt

class Trainer:
    """
    A comprehensive trainer class for PyTorch models, encapsulating training,
    evaluation, and testing workflows.

    This class provides functionalities to train a given PyTorch model,
    evaluate its performance on a validation set using common image metrics
    (PSNR, SSIM, MAE), and test its final performance. It handles device
    management (CPU/GPU), default optimizer and loss function, and progress
    visualization.
    """
    def __init__(self, model:nn.Module) -> None:
        """
        Initializes the Trainer with a specified PyTorch model.

        Args:
            model (nn.Module): The PyTorch model to be trained.
        """
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)
    
    def train(self, dataloader_train:DataLoader, num_epoch:int = 100, lr:float = 1e-3, optimizer: Optimizer = None, loss_function: nn.Module = None)->None:
        """
        Trains the model using the provided training DataLoader.

        Args:
            dataloader_train (DataLoader): The DataLoader containing the training data.
            num_epoch (int, optional): The number of training epochs. Default is 100.
            lr (float, optional): The learning rate for the optimizer. Default is 1e-3.
            optimizer (Optimizer, optional): The optimizer to use for training. If None,
                                             AdamW with the specified learning rate is used.
            loss_function (nn.Module, optional): The loss function to use. If None,
                                                 L1Loss (MAE) is used.

        Returns:
            nn.Module: The trained model.
        """
        if optimizer is None:
            optimizer = AdamW(self.model.parameters(), lr=lr)
        if loss_function is None:
            loss_function = nn.L1Loss()
        
        self.model = self.model.to(self.device)
        self.model = self.model.train()
        losses = list()

        for epoch in tqdm(range(num_epoch)):    
            torch.cuda.empty_cache()

            for data in tqdm(dataloader_train):
                optimizer.zero_grad()
                x, y = data
                x, y = x.to(self.device).to(torch.float32), y.to(self.device).to(torch.float32)
                preds = self.model(x)
                loss = loss_function(preds, y)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
        
        plt.plot(range(len(losses)), losses)
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.show()
        return self.model


    def evaluate(self, dataloader_val:DataLoader):
        """
        Evaluates the model on the provided validation DataLoader.

        Calculates and plots PSNR, SSIM, and MAE metrics for the validation set.

        Args:
            dataloader_val (DataLoader): The DataLoader containing the validation data.
        """
        self.model = self.model.to(self.device)
        self.model = self.model.eval()
        
        psnr_metric = list()
        ssim_metric = list()
        mae_metric = list()

        ssim = StructuralSimilarityIndexMeasure()

        with torch.no_grad():
            torch.cuda.empty_cache()
            for data in dataloader_val:
                x, y = data
                x, y = x.to(self.device).to(torch.float32), y.to(self.device).to(torch.float32)
                preds = self.model(x)
                ssim_metric.append(ssim(preds, y).item())
                psnr_metric.append(self._calc_psnr(preds, y))
                mae_metric.append(F.l1_loss(preds, y))
        fig, ax =  plt.subplots(1, 3, figsize=(15, 5))
        ax[0].plot(psnr_metric)
        ax[0].set_title('PSNR metric')
        ax[1].plot(ssim_metric)
        ax[1].set_title('SSIM metric')
        ax[2].plot(mae_metric)
        ax[2].set_title('MAE metric')
        plt.show()

    def test(self, dataloader_test:DataLoader)->None:
        """
        Tests the model on the provided test DataLoader and prints the average metrics.

        Calculates and prints the mean PSNR, SSIM, and MAE over the entire test set.

        Args:
            dataloader_test (DataLoader): The DataLoader containing the test data.
        """
        self.model = self.model.to(self.device)
        self.model = self.model.eval()

        ssim_metric = list()
        psnr_metric = list()
        mae_metric = list()

        ssim = StructuralSimilarityIndexMeasure()
        
        with torch.no_grad():
            for data in dataloader_test:
                x, y = data
                x, y = x.to(self.device).to(torch.float32), y.to(self.device).to(torch.float32)
                preds = self.model(x)
                ssim_metric.append(ssim(preds, y).item())
                psnr_metric.append(self._calc_psnr(preds, y))
                mae_metric.append(F.l1_loss(preds, y).item())
            
        if len(ssim_metric) == 1:
            print(f'SSIM metric is {ssim_metric[-1]}')
            print('-'*10)
            print(f'PSNR metric is {psnr_metric[-1]}')
            print('-' * 10)
            print(f'MAE metric is {mae_metric[-1]}')
        else:
            print(f'SSIM metric is {torch.tensor(ssim_metric).mean().item()}')
            print('-'*10)
            print(f'PSNR metric is {torch.tensor(psnr_metric).mean().item()}')
            print('-' * 10)
            print(f'MAE metric is {torch.tensor(mae_metric).mean().item()}')


    def _calc_psnr(self, pred:torch.Tensor, label:torch.Tensor, scaled:bool = True, max_val: int = None)->float:
        """
        Calculates the Peak Signal-to-Noise Ratio (PSNR) between predicted and label tensors.

        Args:
            pred (torch.Tensor): The predicted tensor.
            label (torch.Tensor): The ground truth label tensor.
            scaled (bool, optional): If True, assumes pixel values are scaled to [0, 1]
                                     and uses a max_val of 1. Otherwise, assumes
                                     a max_val of 100. Defaults to True.

        Returns:
            float: The calculated PSNR value.
        """
        if scaled:
            max_val = 1
        else:
            max_val = 100
        mse = F.mse_loss(pred, label)
        return 20*torch.log10(max_val / torch.sqrt(mse)).item()
            
