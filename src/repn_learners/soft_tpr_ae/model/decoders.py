import torch 
import torch.nn as nn 

from src.shared.components import View

class VQ_VAE_Decoder(nn.Module): 
    def __init__(self, latent_dim: int, nc: int=3) -> None: 
        super().__init__() 
        self.kwargs_for_loading = {'latent_dim': latent_dim, 'nc': nc}
        self.net = nn.Sequential(
            View(-1, latent_dim, 1, 1),
            #ResBlock(latent_dim, latent_dim),
            #nn.BatchNorm2d(latent_dim),
            #ResBlock(latent_dim, latent_dim),
            nn.ConvTranspose2d(latent_dim, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, nc, kernel_size=4, stride=2, padding=1)
        )
        
    def forward(self, z: torch.Tensor) -> torch.Tensor: 
        return self.net(z) 
    
class VQ_VAE_Decoder2(nn.Module): 
    def __init__(self, latent_dim: int, nc: int=3) -> None: 
        super().__init__() 
        self.kwargs_for_loading = {'latent_dim': latent_dim, 'nc': nc}
        self.net = nn.Sequential(
            View(-1, latent_dim, 1, 1),
            #ResBlock(latent_dim, latent_dim),
            #nn.BatchNorm2d(latent_dim),
            #ResBlock(latent_dim, latent_dim),
            nn.ConvTranspose2d(latent_dim, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, nc, kernel_size=4, stride=2, padding=1)
        )
    def forward(self, z: torch.Tensor) -> torch.Tensor: 
        return self.net(z) 

class BaseDecoder2(nn.Module): 
    def __init__(self, latent_dim: int, nc: int=3) -> None: 
        super().__init__() 
        self.kwargs_for_loading = {'latent_dim': latent_dim, 'nc': nc}
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(), 
            nn.Linear(256, 256*2*2),
            nn.ReLU(), 
            View(-1, 256, 2, 2), 
            nn.ConvTranspose2d(256, 256, 4, 2, 1),
            nn.ReLU(), 
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(), 
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(), 
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, nc, 4, 2, 1)
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor: 
        return self.net(z)

class BaseDecoder(nn.Module): 
    def __init__(self, latent_dim: int, nc: int=3) -> None: 
        super().__init__() 
        self.kwargs_for_loading = {'latent_dim': latent_dim, 'nc': nc}
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256), 
            View(-1, 256, 1, 1),
            nn.ReLU(), 
            nn.ConvTranspose2d(256, 64, 4),
            nn.ReLU(), 
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU(), 
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(), 
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(), 
            nn.ConvTranspose2d(32, nc, 4, 2, 1)
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor: 
        return self.net(z) 
    
class BaseDecoder3(nn.Module): 
    def __init__(self, latent_dim: int, nc: int=3) -> None: 
        super().__init__() 
        self.net = nn.Sequential(
            View(-1, latent_dim, 1, 1),
            nn.ConvTranspose2d(latent_dim, 256, 4),
            nn.ReLU(), 
            nn.ConvTranspose2d(256, 64, 4, 2, 1),
            nn.ReLU(), 
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(), 
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(), 
            nn.ConvTranspose2d(32, nc, 4, 2, 1)
        )
        self.kwargs_for_loading = {'latent_dim': latent_dim, 'nc': nc}
    
    def forward(self, z: torch.Tensor) -> torch.Tensor: 
        return self.net(z) 
    