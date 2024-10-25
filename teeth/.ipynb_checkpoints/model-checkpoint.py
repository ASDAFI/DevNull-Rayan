import copy

import torch
import torch.nn as nn



class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.inc = DoubleConv(1, 32)         
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(32, 64)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(64, 128)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(128, 256)
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(256, 512)
        )

        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(256, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(128, 64)

        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(64, 32)

        self.outc = nn.Conv2d(32, 1, kernel_size=1)

    #######DO NOT CHANGE THIS PART########
    def init(self, path="model.pth"):
        self.load_state_dict(torch.load(path, weights_only=True))
    ######################################
    def save(self, path: str):
        torch.save(copy.deepcopy(self.state_dict()), path)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)     
        x2 = self.down1(x1)    
        x3 = self.down2(x2)    
        x4 = self.down3(x3)    
        x5 = self.down4(x4)   

        x = self.up1(x5)       
        x = torch.cat([x, x4], dim=1) 
        x = self.conv_up1(x)

        x = self.up2(x)       
        x = torch.cat([x, x3], dim=1) 
        x = self.conv_up2(x) 

        x = self.up3(x)        
        x = torch.cat([x, x2], dim=1)  
        x = self.conv_up3(x)  

        x = self.up4(x)        
        x = torch.cat([x, x1], dim=1)  
        x = self.conv_up4(x)  

        mask = self.outc(x)   
        return mask

