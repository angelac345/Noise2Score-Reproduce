import torch 
import torch.nn as nn 

class Model(nn.Module): 
    
    def __init__(self, in_channel, out_channel): 
        super(Model, self).__init__() 
        
        self.pool1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=48, kernel_size=3, padding=1), # enc conv 0
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, padding=1), # enc conv 1
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2)
        )

        self.pool2 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, padding=1), # enc conv 2
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2)
        )
        self.pool3 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, padding=1), # enc conv 3
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2)
        )
        self.pool4 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, padding=1), # enc conv 4
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2)
        )
        self.pool5 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, padding=1), # enc conv 5
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2), 
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, padding=1), 
            nn.LeakyReLU(negative_slope=0.1),
        )

        self.upsample5 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, padding=1), # enc conv 6
            nn.LeakyReLU(0.1), 
            nn.Upsample(scale_factor=(2,2))                                         # upsample 5
            # nn.ConvTranspose2d(48, 48, 3, stride=2, padding=1, output_padding=1)
        )

        self.upsample4 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding=1), # dec conv 5A
            nn.LeakyReLU(0.1), 
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding=1), # dec conv 5B
            nn.LeakyReLU(0.1), 
            nn.Upsample(scale_factor=(2,2)),                                        # upsample 4
            # nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1)
        )
        self.upsample3 = nn.Sequential(
            nn.Conv2d(in_channels=144, out_channels=96, kernel_size=3, padding=1), # dec conv 4A
            nn.LeakyReLU(0.1), 
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding=1),   # dec conv 4B
            nn.LeakyReLU(0.1), 
            nn.Upsample(scale_factor=(2,2)),                                        # upsample 3
            # nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1)
        )
        self.upsample2 = nn.Sequential(
            nn.Conv2d(in_channels=144, out_channels=96, kernel_size=3, padding=1), # dec conv 3A
            nn.LeakyReLU(0.1), 
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding=1),   # dec conv 3B
            nn.LeakyReLU(0.1), 
            nn.Upsample(scale_factor=(2,2)),                                        # upsample 2
            # nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1)
        )
        self.upsample1 = nn.Sequential(
            nn.Conv2d(in_channels=144, out_channels=96, kernel_size=3, padding=1), # dec conv 2A
            nn.LeakyReLU(0.1), 
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding=1),   # dec conv 2B
            nn.LeakyReLU(0.1), 
            nn.Upsample(scale_factor=(2,2)),                                        # upsample 1
            # nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1)
        )

        self.dec_conv = nn.Sequential(
            nn.Conv2d(in_channels=96+in_channel, out_channels=64, kernel_size=3, padding=1), # dec conv 1A
            nn.LeakyReLU(0.1), 
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),              # dec conv 1B
            nn.LeakyReLU(0.1), 
            nn.Conv2d(in_channels=32, out_channels=out_channel, kernel_size=3, padding=1), # dec conv 1C
            nn.ReLU()
        )

    def forward(self, x): 
        # breakpoint()
        p1 = self.pool1(x) 
        p2 = self.pool2(p1) 
        p3 = self.pool3(p2) 
        p4 = self.pool4(p3) 
        p5 = self.pool5(p4)
        
        up5 = self.upsample5(p5) 
        up4 = self.upsample4(torch.cat((up5, p4), dim=1))
        up3 = self.upsample3(torch.cat((up4, p3), dim=1))
        up2 = self.upsample2(torch.cat((up3, p2), dim=1))
        up1 = self.upsample1(torch.cat((up2, p1), dim=1)) 

        out = self.dec_conv(torch.cat((up1, x), dim=1))
        return out

    