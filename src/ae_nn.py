import torch
from torch import nn
import torch.nn.functional as F

from torch.nn.utils.parametrizations import weight_norm
import math

# === General Blocks ===

def WeightNormConv2d(*args, **kwargs):
    return weight_norm(nn.Conv2d(*args, **kwargs))

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()

        n_grps = (2 * ch) // 16 # 16 grps
        
        self.conv1 = WeightNormConv2d(ch, ch, 1, 1, 0)
        self.conv2 = WeightNormConv2d(ch, ch, 3, 1, 1, groups=n_grps)
        self.conv3 = WeightNormConv2d(ch, ch, 1, 1, 0, bias=False)

        self.act1 = nn.LeakyReLU(inplace=True)
        self.act2 = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        h = self.conv1(x)
        h = self.act1(h)
        h = self.conv2(h)
        h = self.act2(h)
        h = self.conv3(h)
        return x + h
    
# === Encoder ===

class LandscapeToSquare(nn.Module):
    # Strict assumption of 360p
    def __init__(self, ch_in, ch_out):
        super().__init__()

        self.proj = WeightNormConv2d(ch_in, ch_out, 3, 3, 1)
    
    def forward(self, x):
        x = F.interpolate(x, (512, 512), mode = 'bicubic')
        x = self.proj(x)
        return x

class Downsample(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()

        self.proj = WeightNormConv2d(ch_in, ch_out, 1, 1, 0, bias = False)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor = 0.5, mode = 'bicubic')
        x = self.proj(x)
        return x

class DownBlock(nn.Module):
    def __init__(self, ch_in, ch_out, num_res=1):
        super().__init__()

        self.down = Downsample(ch_in, ch_out)
        blocks = []
        for _ in range(num_res):
            blocks.append(ResBlock(ch_in))
        self.blocks = nn.ModuleList(blocks)
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.down(x)
        return x

class SpaceToChannel(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        
        self.proj = WeightNormConv2d(ch_in, ch_out // 4, 3, 1, 1)
    
    def forward(self, x):
        x = self.proj(x)
        x = F.pixel_unshuffle(x, 2).contiguous()
        return x

class ChannelAverage(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()

        self.proj = WeightNormConv2d(ch_in, ch_out, 3, 1, 1)
        self.grps = ch_in // ch_out
        self.scale = (self.grps) ** 0.5
    
    def forward(self, x):
        res = x.clone()
        x = self.proj(x.contiguous()) # [b, ch_out, h, w]

        # Residual goes through channel avg
        res = res.view(res.shape[0], self.grps, res.shape[1] // self.grps, res.shape[2], res.shape[3]).contiguous()
        res = res.mean(dim=1) * self.scale # [b, ch_out, h, w]
        
        return res + x

# === Decoder ===

class SquareToLandscape(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        
        self.proj = WeightNormConv2d(ch_in, ch_out, 3, 3, 1)
    
    def forward(self, x):
        x = self.proj(x) # TODO This ordering is wrong for both
        x = F.interpolate(x, (360, 640), mode = 'bicubic') 
        return x

class Upsample(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()

        self.proj = WeightNormConv2d(ch_in, ch_out, 1, 1, 0, bias = False)
    
    def forward(self, x):
        x = self.proj(x)
        x = F.interpolate(x, scale_factor = 2.0, mode = 'bicubic')
        return x

class UpBlock(nn.Module):
    def __init__(self, ch_in, ch_out, num_res=1):
        super().__init__()
        
        self.up = Upsample(ch_in, ch_out)
        blocks = []
        for _ in range(num_res):
            blocks.append(ResBlock(ch_out))
        self.blocks = nn.ModuleList(blocks)
    
    def forward(self, x):
        x = self.up(x)
        for block in self.blocks:
            x = block(x)
        return x

class ChannelToSpace(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        
        self.proj = WeightNormConv2d(ch_in, ch_out * 4, 3, 1, 1)
    
    def forward(self, x):
        x = self.proj(x)
        x = F.pixel_shuffle(x, 2).contiguous()
        return x

class ChannelDuplication(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()

        self.proj = WeightNormConv2d(ch_in, ch_out, 3, 1, 1)
        self.reps = ch_out // ch_in
        self.scale = (self.reps) ** -0.5

    def forward(self, x):
        res = x.clone()
        x = self.proj(x.contiguous())

        res = res.repeat_interleave(self.reps, dim = 1).contiguous() * self.scale

        return res + x

# === Main AE ===

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.conv_in = LandscapeToSquare(config.channels, config.ch_0)

        blocks = []
        residuals = []

        ch = config.ch_0
        for block_count in config.encoder_blocks_per_stage:
            next_ch = min(ch*2, config.ch_max)

            blocks.append(DownBlock(ch, next_ch, block_count))
            residuals.append(SpaceToChannel(next_ch, next_ch))
            
            ch =  next_ch
    
        self.blocks = nn.ModuleList(blocks)
        self.residuals = nn.ModuleList(residuals)
        self.conv_out = ChannelAverage(ch, config.latent_channels)

    def forward(self, x):
        x = self.conv_in(x)
        for block, residual in zip(self.blocks, self.residuals):
            x = block(x) + residual(x)
        return self.conv_out(x)

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.conv_in = ChannelDuplication(config.latent_channels, config.ch_max)

        blocks = []
        residuals = []

        ch = config.ch_0
        for block_count in reversed(config.decoder_blocks_per_stage):
            next_ch = min(ch*2, config.ch_max)

            blocks.append(UpBlock(next_ch, ch, block_count))
            residuals.append(ChannelToSpace(next_ch, next_ch))

            ch = next_ch
        
        self.blocks = nn.ModuleList(reversed(blocks))
        self.residuals = nn.ModuleList(reversed(residuals))
        
        self.act_out = nn.SiLU()
        self.conv_out = SquareToLandscape(config.ch_0, config.channels)
    
    def forward(self, x):
        x = self.conv_in(x)
        for block, residual in zip(self.blocks, self.residuals):
            x = block(x) + residual(x)
        x = self.act_out(x)
        return self.conv_out(x)

class AutoEncoder(nn.Module):
    def __init__(self, encoder_config, decoder_config=None):
        super().__init__()

        if decoder_config is None:
            decoder_config = encoder_config

        self.encoder = Encoder(encoder_config)
        self.decoder = Decoder(decoder_config)
    
    def forward(self, x):
        return self.decoder(self.encoder(x))
