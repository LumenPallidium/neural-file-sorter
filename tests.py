import torch


from options.options import Options
import util.dataloader as dl
from models.visual_models import *
import models.audio_models
from torchsummary import summary

# this is a file i'm using to test, changes feel free to ignore


opt = Options()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VisAutoEncoder(opt.visual_aa,
                           **opt.visual_aa_args)

model = model.to(device)

summary(model, (3, 512, 512))