"""Smoke test for the Grad-CAM engine without requiring Hugging Face downloads."""
from pathlib import Path
from types import SimpleNamespace
import sys
import torch
import torch.nn as nn
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path: sys.path.insert(0, str(REPOSITORY_ROOT))
from explainability.gradcam import GradCAM, overlay_heatmap, prepare_image

class ToySwinBackbone(nn.Module):
    def __init__(self, channels=16):
        super().__init__(); self.features=nn.Sequential(nn.Conv2d(3,channels,kernel_size=3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d((7,7))); self.layernorm=nn.LayerNorm(channels)
    def forward(self,x): return self.layernorm(self.features(x).flatten(2).transpose(1,2))
class ToySwinClassifier(nn.Module):
    def __init__(self,num_labels=5,channels=16): super().__init__(); self.swin=ToySwinBackbone(channels); self.classifier=nn.Linear(channels,num_labels)
    def forward(self,x):
        tokens=self.swin(x); return SimpleNamespace(logits=self.classifier(tokens.mean(dim=1)))

def run_smoke_test(image_path,output_path):
    torch.manual_seed(7); model=ToySwinClassifier().eval(); display_image,input_tensor=prepare_image(image_path,image_size=224)
    with GradCAM(model,model.swin.layernorm) as gradcam: result=gradcam.generate(input_tensor)
    cam=result["cam"]; assert cam.shape==(224,224)
    overlay=overlay_heatmap(display_image,cam); output_path=Path(output_path); output_path.parent.mkdir(parents=True,exist_ok=True); overlay.save(output_path); return result,output_path
