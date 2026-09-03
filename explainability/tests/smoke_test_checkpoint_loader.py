"""Smoke test for the repository checkpoint loader using a local fake HF model."""
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import tempfile
import torch
import torch.nn as nn
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path: sys.path.insert(0, str(REPOSITORY_ROOT))
from explainability.model_loader import load_swin_classifier

class FakeHFClassifier(nn.Module):
    def __init__(self):
        super().__init__(); self.swin=nn.Module(); self.swin.layernorm=nn.LayerNorm(4); self.classifier=nn.Linear(4,5)
    def forward(self,x):
        pooled=x.mean(dim=(-2,-1))[:,:4]; return SimpleNamespace(logits=self.classifier(self.swin.layernorm(pooled)))
class FakeAutoModelForImageClassification:
    @classmethod
    def from_pretrained(cls,*args,**kwargs): return FakeHFClassifier()
fake_transformers=ModuleType("transformers"); fake_transformers.AutoModelForImageClassification=FakeAutoModelForImageClassification; sys.modules["transformers"]=fake_transformers
with tempfile.TemporaryDirectory() as tmp:
    checkpoint_path=Path(tmp)/"model.pth"; source=FakeHFClassifier()
    with torch.no_grad():
        for parameter in source.parameters(): parameter.fill_(0.25)
    torch.save({"epoch_3":source.state_dict()},checkpoint_path)
    loaded,epoch_key=load_swin_classifier("fake-swin",5,checkpoint_path,torch.device("cpu"))
    assert epoch_key=="epoch_3"
    for key,value in source.state_dict().items(): assert torch.equal(value,loaded.state_dict()[key]),key
print("Checkpoint loader smoke test passed.")
