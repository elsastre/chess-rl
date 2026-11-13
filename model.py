"""
Red de valor: posicion (12x8x8) -> valor en [-1, 1] para el bando que mueve.

ResNet chica. Con canales=64 y bloques=4 tiene unos 330 mil parametros:
entra holgada en la 4050 / 5060 y evalua miles de posiciones por segundo.
La ultima capa convolucional queda guardada en self.last_features para Grad-CAM.
"""
import torch
import torch.nn as nn
from encoding import N_PLANES


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        y = torch.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return torch.relu(x + y)


class ValueNet(nn.Module):
    def __init__(self, channels=64, blocks=4):
        super().__init__()
        self.config = {"channels": channels, "blocks": blocks}
        self.stem = nn.Sequential(
            nn.Conv2d(N_PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.body = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        self.head = nn.Sequential(
            nn.Conv2d(channels, 8, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * 64, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )
        self.last_features = None

    def forward(self, x):
        h = self.body(self.stem(x))
        self.last_features = h  # para Grad-CAM
        return self.head(h).squeeze(-1)


def save_checkpoint(path, model, extra=None):
    torch.save({"config": model.config, "state": model.state_dict(), "extra": extra or {}}, path)


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = ValueNet(**ck["config"]).to(device)
    model.load_state_dict(ck["state"])
    model.eval()
    return model, ck.get("extra", {})


def default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"
