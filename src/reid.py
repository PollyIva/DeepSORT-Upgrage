"""Person Re-Identification models."""

import numpy as np
import torch
import torchreid
import torchvision.transforms as T


class ReIDModel:
    def get_embedding(self, image):
        raise NotImplementedError


class TorchReID(ReIDModel):
    def __init__(self, model_name, device=DEVICE):
        self.device = device
        self.model = torchreid.models.build_model(
            name=model_name,
            num_classes=1000,
            pretrained=True
        )
        self.model.to(device)
        self.model.eval()

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    @torch.no_grad()
    def get_embedding(self, image):
        """Остается для обратной совместимости"""
        img = self.transform(image).unsqueeze(0).to(self.device)
        feature = self.model(img).cpu().numpy()[0]
        feature /= np.linalg.norm(feature)
        return feature

    @torch.no_grad()
    def get_embeddings_batch(self, images):
        """Батчевая обработка списка изображений"""
        if not images:
            return []

        # Трансформируем каждый кроп и собираем в единый батч-тензор
        tensors = [self.transform(img) for img in images]
        batch = torch.stack(tensors).to(self.device)  # Форма: [N, C, H, W]

        # Прогоняем весь батч через модель за один пасс
        features = self.model(batch).cpu().numpy()  # Форма: [N, Features]

        # L2 нормализация по строкам (для каждого эмбеддинга отдельно)
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        features /= np.zeros_like(norms) + norms + 1e-8  # Защита от деления на 0

        return features

def create_reid(name, device="cpu"):
    if name in {"osnet_x0_25", "osnet_x1_0", "resnet50"}:
        return TorchReID(name, device=device)
    raise ValueError(f"Unknown ReID model: {name}")
