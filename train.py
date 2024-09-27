from typing import List
from loguru import logger
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm
from datasets import load_dataset

# Logging setup
# logger.add("finetune_swin_detailed.log", rotation="500 MB")

# Dataset configuration
DATASETS = [
    "hongrui/mimic_chest_xray_v_1",
    "BahaaEldin0/NIH-Chest-Xray-14-Augmented-70-percent",
    "wal14567/train_xray_dataset",
]

# Image size configuration (for standardizing input)
IMAGE_SIZE = 384  # Adjusted for Swin Transformer model (swin_large_patch4_window12_384)

# Device configuration
device = torch.device("cpu")  # Only using CPU
logger.info(f"Using device: {device}")


# Custom dataset class for combining datasets
class XrayDataset(Dataset):
    def __init__(self, datasets: List[str], transform=None):
        self.transform = transform
        self.data = []
        self.labels = set()

        try:
            for dataset_name in datasets:
                logger.info(f"Loading dataset: {dataset_name}")
                dataset = load_dataset(dataset_name, split="train")
                for item in dataset:
                    self.data.append(
                        {
                            "image": item["image"],
                            "label": item.get("report")
                            or item.get("label"),
                        }
                    )
                    self.labels.add(
                        item.get("report") or item.get("label")
                    )
            logger.info(
                f"Loaded {len(self.data)} samples from datasets"
            )
        except Exception as e:
            logger.error(f"Error loading datasets: {str(e)}")
            raise

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            item = self.data[idx]
            image = item["image"]
            label = item["label"]

            if self.transform:
                image = self.transform(image)

            return image, label
        except Exception as e:
            logger.error(
                f"Error fetching item at index {idx}: {str(e)}"
            )
            raise

    def get_unique_labels(self) -> List[str]:
        try:
            return list(self.labels)
        except Exception as e:
            logger.error(f"Error retrieving unique labels: {str(e)}")
            raise


# Define image transformations
try:
    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    logger.info("Image transformations defined successfully")
except Exception as e:
    logger.error(f"Error defining image transformations: {str(e)}")
    raise


# Load datasets and fuse them into one DataLoader
def load_datasets(datasets: List[str], batch_size: int) -> DataLoader:
    try:
        logger.info(f"Loading and fusing datasets: {datasets}")
        dataset = XrayDataset(datasets, transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        logger.info(
            f"Dataset and DataLoader created successfully with batch_size {batch_size}"
        )
        return dataset, dataloader
    except Exception as e:
        logger.error(f"Error loading datasets: {str(e)}")
        raise


# Fine-tune the Swin Transformer model
def finetune_swin(
    model_name: str,
    learning_rate: float,
    epochs: int,
    batch_size: int,
):
    try:
        # Load datasets and dataloader
        dataset, dataloader = load_datasets(DATASETS, batch_size)
        unique_labels = dataset.get_unique_labels()
        num_classes = len(unique_labels)

        logger.info(
            f"Number of unique labels (classes): {num_classes}"
        )
        logger.info(f"Unique labels: {unique_labels}")

        # Create label-to-index mapping
        label_to_idx = {
            label: idx for idx, label in enumerate(unique_labels)
        }

        # Load Swin Transformer model from timm
        logger.info(f"Loading model {model_name} from timm.")
        model = timm.create_model(
            model_name, pretrained=True, num_classes=num_classes
        )
        model.to(device)

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            model.train()
            running_loss = 0.0

            for i, (inputs, labels) in enumerate(dataloader):
                try:
                    inputs = inputs.to(device)
                    labels = torch.tensor(
                        [label_to_idx[label] for label in labels]
                    ).to(device)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    if i % 10 == 9:
                        logger.info(
                            f"Batch {i + 1}, Loss: {running_loss / 10}"
                        )
                        running_loss = 0.0
                except Exception as e:
                    logger.error(
                        f"Error during batch {i + 1} of epoch {epoch + 1}: {str(e)}"
                    )
                    raise

            logger.info(f"Finished epoch {epoch + 1}")

        # Save the model after training
        save_path = f"finetuned_{model_name}.pth"
        torch.save(model.state_dict(), save_path)
        logger.info(f"Model saved to {save_path}")

    except Exception as e:
        logger.error(f"Error during finetuning: {str(e)}")
        raise


# Main function to execute finetuning
def main():
    try:
        logger.info(
            "Starting Swin Transformer finetuning on X-ray datasets."
        )

        # model_name = (
        #     "swin_large_patch4_window12_384"  # Swin Transformer model
        # )
        model_name = "vit_base_patch16_224"
        learning_rate = 1e-4
        epochs = 10
        batch_size = 16

        finetune_swin(model_name, learning_rate, epochs, batch_size)
    except Exception as e:
        logger.error(f"Error in the main function: {str(e)}")
        raise


if __name__ == "__main__":
    main()
