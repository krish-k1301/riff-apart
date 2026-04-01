import torch
from torch.utils.data import DataLoader

from app.data.dataset import MUSDBDataset


def _collate_fn(batch):
    """
    Pad all spectrograms in the batch to the maximum time-frame length
    found in that batch, then stack into tensors.
    """
    max_t = max(item["mix_mag"].shape[-1] for item in batch)

    def pad(tensor, max_t):
        t = tensor.shape[-1]
        if t < max_t:
            pad_size = [0] * (tensor.ndim * 2)
            pad_size[1] = max_t - t          # pad last dim on the right
            tensor = torch.nn.functional.pad(tensor, pad_size)
        return tensor

    mix_mags    = torch.stack([pad(item["mix_mag"],    max_t) for item in batch])
    target_mags = torch.stack([pad(item["target_mag"], max_t) for item in batch])
    mix_phases  = torch.stack([pad(item["mix_phase"],  max_t) for item in batch])

    return {
        "mix_mag":    mix_mags,
        "target_mag": target_mags,
        "mix_phase":  mix_phases,
    }


def get_dataloaders(
    root=None,
    batch_size=8,
    chunk_duration=5.0,
    sr=44100,
    num_workers=0,
    target="vocals",
):
    """
    Returns {"train": DataLoader, "test": DataLoader}.
    Uses MUSDB18 7-second preview clips when root is None (download=True).
    """
    pin = torch.cuda.is_available()

    train_ds = MUSDBDataset(
        root=root, split="train", target=target,
        chunk_duration=chunk_duration, sr=sr, download=True,
    )
    test_ds = MUSDBDataset(
        root=root, split="test", target=target,
        chunk_duration=chunk_duration, sr=sr, download=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_collate_fn,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_fn,
        pin_memory=pin,
    )

    return {"train": train_loader, "test": test_loader}
