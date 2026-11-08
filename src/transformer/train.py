from pathlib import Path

import torch
import torch.nn.functional as F

from transformer.transformer import Transformer


def train(
    model: Transformer,
    train_data: torch.Tensor,
    test_data: torch.Tensor,
    batch_size: int,
    lr: float,
    epochs: int,
    save_path: Path,
) -> None:
    n = train_data.size(0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    test_loss = None

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=train_data.device)
        total = 0.0
        for i in range(0, n, batch_size):
            batch = train_data[perm[i : i + batch_size]]
            inputs, targets = batch[:, :-1], batch[:, 1:]
            logits = model(inputs)
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(batch)

        msg = f"epoch {epoch} | train loss {total / n:.4f}"
        test_loss = evaluate(model, test_data, batch_size)
        msg += f" | test loss {test_loss:.4f}"
        print(msg, flush=True)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": model.config,
            "train_config": {"batch_size": batch_size, "lr": lr, "epochs": epochs},
            "test_loss": test_loss,
        },
        save_path,
    )
    print(f"saved to {save_path}")


@torch.inference_mode()
def evaluate(model: Transformer, test: torch.Tensor, batch_size: int) -> float:
    model.eval()
    n = test.size(0)

    total = 0.0
    for i in range(0, n, batch_size):
        batch = test[i : i + batch_size]
        inputs, targets = batch[:, :-1], batch[:, 1:]
        logits = model(inputs)
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        total += loss.item() * len(batch)

    return total / n
