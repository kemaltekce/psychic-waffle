import copy
import logging
import numpy as np
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from psychic.dataset import (
    build_class_weights,
    RavdessAudioDataset,
    ID_EMOTION_MAPPER,
    dataset_summary,
    transform,
)
from psychic.model import (
    CNN,
    inspect_model,
    model_capacity_check,
    evaluate,
    calculate_confusion_matrix,
)

logger = logging.getLogger("psychic")


def run():
    # define seed for reproducebility
    torch.manual_seed(456)
    g = torch.Generator()
    g.manual_seed(456)

    logger.info("Loading data")
    # load data
    dataset = RavdessAudioDataset(transform=transform())
    dataset_summary(dataset)
    train_dataset = dataset.subset_actors(list(range(1, 19)))
    val_dataset = dataset.subset_actors(list(range(19, 22)))
    test_dataset = dataset.subset_actors(list(range(22, 25)))
    train_dataloader = DataLoader(
        train_dataset, batch_size=16, shuffle=True, generator=g
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=16, shuffle=False, generator=g
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=16, shuffle=False, generator=g
    )

    # dataloader = DataLoader(
    #     dataset, batch_size=32, shuffle=True, generator=g
    # )
    # data, meta = next(iter(dataloader))
    # plot_spectrogram(data[0][0], {k: v[0] for k, v in meta.items()})

    logger.info("Defining model parameters")
    # modeling
    model = CNN()
    inspect_model(model)
    model_capacity_check(model, len(train_dataset))
    learning_rate = 0.001
    # AdamW instead of Adam to keep weights small to generalize better
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    # use scheduler to adjust learning rate to avoid validation acc oszilations
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )
    class_weights = build_class_weights(train_dataset)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    epochs = 50
    early_stopping_patience = 5

    # keep history of model training
    model_history = {
        "loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }
    # track model validation performance for early stopping and best model
    best_train_acc = 0.0
    best_train_loss = float("inf")
    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_epoch = 0
    best_model_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    logger.info("Starting training loop")
    # training loop
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0
        total = 0
        correct = 0

        # train model
        for data, meta in train_dataloader:
            labels = meta["emotion"]

            optimizer.zero_grad()
            prediction = model(data)
            loss = loss_fn(prediction, labels)
            loss.backward()
            optimizer.step()

            total += labels.size(0)
            # softmax unnecessary for code but here for debugging purposes
            probability = torch.softmax(prediction, dim=1)
            probability_pred = probability.argmax(dim=1)
            correct += (probability_pred == labels).sum().item()
            # corss entropy loss is mean value. multiply by size to calculate
            # unskewed average with total later
            total_loss += loss.item() * labels.size(0)

        train_acc = correct / total
        avg_train_loss = total_loss / total

        # validation check
        avg_val_loss, val_acc, _, _ = evaluate(model, val_dataloader, loss_fn)
        # inform scheduler to maybe adujst learning rate for next epoch
        scheduler.step(val_acc)

        # save model history
        model_history["loss"].append(avg_train_loss)
        model_history["train_acc"].append(train_acc)
        model_history["val_loss"].append(avg_val_loss)
        model_history["val_acc"].append(val_acc)

        # if epoch % 10 == 0:
        logger.debug(
            f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f} / "
            f"Train Acc = {train_acc:.4f} / "
            f"Val Loss = {avg_val_loss:.4f} / "
            f"Val Acc = {val_acc:.4f} / "
        )

        # validation data for early stopping and model selection
        validation_improved = val_acc > best_val_acc or (
            np.isclose(val_acc, best_val_acc) and avg_val_loss < best_val_loss
        )
        if validation_improved:
            best_train_acc = train_acc
            best_train_loss = avg_train_loss
            best_val_acc = val_acc
            best_val_loss = avg_val_loss
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_patience:
            logger.debug(
                f"Early stopping at epoch {epoch}. "
                f"No improvement for {epochs_without_improvement} epochs. "
                f"Best epoch was {best_epoch}."
            )
            break

    # pick best model
    logger.info(
        f"Restored best model from epoch {best_epoch} "
        "with best performance: \n"
        f"    Train Loss = {best_train_loss:.4f} / "
        f"Train Acc = {best_train_acc:.4f} / "
        f"Val Loss = {best_val_loss:.4f} / "
        f"Val Acc = {best_val_acc:.4f} / "
    )
    model.load_state_dict(best_model_state)

    # final test. test model with test set
    avg_test_loss, test_acc, test_labels, test_predictions = evaluate(
        model, test_dataloader, loss_fn
    )
    logger.info(
        "Final test with testset: "
        f"Loss = {avg_test_loss:.4f} / Acc = {test_acc:.4f}"
    )
    _ = calculate_confusion_matrix(
        test_labels, test_predictions, 8, ID_EMOTION_MAPPER
    )
