from __future__ import annotations

import wandb
import os
import numpy as np
from typing import List, Optional, Dict
from prettytable import PrettyTable

from vint_train.training.train_utils import train, evaluate
from vint_train.training.train_utils import train_nomad, evaluate_nomad
from vint_train.training.training_artifacts import checkpoint_filename
from vint_train.training.training_artifacts import is_best_validation_loss
from vint_train.training.training_artifacts import should_save_numbered_checkpoint
from vint_train.training.training_artifacts import ValidationPoint
from vint_train.training.training_artifacts import write_validation_artifacts

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms

def train_eval_loop(
    train_model: bool,
    model: nn.Module,
    optimizer: Adam,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    dataloader: DataLoader,
    test_dataloaders: Dict[str, DataLoader],
    transform: transforms,
    epochs: int,
    device: torch.device,
    project_folder: str,
    normalized: bool,
    wandb_log_freq: int = 10,
    print_log_freq: int = 100,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    current_epoch: int = 0,
    alpha: float = 0.5,
    learn_angle: bool = True,
    use_wandb: bool = True,
    eval_fraction: float = 0.25,
    goal_type: str = "image",
    model_type: str = "vint",
    checkpoint_frequency: int = 1,
    save_best_checkpoint: bool = False,
    validation_history_filename: str = "validation_loss.csv",
    validation_plot_filename: str = "validation_loss.png",
):
    """
    Train and evaluate the model for several epochs (vint or gnm models)

    Args:
        train_model: whether to train the model or not
        model: model to train
        optimizer: optimizer to use
        scheduler: learning rate scheduler to use
        dataloader: dataloader for train dataset
        test_dataloaders: dict of dataloaders for testing
        transform: transform to apply to images
        epochs: number of epochs to train
        device: device to train on
        project_folder: folder to save checkpoints and logs
        normalized: whether to normalize the action space or not
        wandb_log_freq: frequency of logging to wandb
        print_log_freq: frequency of printing to console
        image_log_freq: frequency of logging images to wandb
        num_images_log: number of images to log to wandb
        current_epoch: epoch to start training from
        alpha: tradeoff between distance and action loss
        learn_angle: whether to learn the angle or not
        use_wandb: whether to log to wandb or not
        eval_fraction: fraction of training data to use for evaluation
    """
    assert 0 <= alpha <= 1
    if checkpoint_frequency <= 0:
        raise ValueError("checkpoint_frequency must be positive")
    latest_path = os.path.join(project_folder, f"latest.pth")
    best_path = os.path.join(project_folder, "best.pth")
    best_validation_loss = None
    validation_history = []

    for epoch in range(current_epoch, current_epoch + epochs):
        if train_model:
            print(
            f"Start ViNT Training Epoch {epoch}/{current_epoch + epochs - 1}"
            )
            train(
                model=model,
                optimizer=optimizer,
                dataloader=dataloader,
                transform=transform,
                device=device,
                project_folder=project_folder,
                normalized=normalized,
                epoch=epoch,
                alpha=alpha,
                learn_angle=learn_angle,
                print_log_freq=print_log_freq,
                wandb_log_freq=wandb_log_freq,
                image_log_freq=image_log_freq,
                num_images_log=num_images_log,
                use_wandb=use_wandb,
                goal_type=goal_type,
            )

        avg_total_test_loss = []
        for dataset_type in test_dataloaders:
            print(
                f"Start {dataset_type} ViNT Testing Epoch {epoch}/{current_epoch + epochs - 1}"
            )
            loader = test_dataloaders[dataset_type]

            test_dist_loss, test_action_loss, total_eval_loss = evaluate(
                eval_type=dataset_type,
                model=model,
                dataloader=loader,
                transform=transform,
                device=device,
                project_folder=project_folder,
                normalized=normalized,
                epoch=epoch,
                alpha=alpha,
                learn_angle=learn_angle,
                num_images_log=num_images_log,
                use_wandb=use_wandb,
                eval_fraction=eval_fraction,
                goal_type=goal_type,
            )

            avg_total_test_loss.append(total_eval_loss)

        state_dict = (
            model.module.state_dict()
            if isinstance(model, nn.DataParallel)
            else model.state_dict()
        )
        validation_loss = float(np.mean(avg_total_test_loss))
        improved = is_best_validation_loss(
            validation_loss, best_validation_loss
        )
        if improved:
            best_validation_loss = validation_loss
        checkpoint = {
            "format_version": 1,
            "model_type": model_type,
            "epoch": epoch,
            "completed_epochs": epoch + 1,
            "state_dict": state_dict,
            "optimizer": optimizer.state_dict(),
            "avg_total_test_loss": validation_loss,
            "best_validation_loss": best_validation_loss,
            "scheduler": scheduler.state_dict() if scheduler is not None else None
        }
        # log average eval loss
        if use_wandb:
            wandb.log({}, commit=False)

        if scheduler is not None:
            # scheduler calls based on the type of scheduler
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(validation_loss)
            else:
                scheduler.step()
        if use_wandb:
            wandb.log({
                "avg_total_test_loss": validation_loss,
                "lr": optimizer.param_groups[0]["lr"],
            }, commit=False)

        torch.save(checkpoint, latest_path)
        if should_save_numbered_checkpoint(epoch, checkpoint_frequency):
            numbered_path = os.path.join(
                project_folder, checkpoint_filename(epoch)
            )
            torch.save(checkpoint, numbered_path)
            print(f"Saved interval checkpoint to {numbered_path}")
        if save_best_checkpoint and improved:
            torch.save(checkpoint, best_path)
            print(
                f"Saved best checkpoint to {best_path} "
                f"(validation loss {validation_loss:.6f})"
            )
        validation_history.append(ValidationPoint(epoch, validation_loss))
        csv_path, plot_path = write_validation_artifacts(
            project_folder,
            validation_history,
            csv_filename=validation_history_filename,
            plot_filename=validation_plot_filename,
        )
        print(f"Updated validation artifacts: {csv_path}, {plot_path}")

    # Flush the last set of eval logs
    if use_wandb:
        wandb.log({})
    print()

def train_eval_loop_nomad(
    train_model: bool,
    model: nn.Module,
    optimizer: Adam, 
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    noise_scheduler: DDPMScheduler,
    train_loader: DataLoader,
    test_dataloaders: Dict[str, DataLoader],
    transform: transforms,
    goal_mask_prob: float,
    epochs: int,
    device: torch.device,
    project_folder: str,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    current_epoch: int = 0,
    alpha: float = 1e-4,
    use_wandb: bool = True,
    eval_fraction: float = 0.25,
    eval_freq: int = 1,
):
    """
    Train and evaluate the model for several epochs (vint or gnm models)

    Args:
        model: model to train
        optimizer: optimizer to use
        lr_scheduler: learning rate scheduler to use
        noise_scheduler: noise scheduler to use
        dataloader: dataloader for train dataset
        test_dataloaders: dict of dataloaders for testing
        transform: transform to apply to images
        goal_mask_prob: probability of masking the goal token during training
        epochs: number of epochs to train
        device: device to train on
        project_folder: folder to save checkpoints and logs
        wandb_log_freq: frequency of logging to wandb
        print_log_freq: frequency of printing to console
        image_log_freq: frequency of logging images to wandb
        num_images_log: number of images to log to wandb
        current_epoch: epoch to start training from
        alpha: tradeoff between distance and action loss
        use_wandb: whether to log to wandb or not
        eval_fraction: fraction of training data to use for evaluation
        eval_freq: frequency of evaluation
    """
    from diffusers.training_utils import EMAModel

    latest_path = os.path.join(project_folder, f"latest.pth")
    ema_model = EMAModel(model=model,power=0.75)
    
    for epoch in range(current_epoch, current_epoch + epochs):
        if train_model:
            print(
            f"Start ViNT DP Training Epoch {epoch}/{current_epoch + epochs - 1}"
            )
            train_nomad(
                model=model,
                ema_model=ema_model,
                optimizer=optimizer,
                dataloader=train_loader,
                transform=transform,
                device=device,
                noise_scheduler=noise_scheduler,
                goal_mask_prob=goal_mask_prob,
                project_folder=project_folder,
                epoch=epoch,
                print_log_freq=print_log_freq,
                wandb_log_freq=wandb_log_freq,
                image_log_freq=image_log_freq,
                num_images_log=num_images_log,
                use_wandb=use_wandb,
                alpha=alpha,
            )
            lr_scheduler.step()

        numbered_path = os.path.join(project_folder, f"ema_{epoch}.pth")
        torch.save(ema_model.averaged_model.state_dict(), numbered_path)
        numbered_path = os.path.join(project_folder, f"ema_latest.pth")
        print(f"Saved EMA model to {numbered_path}")

        numbered_path = os.path.join(project_folder, f"{epoch}.pth")
        torch.save(model.state_dict(), numbered_path)
        torch.save(model.state_dict(), latest_path)
        print(f"Saved model to {numbered_path}")

        # save optimizer
        numbered_path = os.path.join(project_folder, f"optimizer_{epoch}.pth")
        latest_optimizer_path = os.path.join(project_folder, f"optimizer_latest.pth")
        torch.save(optimizer.state_dict(), latest_optimizer_path)

        # save scheduler
        numbered_path = os.path.join(project_folder, f"scheduler_{epoch}.pth")
        latest_scheduler_path = os.path.join(project_folder, f"scheduler_latest.pth")
        torch.save(lr_scheduler.state_dict(), latest_scheduler_path)


        if (epoch + 1) % eval_freq == 0: 
            for dataset_type in test_dataloaders:
                print(
                    f"Start {dataset_type} ViNT DP Testing Epoch {epoch}/{current_epoch + epochs - 1}"
                )
                loader = test_dataloaders[dataset_type]
                evaluate_nomad(
                    eval_type=dataset_type,
                    ema_model=ema_model,
                    dataloader=loader,
                    transform=transform,
                    device=device,
                    noise_scheduler=noise_scheduler,
                    goal_mask_prob=goal_mask_prob,
                    project_folder=project_folder,
                    epoch=epoch,
                    print_log_freq=print_log_freq,
                    num_images_log=num_images_log,
                    wandb_log_freq=wandb_log_freq,
                    use_wandb=use_wandb,
                    eval_fraction=eval_fraction,
                )
        wandb.log({
            "lr": optimizer.param_groups[0]["lr"],
        }, commit=False)

        if lr_scheduler is not None:
            lr_scheduler.step()

        # log average eval loss
        wandb.log({}, commit=False)

        wandb.log({
            "lr": optimizer.param_groups[0]["lr"],
        }, commit=False)

        
    # Flush the last set of eval logs
    wandb.log({})
    print()

def load_model(model, model_type, checkpoint: dict) -> None:
    """Load model from checkpoint."""
    if model_type == "nomad":
        state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
    elif "state_dict" in checkpoint:
        checkpoint_type = checkpoint.get("model_type")
        if checkpoint_type is not None and checkpoint_type != model_type:
            raise ValueError(
                f"Checkpoint model type {checkpoint_type} does not match {model_type}"
            )
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    else:
        loaded_model = checkpoint["model"]
        try:
            state_dict = loaded_model.module.state_dict()
            model.load_state_dict(state_dict, strict=False)
        except AttributeError as e:
            state_dict = loaded_model.state_dict()
            model.load_state_dict(state_dict, strict=False)


def load_ema_model(ema_model, state_dict: dict) -> None:
    """Load model from checkpoint."""
    ema_model.load_state_dict(state_dict)


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    # print(table)
    print(f"Total Trainable Params: {total_params/1e6:.2f}M")
    return total_params
