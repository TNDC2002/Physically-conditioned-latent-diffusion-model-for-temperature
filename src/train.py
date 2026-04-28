from typing import List, Optional, Tuple

import hydra
import lightning as L
import pyrootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
import wandb

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src import utils

log = utils.get_pylogger(__name__)

torch.set_float32_matmul_precision('medium')


def _reset_reduce_on_plateau_scheduler(scheduler) -> bool:
    """Reset ReduceLROnPlateau internals to fresh state."""
    class_name = scheduler.__class__.__name__
    if class_name != "ReduceLROnPlateau":
        return False

    mode = getattr(scheduler, "mode", "min")
    scheduler.best = float("inf") if mode == "min" else float("-inf")
    if hasattr(scheduler, "num_bad_epochs"):
        scheduler.num_bad_epochs = 0
    if hasattr(scheduler, "cooldown_counter"):
        scheduler.cooldown_counter = 0
    if hasattr(scheduler, "last_epoch"):
        scheduler.last_epoch = 0
    if hasattr(scheduler, "_last_lr"):
        scheduler._last_lr = [pg["lr"] for pg in scheduler.optimizer.param_groups]
    return True


class ResumeResetCallback(Callback):
    """On resumed runs, optionally reset LR scheduler and LR before first train batch."""

    def __init__(self, reset_scheduler: bool, reset_lr_to_default: bool, default_lr: Optional[float]) -> None:
        super().__init__()
        self.reset_scheduler = bool(reset_scheduler)
        self.reset_lr_to_default = bool(reset_lr_to_default)
        self.default_lr = default_lr
        self._applied = False

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self._applied:
            return

        if self.reset_lr_to_default:
            lr_target = self.default_lr
            if lr_target is None:
                lr_target = getattr(pl_module, "lr", None)
            if lr_target is None:
                raise ValueError(
                    "reset_lr_to_default_on_resume=true but no default LR found in config/model."
                )
            for opt in trainer.optimizers:
                for param_group in opt.param_groups:
                    param_group["lr"] = float(lr_target)
            log.info(f"Resume reset: restored optimizer state but set current LR to default={float(lr_target):.8g}")

        if self.reset_scheduler:
            reset_ok = 0
            total = 0
            for sched_cfg in trainer.lr_scheduler_configs:
                scheduler = sched_cfg.scheduler
                total += 1
                if _reset_reduce_on_plateau_scheduler(scheduler):
                    reset_ok += 1
            if total > 0:
                log.info(f"Resume reset: scheduler state reset for {reset_ok}/{total} schedulers")
            else:
                log.info("Resume reset: no schedulers attached to trainer")

        self._applied = True


@utils.task_wrapper
def train(cfg: DictConfig) -> Tuple[dict, dict]:
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    ckpt_path = cfg.get("ckpt_path")
    # NEW: Read the flag from the root config instead of from cfg.model.
    load_opt_state = cfg.get("load_optimizer_state", True)
    if ckpt_path and not load_opt_state:
        log.info(f"Partial training mode: loading model weights from {ckpt_path} without optimizer state.")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        ckpt_path = None

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))
    if cfg.get("reset_scheduler_on_resume", False) or cfg.get("reset_lr_to_default_on_resume", False):
        callbacks.append(
            ResumeResetCallback(
                reset_scheduler=cfg.get("reset_scheduler_on_resume", False),
                reset_lr_to_default=cfg.get("reset_lr_to_default_on_resume", False),
                default_lr=cfg.get("reset_lr_default_value"),
            )
        )

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    if cfg.get("compile"):
        log.info("Compiling model!")
        model = torch.compile(model)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    utils.extras(cfg)
    metric_dict, _ = train(cfg)
    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )
    return metric_value


if __name__ == "__main__":

    # wandb.init(
    #     project="689", 
    #     name="Latent_Diffusion", 
    #     entity="bahngmc-duke-university"
    # )
    main()
    # wandb.finish()
