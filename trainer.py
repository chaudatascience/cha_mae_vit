#### Adapted from https://github.com/THUDM/CogVideo/blob/5ab1e2449ffc8887ffad3ca3b9efd22ad7e356f7/finetune/trainer.py

## General imports
import json
import math
from datetime import timedelta
from pathlib import Path
from typing import Any, Optional
from copy import deepcopy
import collections
import numpy as np
import random
import time
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from tqdm import tqdm
from safetensors.torch import load_file
import os
import socket
import shutil

## ML stuff
import torch
import kornia.augmentation as K
import torch.nn as nn
import torch._dynamo as dynamo
from torch.utils.data import DataLoader
from einops import rearrange
from diffusers.optimization import get_scheduler
from accelerate.accelerator import Accelerator, DistributedType
from accelerate.utils import (
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    set_seed,
)

## Local imports
from custom_log import MyWandBLogger, DummyLogger
from dataset.chammiv1 import get_chammiv1_dataloaders
from dataset.jumpcp import get_jumpcp_dataloaders
from dataset.so2sat import get_so2sat_dataloaders
from dataset.dataset_utils import self_normalize
from models.cha_mae_vit import get_multi_channel_vit
from optimizer import get_optimizer
from morphem.benchmark import run_benchmark as run_chammiv1_benchmark
from utils import *
from torchvision.utils import save_image


class Trainer:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

        self.train_weight_dtype: torch.dtype = self._get_training_dtype(self.cfg.train.weight_dtype)
        self.overwrote_max_train_steps = False
        self.run_id, self.scc_jobid, self.seed = self._initialize_seed_and_jobid()

        self.accelerator = self._init_distributed(self.seed)
        self.use_ddp = self.accelerator.state.distributed_type != DistributedType.NO
        self.is_main_process = self.accelerator.is_main_process
        self.local_rank = self.accelerator.local_process_index  # int(os.environ.get("LOCAL_RANK", 0))
        self.global_rank = self.accelerator.process_index  # int(os.environ.get("RANK", 0))

        self.logger = self._init_logging()

        self._init_directories()

        self.use_deepspeed = self.accelerator.state.deepspeed_plugin is not None

        ## prepare  dataloaders
        self.training_dataset = cfg.data.training_dataset
        self.train_loader, self.valid_loader, self.test_loaders = self.prepare_dataloaders()

        ## prepare model
        self.model = self.prepare_models()

        ## TODO: make flag to skip this if only inference
        ## https://huggingface.co/docs/accelerate/quicktour for validation and testing
        self.optimizer, self.lr_scheduler = self.prepare_optimizer()
        self.prepare_for_training()

        if self.is_main_process:
            self._log_config_and_model_info()

    def _initialize_seed_and_jobid(self) -> tuple[str, str, int]:
        scc_jobid = default(self.cfg.logging.scc_jobid, "")
        seed = default(self.cfg.train.seed, random.randint(10, 100000))

        if scc_jobid != "":
            run_id = f"jobid_{scc_jobid}--seed_{seed}"
        else:  ## run locally (not on SCC), mainly for debugging
            cur_date = datetime_now("%Y-%b-%d")
            run_id = f"{cur_date}--seed_{seed}"
        return run_id, scc_jobid, seed

    def _init_distributed(self, seed):
        logging_dir = Path(self.cfg.logging.output_dir, "logs")
        project_config = ProjectConfiguration(project_dir=self.cfg.logging.output_dir, logging_dir=logging_dir)
        ## if using mae loss and caemae decoder
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)  # True)
        init_process_group_kwargs = InitProcessGroupKwargs(backend="nccl", timeout=timedelta(seconds=self.cfg.train.nccl_timeout))
        self.mixed_precision = "no" if torch.backends.mps.is_available() else self.cfg.train.mixed_precision
        report_to = self.cfg.logging.report_to

        accelerator = Accelerator(
            project_config=project_config,
            gradient_accumulation_steps=self.cfg.train.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            log_with=report_to,
            kwargs_handlers=[ddp_kwargs, init_process_group_kwargs],
        )

        # Disable AMP for MPS.
        if torch.backends.mps.is_available():
            accelerator.native_amp = False

        set_seed(seed + accelerator.process_index)

        return accelerator

    def _init_logging(self) -> MyWandBLogger | DummyLogger:
        if self.is_main_process:
            logger = MyWandBLogger(
                self.cfg,
                project_name=self.cfg.logging.project_name,
                run_id=self.run_id,
                use_ddp=self.use_ddp,
                use_wandb=self.cfg.logging.use_wandb,
            )
        else:
            print("DummyLogger for self.local_rank", self.local_rank)
            logger = DummyLogger(self.cfg, use_ddp=self.use_ddp)

        logger.info("Initialized Trainer")
        logger.info(f"Accelerator state: \n{self.accelerator.state}", silent=False)
        return logger

    def _log_config_and_model_info(self) -> None:

        if self.cfg.logging.use_wandb:
            self.logger.log_config(self.cfg)

        self.logger.info(OmegaConf.to_yaml(self.cfg))
        self.logger.info(str(self.model))
        total_num, trainable_num = analyze_model(self.model, logger=self.logger.info)

        self.logger.info(f"Total parameters: {total_num:,};\tTrainable: {trainable_num:,};\tNon-trainable: {total_num - trainable_num:,}")
        self.logger.info({"total_params (M)": total_num / 1e6, "trainable_params": trainable_num / 1e6}, print_out=False)
        self.logger.info("Pytorch version: {}".format(torch.__version__))
        self.logger.info("Cuda version: {}".format(torch.version.cuda))  # type: ignore
        self.logger.info("Cudnn version: {}".format(torch.backends.cudnn.version()))  # type: ignore

    def _init_directories(self) -> None:
        self.cfg.logging.output_dir = Path(self.cfg.logging.output_dir, self.run_id)
        if self.is_main_process:
            self.cfg.logging.output_dir.mkdir(parents=True, exist_ok=True)

    @time_this_function
    def prepare_dataloaders(self) -> tuple[DataLoader, Optional[DataLoader], dict[str, DataLoader]]:
        self.logger.info("Initializing datasets and dataloaders...")
        train_loader, valid_loader, test_loaders = None, None, None

        batch_size = self.cfg.train.batch_size
        eval_batch_size = self.cfg.eval.eval_batch_size
        num_workers = self.cfg.train.num_workers

        ################################
        ##### set training loader ######
        ################################
        #### CHAMMI dataset: https://arxiv.org/abs/2310.19224
        if self.training_dataset == "chammiv1":
            chammiv1_cfg = self.cfg.data.chammiv1
            folder = self.cfg.logging.output_dir
            self.cfg.eval.chammiv1.dest_dir = os.path.join(folder, self.cfg.eval.chammiv1.dest_dir)
            self.cfg.eval.chammiv1.feature_dir = os.path.join(folder, self.cfg.eval.chammiv1.feature_dir)
            self.logger.info("Loading CHAMMIv1 dataset...")
            train_loader, _, test_loaders = get_chammiv1_dataloaders(
                **chammiv1_cfg,
                batch_size=batch_size,
                eval_batch_size=eval_batch_size,
                training_chunk="morphem70k",
                num_workers=num_workers,
            )
        elif self.training_dataset == "jumpcp":
            self.channels_split = self.cfg.data.jumpcp.channels
            jumpcp_cfg = self.cfg.data.jumpcp
            self.logger.info("Loading JUMP-CP dataset...")
            train_loader, valid_loader, test_loaders = get_jumpcp_dataloaders(
                **jumpcp_cfg,
                batch_size=batch_size,
                eval_batch_size=eval_batch_size,
                num_workers=num_workers,
            )
        elif self.training_dataset == "so2sat":
            self.channels_split = self.cfg.data.so2sat.channels
            so2sat_cfg = self.cfg.data.so2sat
            self.logger.info("Loading So2Sat dataset...")
            train_loader, valid_loader, test_loaders = get_so2sat_dataloaders(
                **so2sat_cfg,
                batch_size=batch_size,
                eval_batch_size=eval_batch_size,
                num_workers=num_workers,
            )
        else:
            raise NotImplementedError(f"Unsupported training dataset: {self.training_dataset}")

        return train_loader, valid_loader, test_loaders

    def prepare_models(self) -> torch.nn.Module:
        self.logger.info("Initializing models")

        if self.training_dataset == "chammiv1":
            num_classes = self.cfg.data.chammiv1.num_classes
            num_channels = self.cfg.data.chammiv1.in_chans
            img_size = self.cfg.data.chammiv1.img_size
        elif self.training_dataset == "jumpcp":
            num_classes = self.cfg.data.jumpcp.num_classes
            num_channels = self.cfg.data.jumpcp.in_chans
            img_size = self.cfg.data.jumpcp.img_size
        elif self.training_dataset == "so2sat":
            num_classes = self.cfg.data.so2sat.num_classes
            num_channels = self.cfg.data.so2sat.in_chans
            img_size = self.cfg.data.so2sat.img_size
        else:
            raise ValueError(f"Unsupported training dataset: {self.training_dataset}")

        model_cfg = {
            **self.cfg.model,  # Automatically include all parameters in cfg.model
            "in_chans": num_channels,
            "num_classes": num_classes,
            "num_proxies": num_classes,
            "img_size": img_size,
        }
        model = get_multi_channel_vit(**model_cfg)
        model = model.to(self.accelerator.device, dtype=self.train_weight_dtype)

        if self.cfg.train.use_torch_compile:
            ## TODO: make this faster. It runs very slow now. It seems `torch.compile` wants a fixed input size,
            # but we have dynamic input size depending on the channels
            self.logger.info("------------- Compiling model with torch.compile")
            dynamo.config.dynamic_shapes = True  # Enable dynamic shapes (optional)
            dynamo.config.cache_size_limit = 128
            model = torch.compile(model, backend="inductor", mode="default")

        return model

    def prepare_optimizer(self, reinit_step: int = 0) -> tuple[torch.optim.Optimizer, Any]:
        """
        reinit_step: used for reinitializing optimizer and lr_scheduler after training for `reinit_step`.
        """
        self.logger.info("Initializing optimizer and lr scheduler")

        learning_rate = self.cfg.train.learning_rate
        scheduler_name = self.cfg.train.lr_scheduler

        # trainable_parameters = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        all_parameters = list(self.model.parameters())

        transformer_parameters_with_lr = {
            "params": all_parameters,
            "lr": learning_rate,
        }
        params_to_optimize = [transformer_parameters_with_lr]
        self.all_parameters_in_optimizer = sum(p.numel() for p in all_parameters)

        use_deepspeed_opt = (
            self.accelerator.state.deepspeed_plugin is not None and "optimizer" in self.accelerator.state.deepspeed_plugin.deepspeed_config
        )

        optimizer = get_optimizer(
            params_to_optimize=params_to_optimize,
            optimizer_name=self.cfg.train.optimizer,
            learning_rate=learning_rate,
            beta1=self.cfg.train.beta1,
            beta2=self.cfg.train.beta2,
            beta3=self.cfg.train.get("beta3", None),
            epsilon=self.cfg.train.epsilon,
            weight_decay=self.cfg.train.weight_decay,
            use_deepspeed=use_deepspeed_opt,
        )
        # from timm.optim import AdamW

        # optimizer = AdamW(
        #     params=params_to_optimize,
        #     lr=learning_rate,
        #     betas=(self.cfg.train.beta1, self.cfg.train.beta2),
        #     eps=self.cfg.train.epsilon,
        #     weight_decay=self.cfg.train.weight_decay,
        # )

        num_update_steps_per_epoch = math.ceil(len(self.train_loader) / self.cfg.train.gradient_accumulation_steps)
        num_warmup_steps = self.cfg.train.lr_warmup_epochs * num_update_steps_per_epoch
        if reinit_step == 0:  ## regular training
            if self.cfg.train.train_steps is None:
                self.cfg.train.train_steps = self.cfg.train.train_epochs * num_update_steps_per_epoch
                self.overwrote_max_train_steps = True
            total_training_steps = self.cfg.train.train_steps
        else:  ## reinit optimizer and lr_scheduler after training for `reinit_step` steps
            #  `self.cfg.train.train_steps` here is already adjusted to distributed training, thus, the total steps is:
            total_training_steps = self.cfg.train.train_steps * self.accelerator.num_processes
            total_training_steps -= reinit_step * self.accelerator.num_processes  ## num steps left to train

        use_deepspeed_lr_scheduler = (
            self.accelerator.state.deepspeed_plugin is not None and "scheduler" in self.accelerator.state.deepspeed_plugin.deepspeed_config
        )

        if use_deepspeed_lr_scheduler:
            from accelerate.utils import DummyScheduler

            lr_scheduler = DummyScheduler(
                name=scheduler_name,
                optimizer=optimizer,
                total_num_steps=total_training_steps,
                num_warmup_steps=num_warmup_steps,
            )
        else:
            lr_scheduler = get_scheduler(
                name=scheduler_name,
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_training_steps,
                num_cycles=self.cfg.train.lr_num_cycles,
                power=self.cfg.train.lr_power,
            )

        return optimizer, lr_scheduler

    def prepare_for_training(self) -> None:
        self.model, self.optimizer, self.train_loader, self.valid_loader, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.valid_loader, self.lr_scheduler
        )
        # Prepare each test_loader in self.test_loaders: dict[str, DataLoader]
        for k in self.test_loaders:
            self.test_loaders[k] = self.accelerator.prepare(self.test_loaders[k])

        # We need to recalculate our total training steps
        # as `self.train_loader` may be divided by `num_processes`
        num_update_steps_per_epoch = math.ceil(len(self.train_loader) / self.cfg.train.gradient_accumulation_steps)
        if self.overwrote_max_train_steps:
            self.cfg.train.train_steps = self.cfg.train.train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        self.cfg.train.train_epochs = math.ceil(self.cfg.train.train_steps / num_update_steps_per_epoch)
        print("---- self.cfg.train.train_epochs=", self.cfg.train.train_epochs)
        print("==== self.accelerator.num_processes=", self.accelerator.num_processes)
        self.num_update_steps_per_epoch = num_update_steps_per_epoch

    @property
    def model_(self) -> torch.nn.Module:
        ## get model from accelerator
        if self.accelerator.state.distributed_type in [DistributedType.DEEPSPEED, DistributedType.MULTI_GPU]:
            return self.accelerator.unwrap_model(self.model)
        else:
            return self.model

    def _debug_forward_pass(self, batch_i, batch, epoch) -> None:
        print(
            f"DEBUG: batch_i={batch_i}, batch keys: {list(batch.keys())}, \
            image shape: {batch['image'].shape}, label shape: {batch['label'].shape}"
        )
        self.logger.info(f"DEBUG: batch_i={batch_i}")
        b = batch["image"].shape[0]
        ## save 16 images in the batch, note images can have more/fewer than 3 channels
        imgs = torch.cat([batch["image"][:8], batch["image"][b // 2 : b // 2 + 8]], dim=0)  # (16, c, h, w)
        n_channels = imgs.shape[1]
        if n_channels > 3:
            imgs = imgs[:, :3]  # (16, 3, h, w)
        elif n_channels == 2:
            imgs = torch.cat([imgs, torch.zeros_like(imgs[:, :1])], dim=1)  # (16, 3, h, w)
        elif n_channels == 1:
            imgs = torch.cat([imgs, torch.zeros_like(imgs[:, :2])], dim=1)  # (16, 3, h, w)
        datetime_str = datetime_now("%Y%m%d")
        out_img_folder_ = os.path.join("output_temp", datetime_str)
        if not os.path.exists(out_img_folder_):
            os.makedirs(out_img_folder_)
        out_img_path = os.path.join(out_img_folder_, f"epoch{epoch}_batch{batch_i}_img.png")
        print(f"Saving debug images to {out_img_path}")
        save_image(imgs, out_img_path, nrow=4)

    def train(self) -> None:
        self.logger.info("Starting training...")

        memory_statistics = get_memory_statistics()
        self.logger.info(f"Memory before training start: {json.dumps(memory_statistics, indent=4)}")

        self.total_batch_size_count = self.cfg.train.batch_size * self.accelerator.num_processes * self.cfg.train.gradient_accumulation_steps
        master_ip = os.environ.get("MASTER_ADDR", "<unknown>")
        master_port = os.environ.get("MASTER_PORT", "<unknown>")

        info = {
            "total parameters in optimizer": self.all_parameters_in_optimizer,
            "total samples": len(self.train_loader.dataset) if self.train_loader is not None else 0,
            "train epochs": self.cfg.train.train_epochs,
            "train steps": self.cfg.train.train_steps,
            "batches per device": self.cfg.train.batch_size,
            "total batches observed per epoch": len(self.train_loader),
            "train batch size total count": self.total_batch_size_count,
            "gradient accumulation steps": self.cfg.train.gradient_accumulation_steps,
            "use deepspeed": self.use_deepspeed,
            "attention cls": self.cfg.model.attention_cls,
            "mixed precision": self.mixed_precision,
            "accelerator master_ip": master_ip,
            "accelerator master_port": master_port,
        }
        self.logger.info(f"Training configuration: {json.dumps(info, indent=4)}")

        global_step = 0
        first_epoch = 0
        initial_global_step = 0

        # Potentially load in the weights and states from a previous save
        (
            resume_from_checkpoint_path,
            initial_global_step,
            global_step,
            first_epoch,
        ) = get_latest_ckpt_path_to_resume_from(
            resume_from_checkpoint=self.cfg.train.resume_from_checkpoint,
            num_update_steps_per_epoch=self.num_update_steps_per_epoch,
            logger=self.logger,
        )

        if resume_from_checkpoint_path is not None:
            self.accelerator.load_state(resume_from_checkpoint_path)
            self.logger.info(f">>>>>>>>>>> Loaded model from {resume_from_checkpoint_path} <<<<<<<<<<<<")

        if self.cfg.train.show_progress_bar:
            progress_bar = tqdm(
                range(0, self.cfg.train.train_steps),
                initial=initial_global_step,
                desc="Training steps",
                disable=not self.accelerator.is_local_main_process,
            )

        if self.cfg.eval.eval_before_training:
            self.logger.info("Evaluating model before training...")
            self.evaluate_model(epoch=first_epoch, global_step=global_step)

        accelerator = self.accelerator
        free_memory()
        epoch_timer = AverageMeter()
        total_epochs = self.cfg.train.train_epochs

        for epoch in range(first_epoch + 1, total_epochs + 1):
            time_epoch_start = time.time()
            debug_msg = " --------- DEBUGGING" if self.cfg.train.debug else ""
            self.logger.info(f"Starting epoch {epoch}/{total_epochs}{debug_msg}")
            self.model.train()

            batch_time = AverageMeter()
            data_time = AverageMeter()
            loss_meter = collections.defaultdict(lambda: AverageMeter())

            time_end_batch = time.time()
            for batch_i, batch in enumerate(self.train_loader):
                data_time.update(time.time() - time_end_batch)

                if self.cfg.train.debug:
                    self._debug_forward_pass(batch_i, batch, epoch)
                    if batch_i >= 4:
                        print(f"DEBUG mode: break after {batch_i+1} batches")
                        break

                with accelerator.accumulate(self.model):
                    if "channel_ids_list" in batch:
                        channel_ids_list = batch["channel_ids_list"]
                        valid_channel_masks = batch["valid_channel_masks"]
                    else:
                        channel_ids_list = valid_channel_masks = None
                    images, channel_ids_list, valid_channel_masks, labels = (
                        batch["image"],
                        channel_ids_list,
                        valid_channel_masks,
                        batch["label"],
                    )

                    images = images.to(
                        dtype=self.train_weight_dtype, device=accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
                    )

                    res = self.model(
                        x=images,
                        channel_ids_list=channel_ids_list,
                        valid_channel_masks=valid_channel_masks,
                        y=labels,
                    )
                    loss = res["loss"]
                    ########## done with forward pass ##############

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        if accelerator.distributed_type == DistributedType.DEEPSPEED:
                            grad_norm = self.model.get_global_grad_norm()
                            # In some cases the grad norm may not return a float
                            if torch.is_tensor(grad_norm):
                                grad_norm = grad_norm.item()
                        else:
                            if self.cfg.train.max_grad_norm:
                                grad_norm = accelerator.clip_grad_norm_(self.model.parameters(), self.cfg.train.max_grad_norm)
                                if torch.is_tensor(grad_norm):
                                    grad_norm = grad_norm.item()
                            else:
                                ## still compute grad norm to keep track of it, but do not clip
                                grad_norm = compute_grad_norm(self.model.parameters())

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    if self.cfg.train.show_progress_bar:
                        progress_bar.update(1)
                    global_step += 1

                for k, v in res.items():
                    if "loss" in k:
                        loss_meter[k].update(v.detach().item())
                del loss

                batch_time.update(time.time() - time_end_batch)
                time_end_batch = time.time()

                if global_step % self.cfg.logging.log_interval == 0:
                    batch_logs = {
                        "epoch": epoch,
                        "global_step": global_step,
                        **{k: round(v.avg, 4) for k, v in loss_meter.items()},
                        "lr": round(self.lr_scheduler.get_last_lr()[0], 8),
                        "grad_norm": round(grad_norm, 2),
                        "data_time[s]": round(data_time.avg, 4),
                        "batch_time[s]": round(batch_time.avg, 2),
                    }
                    # progress_bar.set_postfix(batch_logs)
                    self.logger.info(batch_logs, step=global_step)
                    ## reset meters
                    batch_time.reset()
                    data_time.reset()
                    for k in loss_meter:
                        loss_meter[k].reset()

            self._maybe_save_checkpoint(epoch)

            if (epoch % self.cfg.eval.eval_every_epochs == 0) or (epoch == total_epochs):
                self.logger.info(f"Evaluating model at epoch {epoch}...")
                self.evaluate_model(epoch, global_step)

            memory_statistics = get_memory_statistics()
            self.logger.info(f"Memory after epoch {epoch}: {json.dumps(memory_statistics, indent=4)}")

            epoch_timer.update(time.time() - time_epoch_start)
            self.logger.info({"minute/epoch": round(epoch_timer.avg / 60, 2)})
            need_time_str = convert_secs2time(epoch_timer.avg * (total_epochs - epoch), return_string=True)
            self.logger.info(need_time_str)  # type: ignore
            self.logger.info("=" * 40)

        accelerator.wait_for_everyone()
        self._maybe_save_checkpoint(total_epochs, must_save=True)
        self.logger.finish(f"Done training! Time now: {datetime_now()}")

    @torch.inference_mode()
    def evaluate_model(self, epoch: int, global_step: int) -> None:
        self.logger.info(f"--- Starting evaluate_model epoch {epoch}...")
        self.model.eval()
        free_memory()

        if self.training_dataset == "chammiv1":
            self.logger.info(f"evaluating dataset on CHAMMI...")
            ### call evaluation of CHAMMI benchmark for representation learning tasks
            self.eval_chammiv1(epoch=epoch, eval_loaders=self.test_loaders)
        else:
            ### regular classification evaluation
            self.logger.info(f"evaluating dataset: valid")
            self.eval_regular(epoch=epoch, loader=self.valid_loader, eval_name="valid")

            for test_name, test_loader in self.test_loaders.items():
                self.logger.info(f"evaluating dataset: {test_name}")
                self.eval_regular(epoch=epoch, loader=test_loader, eval_name=test_name)

    def _get_training_dtype(self, dtype: str) -> torch.dtype:
        DTYPE_MAP = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        if dtype == "no" or dtype == "fp32":
            return DTYPE_MAP["fp32"]
        elif dtype == "fp16":
            return DTYPE_MAP["fp16"]
        elif dtype == "bf16":
            return DTYPE_MAP["bf16"]
        else:
            raise ValueError(f"Invalid dtype precision: {dtype}")

    def _maybe_save_checkpoint(self, epoch: int, must_save: bool = False):
        if self.accelerator.distributed_type == DistributedType.DEEPSPEED or self.is_main_process:
            if must_save or epoch % self.cfg.train.checkpointing_epochs == 0:
                # for training
                save_path = get_intermediate_ckpt_path(
                    checkpointing_limit=self.cfg.train.checkpointing_limit,
                    epoch=epoch,
                    output_dir=self.cfg.logging.output_dir,
                    logger=self.logger,
                )
                self.accelerator.save_state(save_path, safe_serialization=True)

    @time_this_function
    @torch.inference_mode()
    def eval_chammiv1(self, epoch: int, eval_loaders: dict[str, DataLoader]) -> dict[str, float] | None:
        def log_res(eval_cfg, knn_metric):
            call_umap = eval_cfg["umap"] and (epoch == 0 or epoch == self.cfg.train.num_epochs)
            dataset = "morphem70k"
            assert knn_metric in ["l2", "cosine"]

            full_res = run_chammiv1_benchmark(
                eval_cfg["root_dir"],
                eval_cfg["dest_dir"],
                eval_cfg["feature_dir"],
                eval_cfg["feature_file"],
                eval_cfg["classifier"],
                call_umap,
                eval_cfg["use_gpu"],
                knn_metric,
                dataset,  # quick hack to run benchmark on only 1 dataset
            )
            ## log results
            full_res["key"] = full_res.iloc[:, 0:3].apply(lambda x: "/".join(x.astype(str)), axis=1)

            TASK_ONE_INDICES = [0, 2, 5]
            acc = dict(zip(full_res["key"] + f"/{knn_metric}/acc", full_res["accuracy"]))
            acc_task1 = np.mean(np.array(list(acc.values()))[TASK_ONE_INDICES]) * 100
            f1 = dict(zip(full_res["key"] + f"/{knn_metric}/f1", full_res["f1_score_macro"]))
            f1_task1 = np.mean(np.array(list(f1.values()))[TASK_ONE_INDICES])

            metrics_logger = {
                **acc,
                **f1,
                f"{classifier}/{knn_metric}/avg_task1_acc/": acc_task1,
                f"{classifier}/{knn_metric}/avg_task1_f1/": f1_task1,
            }

            self.logger.info(metrics_logger, sep="| ", padding_space=True)
            return metrics_logger

        eval_cfg = deepcopy(self.cfg.eval.chammiv1)
        ## make a new folder for each epoch
        eval_cfg.dest_dir = os.path.join(eval_cfg.dest_dir, f"epoch_{epoch}")
        mkdir(eval_cfg.dest_dir)
        mkdir(eval_cfg.feature_dir)

        self.model.eval()

        out_path_list = []
        ALL_CHUNKS = ["Allen", "HPA", "CP"]
        if self.accelerator.local_process_index < len(ALL_CHUNKS):
            with self.accelerator.split_between_processes(ALL_CHUNKS) as eval_chunks:
                print(f" Evaluating CHAMMIv1: [Rank_{self.accelerator.local_process_index}] running forward() with chunks {eval_chunks}...")
                for chunk_name in eval_chunks:
                    free_memory()
                    feat_outputs = []  # store feature vectors

                    self.logger.info(f"Start getting features for {chunk_name}...")
                    eval_loader = eval_loaders[chunk_name]
                    total_len = len(eval_loader)
                    desc = f"Getting features for {chunk_name}"
                    channel_ids_list_base = [self.cfg.data.chammiv1.channel_ids[chunk_name]]
                    channel_ids_list = channel_ids_list_base
                    for x in tqdm(eval_loader, total=total_len, miniters=int(total_len / 10), maxinterval=100, desc=desc):
                        b = x.shape[0]
                        x = x.to(self.accelerator.device, dtype=self.train_weight_dtype)
                        res = self.model(
                            x,
                            channel_ids_list=channel_ids_list,
                            valid_channel_masks=None,
                            y=None,
                        )
                        output = res["output"]
                        feat_outputs.append(output.detach().cpu().numpy())
                        del res
                    free_memory()

                    folder_path = os.path.join(eval_cfg.feature_dir, chunk_name)
                    mkdir(folder_path)

                    out_path = os.path.join(folder_path, eval_cfg.feature_file)
                    out_path_list.append(out_path)
                    feat_outputs = np.concatenate(feat_outputs, axis=0)
                    write_numpy(feat_outputs, out_path)  # type: ignore
                    del feat_outputs
        else:
            # idle ranks
            pass

        self.accelerator.wait_for_everyone()
        free_memory()
        ## after we have all features for 3 chunks (i.e., Allen, HPA, CP), we run the benchmark
        if self.is_main_process:
            cosine_res = None
            start = time.time()
            for classifier in eval_cfg.classifiers:  ## choices: ["knn", "svm"]
                eval_cfg.classifier = classifier
                if classifier == "knn":
                    for knn_metric in eval_cfg.knn_metrics:  ## choices: ["l2", "cosine"]
                        if "cosine" in knn_metric:
                            cosine_res = log_res(eval_cfg=eval_cfg, knn_metric=knn_metric)
                        else:
                            log_res(eval_cfg=eval_cfg, knn_metric=knn_metric)
                else:
                    log_res(eval_cfg=eval_cfg, knn_metric=None)

            stop = time.time()
            print(f"Done running benchmark in {(stop - start) / 60:.2f} minutes")

            if cosine_res:
                task_2 = "Task_two/knn/cosine/f1"
                task_3 = "Task_three/knn/cosine/f1"
                task_4 = "Task_four/knn/cosine/f1"

                task_mapper = {
                    "allen_score": ("Allen", [task_2]),
                    "hpa_score": ("HPA", [task_2, task_3]),
                    "cp_score": ("CP", [task_2, task_3, task_4]),
                }

                final_scores = {}
                # compute each group’s average in a loop
                for score_name, (chunk_name, tasks) in task_mapper.items():
                    vals = [cosine_res.get(f"{chunk_name}/{t}", 0) for t in tasks]
                    final_scores[f"score/{score_name}"] = sum(vals) / len(vals)

                allen_score = final_scores["score/allen_score"]
                hpa_score = final_scores["score/hpa_score"]
                cp_score = final_scores["score/cp_score"]

                final_scores["score/allen_hpa"] = (allen_score + hpa_score) / 2
                final_scores["score/final_score"] = (allen_score + hpa_score + cp_score) / 3

                self.logger.info(final_scores, sep="| ", padding_space=True)

            if self.cfg.eval.chammiv1.clean_up:
                ## remove folder_path
                shutil.rmtree(eval_cfg.feature_dir)
                self.logger.info(f"cleaned up {eval_cfg.feature_dir} after evaluation")
            return final_scores

    @torch.inference_mode()
    def eval_accuracy(self, output, y):
        pred = torch.argmax(output, dim=-1)
        correct = 100 * torch.sum(pred == y) / len(y)
        acc = correct.item()
        return acc

    @torch.inference_mode()
    def eval_regular(self, epoch: int, loader: DataLoader, eval_name: str) -> dict[str, float]:
        if eval_name in ["valid", "test"]:  ## evaluate on full channels, no need to keep track of channel ids
            channel_ids_list = None
        else:
            channel_ids_list = [self.channels_split[eval_name]]  ## e.g, test on the first 5 channels of JUMPCP

        self.model.eval()

        start_time = time.time()
        output_list = []
        gt_list = []
        for batch in tqdm(loader):
            images = batch["image"]
            labels = batch["label"]

            ## convert img to model type
            images = images.to(dtype=self.train_weight_dtype)
            res = self.model(x=images, channel_ids_list=channel_ids_list, valid_channel_masks=None)

            output_list.append(res["output"])
            gt_list.append(labels)

        output = torch.cat(output_list, dim=0)
        gt = torch.cat(gt_list, dim=0)
        accuracy = self.eval_accuracy(output, gt)

        ### make sure all GPUs wait here
        if self.use_ddp:
            barrier()
            acc_all_gpus = all_reduce_item(accuracy, "mean")
            print("------------- acc_all", acc_all_gpus)
        else:
            acc_all_gpus = accuracy

        self.logger.info(f"Done {eval_name} evaluation for epoch {epoch} in {(time.time() - start_time) / 60:.2f} minutes")

        if self.use_ddp:
            self.logger.info({f"acc_GPU{self.local_rank}/{eval_name}": accuracy})
        self.logger.info({f"acc/{eval_name}": acc_all_gpus})

        res[f"acc/{eval_name}"] = accuracy

        return res
