from typing import Union
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
import time
import os
import wandb
from dotenv import load_dotenv

from utils import exists, get_machine_name


def init_wandb(cfg: DictConfig, run_id, project_name):
    # wandb.run.dir
    # https://docs.wandb.ai/guides/track/advanced/save-restore

    try:
        load_dotenv()
        os.environ["WANDB__SERVICE_WAIT"] = "300"
        wandb.login(key=os.getenv("WANDB_API_KEY"))
    except Exception as e:
        print(f"--- was trying to log in Weights and Biases... e={e}")

    ## run_name for wandb's run
    machine_name = get_machine_name()

    if cfg.logging.run_name is not None:
        run_name = cfg.logging.run_name
    else:
        run_name = "{}--{}--{}".format(machine_name, run_id, time.strftime("%I-%M%p-%B-%d-%Y"))
    wandb.init(
        project=project_name,
        entity="chammi",
        name=run_name,
        settings=wandb.Settings(start_method="fork"),
        notes=cfg.logging.get("notes"),
    )

    return run_name


class DummyLogger:
    def __init__(self, cfg, use_ddp: bool = False):
        self.cfg = cfg
        self.use_ddp = use_ddp

    def info(
        self,
        msg: Union[dict, str],
        sep=", ",
        padding_space=False,
        pref_msg: str = "",
        silent: bool = True,
        *args,
        **kwargs,
    ):
        if silent:
            return
        if isinstance(msg, dict):
            msg_str = pref_msg + " " + sep.join(f"{k} {round(v, 4) if isinstance(v, int) else v}" for k, v in msg.items())
            if padding_space:
                msg_str = sep + msg_str + " " + sep
            if self.use_ddp:
                msg_str = f'\t[Rank_{os.environ["LOCAL_RANK"]}]: {msg_str}'
            print(msg_str)
        else:
            if self.use_ddp:
                msg = f'\t[Rank_{os.environ["LOCAL_RANK"]}]: {msg}'
            print(msg)

    def log_imgs(self, *args, **kwargs):
        pass

    def log_config(self, *args, **kwargs):
        pass

    def update_best_result(self, *args, **kwargs):
        pass

    def finish(self, *args, **kwargs):
        pass


class MyWandBLogger:
    def __init__(self, cfg: DictConfig, project_name: str, run_id: str, use_ddp: bool = False, use_wandb: bool = True):
        self.args = cfg
        self.have_wandb = use_wandb
        if self.have_wandb:
            init_wandb(
                cfg,
                project_name=project_name,
                run_id=run_id,
            )
        self.use_ddp = use_ddp

    def info(
        self,
        msg: Union[dict, str],
        use_wandb=True,
        sep=", ",
        padding_space=False,
        pref_msg: str = "",
        print_out: bool = True,
        **kwargs,
    ):

        if isinstance(msg, dict):
            msg_str = pref_msg + " " + sep.join(f"{k} {round(v, 4) if isinstance(v, int) else v}" for k, v in msg.items())
            if padding_space:
                msg_str = sep + msg_str + " " + sep
            if self.use_ddp:
                msg_str = f'[Rank_{os.environ["LOCAL_RANK"]}]: {msg_str}'
            if use_wandb and self.have_wandb:
                wandb.log(msg, step=kwargs.get("step", None))
            if print_out:
                print(msg_str)

        else:
            if self.use_ddp:
                msg = f'[Rank_{os.environ["LOCAL_RANK"]}]: {msg}'
            if print_out:
                print(msg)

    def log_imgs(self, x, y, y_hat, classes, max_scores, name: str):
        columns = ["image", "pred", "label", "score", "correct"]
        data = []
        for j, image in enumerate(x, 0):
            # pil_image = Image.fromarray(image, mode="RGB")
            data.append(
                [
                    wandb.Image(image[:3]),
                    classes[y_hat[j].item()],
                    classes[y[j].item()],
                    max_scores[j].item(),
                    y_hat[j].item() == y[j].item(),
                ]
            )

        table = wandb.Table(data=data, columns=columns)
        wandb.log({name: table})

    def log_config(self, config: DictConfig):
        wandb.config.update(OmegaConf.to_container(config))  # , allow_val_change=True)

    def finish(
        self,
        msg_str: str = None,
    ):

        if exists(msg_str):
            print(msg_str)
        if self.have_wandb:
            wandb.finish()
