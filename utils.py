import os
from typing import Callable, Union, Any, Optional
import torch
import gc
from datetime import datetime
import shutil
from pathlib import Path
import functools
import time
import pathlib
import numpy as np
import pprint
import json
import getpass
import pandas as pd
import torch.distributed as dist
from lxml import etree
from contextlib import contextmanager


def exists(val):
    return val is not None


def default(val, default):
    return val if exists(val) else default


def get_machine_name():
    import socket

    machine_name = socket.gethostname()
    return machine_name


def running_on_server(verbose=True):
    MY_MACBOOK = "chaumac.local"

    machine_name = get_machine_name()
    if verbose:
        print(f"running on {machine_name}")
    return machine_name != MY_MACBOOK


def get_memory_statistics(precision: int = 3) -> dict[str, Any]:
    memory_allocated = None
    memory_reserved = None  # includes allocated + cached
    peak_memory_allocated = None
    peak_memory_reserved = None
    total_memory = None
    free_memory = None

    if torch.cuda.is_available():
        device = torch.cuda.current_device()

        # current usage
        memory_allocated = torch.cuda.memory_allocated(device)
        memory_reserved = torch.cuda.memory_reserved(device)

        # peaks
        peak_memory_allocated = torch.cuda.max_memory_allocated(device)
        peak_memory_reserved = torch.cuda.max_memory_reserved(device)

        # device capacity and free at OS level
        props = torch.cuda.get_device_properties(device)
        total_memory = props.total_memory
        free_memory = total_memory - memory_reserved

    elif torch.mps.is_available():
        memory_allocated = torch.mps.current_allocated_memory()
        # no direct API for total/free on MPS
        peak_memory_allocated = None
        peak_memory_reserved = None

    else:
        print("No CUDA, MPS, or ROCm device found. Memory statistics are not available.")

    return {
        "memory_allocated": round(bytes_to_gigabytes(memory_allocated), ndigits=precision),
        "memory_reserved": round(bytes_to_gigabytes(memory_reserved), ndigits=precision),
        "peak_memory_allocated": round(bytes_to_gigabytes(peak_memory_allocated), ndigits=precision),
        "peak_memory_reserved": round(bytes_to_gigabytes(peak_memory_reserved), ndigits=precision),
        "total_memory": round(bytes_to_gigabytes(total_memory), ndigits=precision),
        "free_memory": round(bytes_to_gigabytes(free_memory), ndigits=precision),
    }


def bytes_to_gigabytes(x: int) -> float:
    if x is not None:
        return x / 1024**3


def free_memory() -> None:
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def unload_model(model):
    model.to("cpu")


def make_contiguous(
    x: Union[torch.Tensor, dict[str, torch.Tensor]],
) -> Union[torch.Tensor, dict[str, torch.Tensor]]:
    if isinstance(x, torch.Tensor):
        return x.contiguous()
    elif isinstance(x, dict):
        return {k: make_contiguous(v) for k, v in x.items()}
    else:
        return x


def find_files(dir: Union[str, Path], prefix: str = "checkpoint") -> list[str]:
    if not isinstance(dir, Path):
        dir = Path(dir)
    if not dir.exists():
        return []
    checkpoints = os.listdir(dir.as_posix())
    checkpoints = [c for c in checkpoints if c.startswith(prefix)]
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
    checkpoints = [dir / c for c in checkpoints]
    return checkpoints


def delete_files(dirs: Union[str, list[str], Path, list[Path]], logger) -> None:
    if not isinstance(dirs, list):
        dirs = [dirs]
    dirs = [Path(d) if isinstance(d, str) else d for d in dirs]
    logger.info(f"Deleting files: {dirs}")
    for dir in dirs:
        if not dir.exists():
            continue
        shutil.rmtree(dir, ignore_errors=True)


def string_to_filename(s: str) -> str:
    return (
        s.replace(" ", "-")
        .replace("/", "-")
        .replace(":", "-")
        .replace(".", "-")
        .replace(",", "-")
        .replace(";", "-")
        .replace("!", "-")
        .replace("?", "-")
    )


def get_latest_ckpt_path_to_resume_from(
    resume_from_checkpoint: str | None, num_update_steps_per_epoch: int, logger: Callable = print
) -> tuple[str | None, int, int, int]:
    if resume_from_checkpoint is None:
        initial_global_step = 0
        global_step = 0
        first_epoch = 0
        resume_from_checkpoint_path = None
    else:
        resume_from_checkpoint_path = Path(resume_from_checkpoint)

        # if given a file, check if it exists
        if not resume_from_checkpoint_path.exists():
            logger.info(f">>>> Checkpoint '{resume_from_checkpoint}' does not exist. Starting a new training run.")
            initial_global_step = 0
            global_step = 0
            first_epoch = 0
            resume_from_checkpoint_path = None
        ## if given folder, find the latest checkpoint
        elif resume_from_checkpoint_path.is_dir() and "checkpoint" not in resume_from_checkpoint_path.name:
            checkpoints = find_files(resume_from_checkpoint_path, prefix="checkpoint")
            if len(checkpoints) == 0:
                logger.info(f">>>> No checkpoint found in {resume_from_checkpoint_path}, starting a new training run.")
                initial_global_step = 0
                global_step = 0
                first_epoch = 0
                resume_from_checkpoint_path = None
            else:
                resume_from_checkpoint_path = checkpoints[-1]  # latest checkpoint
                logger.info(f">>>>++++ Resuming from latest checkpoint {resume_from_checkpoint_path}")
                first_epoch = int(resume_from_checkpoint_path.name.split("-")[1])
                global_step = first_epoch * num_update_steps_per_epoch
                initial_global_step = global_step
        else:
            logger.info(f">>>> Resuming from checkpoint {resume_from_checkpoint}")
            first_epoch = int(resume_from_checkpoint_path.name.split("-")[1])
            global_step = first_epoch * num_update_steps_per_epoch
            initial_global_step = global_step

    return resume_from_checkpoint_path, initial_global_step, global_step, first_epoch


def get_intermediate_ckpt_path(checkpointing_limit: int, epoch: int, output_dir: str, logger) -> str:
    # before saving state, check if this save would set us over the `checkpointing_limit`
    if checkpointing_limit is not None:
        checkpoints = find_files(output_dir, prefix="checkpoint")

        # before we save the new checkpoint, we need to have at_most `checkpoints_total_limit - 1` checkpoints
        if len(checkpoints) >= checkpointing_limit:
            num_to_remove = len(checkpoints) - checkpointing_limit + 1
            checkpoints_to_remove = checkpoints[0:num_to_remove]
            delete_files(checkpoints_to_remove, logger=logger)

    logger.info(f"Checkpointing at epoch {epoch}")
    save_path = os.path.join(output_dir, f"checkpoint-{epoch}")
    logger.info(f"Saving state to {save_path}")
    return save_path


def datetime_now(time_format: str = None) -> str:

    # time_format = default(time_format, "%Y-%b-%d %H:%M:%S.%f")
    time_format = default(time_format, "%Y-%b-%d %H:%M:%S")
    return datetime.now().strftime(time_format)


def get_gpu_mem(cuda="cuda:0", return_total_mem=False):
    free, total = torch.cuda.mem_get_info(device=cuda)
    free_gb, total_gb = free / 1024**3, total / 1024**3
    use_gb = total_gb - free_gb
    out = f"used/avail mem: {use_gb:.1f}/{total_gb:.1f} GB"
    if return_total_mem:
        return total_gb
    else:
        return out


def get_gpu_mem_all() -> None:
    ## get all gpu available
    n_gpus = torch.cuda.device_count()
    for i in range(n_gpus):
        free_gb = get_gpu_mem(cuda=f"cuda:{i}")
        print(f"\tdevice: {i+1}/{n_gpus}, avail mem: {free_gb}GB")


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


# def gpu_mem_report_details():
#     import humanize, psutil, GPUtil

#     print("CPU RAM Free: " + humanize.naturalsize(psutil.virtual_memory().available))
#     gpu_list = GPUtil.getGPUs()
#     for i, gpu in enumerate(gpu_list):
#         print(
#             "GPU {:d} ... Mem Used: {:.0f}MB\t Free: {:.0f}MB / {:.0f}MB | Utilization {:3.0f}%".format(
#                 i,
#                 gpu.memoryTotal - gpu.memoryFree,
#                 gpu.memoryFree,
#                 gpu.memoryTotal,
#                 gpu.memoryUtil * 100,
#             )
#         )


# def gpu_mem_report(device: Union[int, list, torch.device, None] = None, msg=None):
#     def get_mem_msg(cuda):
#         free, total = torch.cuda.mem_get_info(device=cuda)
#         free_gb, total_gb = free / 1024**3, total / 1024**3
#         used_gb = total_gb - free_gb
#         msg_1 = f"Device {cuda} - {torch.cuda.get_device_name(cuda)}"
#         msg_2 = f"Mem used: {used_gb:.2f} GB; free/total: {free_gb:.2f}/{total_gb:.2f} GB\n"
#         return msg_1, msg_2

#     def ensure_list(x: Union[int, list]):
#         if isinstance(x, list):
#             return x
#         else:
#             return [x]

#     if not torch.cuda.is_available():  # skip if gpu is not available
#         return None
#     if device is None:
#         device = range(torch.cuda.device_count())
#     else:
#         device = ensure_list(device)

#     if msg is not None:
#         print(msg)

#     for cuda in device:
#         msg_1, msg_2 = get_mem_msg(cuda)
#         print(msg_1, "\n", msg_2, "------")


# def move_to_cuda(sample, device):
#     def _move_to_cuda(tensor):
#         return tensor.to(device)

#     return apply_to_sample(_move_to_cuda, sample)


# def apply_to_sample(f, sample):
#     if len(sample) == 0:
#         return {}

#     def _apply(x):
#         if torch.is_tensor(x):
#             return f(x)
#         elif isinstance(x, dict):
#             return {key: _apply(value) for key, value in x.items()}
#         elif isinstance(x, list):
#             return [_apply(x) for x in x]
#         else:
#             return x

#     return _apply(sample)


# ############## Model, gradient ##############


# def save_pytorch_model(path, model, epoch, optimizer=None):
#     model_dict = {"epoch": epoch, "model_state": model.state_dict()}
#     if optimizer is not None:
#         model_dict["optimizer_state"] = optimizer.state_dict()

#     torch.save(model_dict, path)


# def set_requires_grad(model: nn.Module, val: bool):
#     for p in model.parameters():
#         p.requires_grad = val


def analyze_model(model, print_trainable=True, print_model=False, print_non_trainable=True, logger=print):
    pp = pprint.PrettyPrinter(indent=4)
    if print_model:
        logger("--------------------")
        pp.pprint(list(model.state_dict().keys()))
        logger("--------------------")

    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if print_trainable:
        logger(f"Trainable parameters: {trainable_num:,}")
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger(f"{name}, {param.shape}, {param.numel()}, {param.requires_grad}")

    if print_non_trainable:
        logger(f"Non-trainable parameters: {total_num - trainable_num:,}")
        for name, param in model.named_parameters():
            if not param.requires_grad:
                logger(f"{name}, {param.shape}, {param.numel():,}, {param.requires_grad}")

    return total_num, trainable_num


# #################### Write, Read files ###############################


# def write_hdf5(output_path, data, data_id, mode, dtype="uint8"):
#     try:
#         with h5py.File(output_path, mode) as hf:
#             hf.create_dataset(data_id, data=data, dtype=dtype, compression="gzip")
#     except ValueError:
#         print("file exists, skipped.")


# def read_hdf5(data_path, data_id=None):
#     with h5py.File(data_path, "r") as hf:
#         data = hf.get(data_id)
#         if data is not None:
#             return data[:]
#         else:
#             return None


# def write_json(file_path, my_dict, cls=None):
#     with open(file_path, "w") as fp:
#         json.dump(my_dict, fp, cls=cls)
#     return None


# def read_json(filename):
#     with open(filename, encoding="utf8") as fr:
#         return json.load(fr)


# def read_yaml(file_path):
#     with open(file_path, "r") as f:
#         return yaml.safe_load(f)


# def read_dill(path):
#     with open(path, "rb") as f:
#         generator = dill.load(f)
#     print(f"Done reading {path}!")

#     return generator


# def write_dill(output_path, obj):
#     ## Write to file
#     if output_path is not None:
#         folder_path = os.path.dirname(output_path)  # _output folder
#         pathlib.Path(folder_path).mkdir(parents=True, exist_ok=True)  # create the folder(s) recursively if does not exist
#         with open(output_path, "wb") as f:
#             dill.dump(obj, f)
#     print(f"Done writing {output_path}!")
#     return True


def mkdir(path, mode=0o700):
    pathlib.Path(path).mkdir(mode=mode, parents=True, exist_ok=True)


def write_numpy(x: np.ndarray, output_path: str):
    np.save(output_path, x)
    print(f"Done writing {output_path}!")
    return True


# ######################## Timers ########################
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        :param val:
        :param n:
        :return:
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def time_this_function(func):
    @functools.wraps(func)
    def wrap_func(*args, **kwargs):
        t_start = time.time()
        result = func(*args, **kwargs)
        t_stop = time.time()
        minutes = (t_stop - t_start) / 60
        print(f'Function "{func.__name__}" executed in {minutes:.4f} minutes')
        return result

    return wrap_func


def convert_secs2time(epoch_time_in_sec: int, return_string=True) -> Union[str, tuple[int, int, int]]:
    """return hour_min_second in string message format, or (hour, min, second)"""
    now = datetime_now(time_format="%Y-%b-%d %H:%M:%S")

    need_hour = int(epoch_time_in_sec / 3600)
    need_mins = int((epoch_time_in_sec - 3600 * need_hour) / 60)
    need_secs = int(epoch_time_in_sec - 3600 * need_hour - 60 * need_mins)
    need_time = "[{}]  Need [hh:mm:ss] {:02d}:{:02d}:{:02d}".format(now, need_hour, need_mins, need_secs)

    if return_string:
        return need_time
    else:
        return need_hour, need_mins, need_secs


def read_json(filename):
    with open(filename, encoding="utf8") as fr:
        return json.load(fr)


def get_all_running_jobs(return_jobids_only=False):
    """
    Get all job ids from the current user.
    """
    cur_user = getpass.getuser()
    xml_data = os.popen(f"qstat -u {cur_user} -xml").read()
    root = etree.fromstring(xml_data)
    job_data = [{child.tag: child.text for child in job_list} for job_list in root.xpath("//job_list")]
    data = pd.DataFrame(job_data)
    if "JB_owner" in data.columns:
        data = data.drop(columns=["JB_owner"])
    data["JB_job_number"] = pd.to_numeric(data["JB_job_number"], errors="coerce")
    if return_jobids_only:
        data = data["JB_job_number"].dropna().astype(int).values

    return data


def compute_grad_norm(params, norm_type=2):
    # collect all existing grads
    grads = [p.grad.detach().view(-1) for p in params if p.grad is not None]
    if not grads:
        return 0.0
    all_grads = torch.cat(grads)  # one long 1-D tensor
    return all_grads.norm(norm_type).item()  # single norm call


def init_distributed(cuda):
    """
    Initializes distributed backend.

    :param cuda: (bool) if True initializes nccl backend, if False initializes
        gloo backend
    """
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    distributed = world_size > 1
    if distributed:
        backend = "nccl" if cuda else "gloo"
        torch.distributed.init_process_group(backend=backend, init_method="env://")
        assert torch.distributed.is_initialized()
    return distributed


def barrier():
    """
    Call torch.distributed.barrier() if distritubed is in use
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def get_rank():
    """
    Gets distributed rank or returns zero if distributed is not initialized.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    return rank


def get_world_size():
    """
    Gets total number of distributed workers or returns one if distributed is
    not initialized.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
    else:
        world_size = 1
    return world_size


def all_reduce_item(value, op="sum"):
    """
    All-reduces single scalar value if distributed is in use
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if op == "sum" or op == "mean":
            dop = torch.distributed.ReduceOp.SUM
        elif op == "min":
            dop = torch.distributed.ReduceOp.MIN
        elif op == "max":
            dop = torch.distributed.ReduceOp.MAX
        elif op == "product":
            dop = torch.distributed.ReduceOp.PRODUCT
        else:
            raise RuntimeError("Unsupported reduce op")

        backend = torch.distributed.get_backend()
        if backend == torch.distributed.Backend.NCCL:
            device = torch.device("cuda")
        elif backend == torch.distributed.Backend.GLOO:
            device = torch.device("cpu")
        else:
            raise RuntimeError("Unsupported distributed backend")

        tensor = torch.tensor(value, device=device)
        torch.distributed.all_reduce(tensor, dop)
        if op == "mean":
            tensor /= get_world_size()
        ret = tensor.item()
    else:
        ret = value
    return ret


def all_reduce_tensor(value, op="sum"):
    """
    All-reduces single scalar value if distributed is in use
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if op == "sum" or op == "mean":
            dop = torch.distributed.ReduceOp.SUM
        elif op == "min":
            dop = torch.distributed.ReduceOp.MIN
        elif op == "max":
            dop = torch.distributed.ReduceOp.MAX
        elif op == "product":
            dop = torch.distributed.ReduceOp.PRODUCT
        else:
            raise RuntimeError("Unsupported reduce op")

        backend = torch.distributed.get_backend()
        if backend == torch.distributed.Backend.NCCL:
            device = torch.device("cuda")
        elif backend == torch.distributed.Backend.GLOO:
            device = torch.device("cpu")
        else:
            raise RuntimeError("Unsupported distributed backend")

        tensor = value
        torch.distributed.all_reduce(tensor, dop)
        if op == "mean":
            tensor /= get_world_size()
        ret = tensor
    else:
        ret = value
    return ret


@contextmanager
def sync_workers():
    """
    Yields distributed rank and synchronizes all workers on exit.
    """
    rank = get_rank()
    yield rank
    barrier()


# def compute_grad_norm2(model):
#     total_norm = 0.0
#     for p in model.parameters():
#         if p.grad is not None:
#             param_norm = p.grad.data.norm(2)
#             total_norm += param_norm.item() ** 2
#     total_norm = total_norm**0.5
#     print(f"grad norm: {total_norm:.4f}")
#     return total_norm


if __name__ == "__main__":
    # Example usage
    running_jobs = get_all_running_jobs(return_jobids_only=True)
    print(running_jobs)
