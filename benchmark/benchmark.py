import inspect
from logging import getLogger
from posixpath import dirname
from time import time
from typing import Any, Callable, Union

import numpy as np
import torch
import yaml
from tqdm import tqdm

from .format import format_num, format_time
from .machine_info import get_machine_info
import os

logger = getLogger("torch-benchmark")
_INVALID = float("nan")


def _is_valid(val):
    return val == val


def get_call_arg_names(module_or_fn):
    if isinstance(module_or_fn, torch.nn.Module):
        return inspect.getfullargspec(module_or_fn.forward)[0][1:]
    return inspect.getfullargspec(module_or_fn)[0]


def get_device(model):
    return next(model.parameters()).device


def generate_input_sample(input_shape: Union[list, tuple], mode) -> Union[torch.Tensor, np.array]:
    if mode == 'torch':
        return torch.randn(*input_shape, dtype=torch.float32).to('cpu')
    else:
        return np.random.rand(*input_shape).astype('float32')


def measure_params(model):
    num_params = _INVALID

    try:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    except AttributeError as e:
        logger.error(f"Unable to measure model params due to error: {e}")

    return num_params


def measure_allocated_memory(
    model,
    sample,
    model_device,
    transfer_to_device_fn=torch.Tensor.to,
    print_details=False,
):
    assert model_device.type == "cuda"

    torch.cuda.reset_peak_memory_stats(device=model_device)
    pre_mem = torch.cuda.memory_allocated(device=model_device)

    transfer_to_device_fn(
        model(transfer_to_device_fn(sample, model_device)),
        "cpu",
    )

    if print_details:
        logger.info(torch.cuda.memory_summary(device=model_device, abbreviated=True))

    post_mem = torch.cuda.memory_allocated(device=model_device)
    max_mem = torch.cuda.max_memory_allocated(device=model_device)

    return pre_mem, post_mem, max_mem


def warm_up(
    model,
    input_shape: Union[list, tuple],
    model_device,
    transfer_to_device_fn=torch.Tensor.to,
    generate_input_sample_fn=generate_input_sample,
    num_runs: int = 100,
    mode: str = torch
):
    for _ in tqdm(range(num_runs), desc=f"Warming up with batch_size={input_shape[0]}"):    
        sample = generate_input_sample_fn(input_shape, mode=mode)

        if mode == 'torch':
            transfer_to_device_fn(
                model(transfer_to_device_fn(sample, model_device)),
                "cpu",
            )
        else:
            model(sample)



def measure_detailed_inference_timing(
    model, sample, model_device, transfer_to_device_fn=torch.Tensor.to
):

    try:
        with torch.autograd.profiler.profile(
            use_cuda=(model_device.type == "cuda"), profile_memory=True
        ) as prof:
            transfer_to_device_fn(
                model(transfer_to_device_fn(sample, model_device)),
                "cpu",
            )

        detailed_timing = prof.key_averages().table(sort_by="self_cpu_time_total")
        logger.info(detailed_timing)

    except Exception as e:
        logger.error(
            f"Caught exception while attempting to measure detailed model inference: {e}"
        )


def measure_repeated_inference_timing(
    model,
    input_shape: Union[list, tuple],
    model_device: Union[None, torch.device],
    transfer_to_device_fn=torch.Tensor.to,
    generate_input_sample_fn=generate_input_sample,
    num_runs: int = 100,
    mode: str = 'torch'
):
    if model_device is None:
        assert mode != 'torch', "Must specified model device when inference using pytorch"

    t_c2d = []
    t_inf = []
    t_d2c = []
    t_tot = []

    for _ in tqdm(
        range(num_runs), desc=f"Measuring inference for batch_size={input_shape[0]}"
    ):

        sample = generate_input_sample_fn(input_shape, mode=mode)

        if mode == 'torch':
            start_on_cpu = time()
            device_sample = transfer_to_device_fn(sample, model_device)
            start_on_device = time()
            device_result = model(device_sample)
            stop_on_device = time()
            transfer_to_device_fn(device_result, "cpu")
            stop_on_cpu = time()

            t_c2d.append(start_on_device - start_on_cpu)
            t_inf.append(stop_on_device - start_on_device)
            t_d2c.append(stop_on_cpu - stop_on_device)
            t_tot.append(stop_on_cpu - start_on_cpu)
        else:
            start_on_cpu = time()
            model(sample)
            stop_on_cpu = time()

            t_tot.append(stop_on_cpu - start_on_cpu)

    results_dict = {}

    if mode == 'torch':
        times_and_titles = [(t_inf, "on_device_inference")]
        if model_device.type == "cuda":
            times_and_titles.extend(
                [
                    (t_c2d, "cpu_to_gpu"),
                    (t_d2c, "gpu_to_cpu"),
                    (t_tot, "total"),
                ]
            )
    else:
        times_and_titles = [(t_tot, "total")]


    for s_per_batch, title in times_and_titles:
        s_per_batch = np.array(s_per_batch)
        batches_per_s = 1 / s_per_batch

        metrics = {
            "batches_per_second_mean": float(batches_per_s.mean()),
            "batches_per_second_std": float(batches_per_s.std()),
            "batches_per_second_min": float(batches_per_s.min()),
            "batches_per_second_max": float(batches_per_s.max()),
            "seconds_per_batch_mean": float(s_per_batch.mean()),
            "seconds_per_batch_std": float(s_per_batch.std()),
            "seconds_per_batch_min": float(s_per_batch.min()),
            "seconds_per_batch_max": float(s_per_batch.max()),
        }

        human_readable = {
            "batches_per_second": f"{format_num(batches_per_s.mean())} +/- {format_num(batches_per_s.std())} [{format_num(batches_per_s.min())}, {format_num(batches_per_s.max())}]",
            "batch_latency": f"{format_time(s_per_batch.mean())} +/- {format_time(s_per_batch.std())} [{format_time(s_per_batch.min())}, {format_time(s_per_batch.max())}]",
        }

        results_dict[title] = {"metrics": metrics, "human_readable": human_readable}

    return results_dict


def measure_energy(
    model,
    input_shape: Union[list, tuple],
    model_device,
    transfer_to_device_fn=torch.Tensor.to,
    generate_input_sample_fn=generate_input_sample,
    num_runs=100,
    include_transfer_costs=True,
    print_fn=logger.info,
    mode: str = 'torch'
):
    inference_joules = _INVALID

    def test_with_transfer():
        nonlocal model, sample
        transfer_to_device_fn(
            model(transfer_to_device_fn(sample, model_device)),
            "cpu",
        )

    def test_without_transfer():
        nonlocal model, sample
        model(sample)

    def test_onnx():
        nonlocal model, sample
        model(sample)

    if mode == 'torch':
        if include_transfer_costs:
            test_fn = test_with_transfer
        else:
            test_fn = test_without_transfer
            sample = sample.to(model_device)
    else:
        test_fn = test_onnx


    try:
        from .jetson_power import PowerEstimator

        p_est = PowerEstimator(
            print_fn=print_fn,
            idle_load_duration=10,
            idle_load_samples=100
        )
        # index 0 is total energy, index 1 is energy over idle consumption:
        meas = []
        for _ in tqdm(
            range(num_runs), desc=f"Measuring energy for batch_size={input_shape[0]}"
        ):
            sample = generate_input_sample_fn(input_shape, mode=mode)
            meas.append(p_est.estimate_fn_power(test_fn)[0] / 1000)
        
        inference_joules = float(np.array(meas).mean())
    except Exception:
        pass

    if not _is_valid(inference_joules):
        logger.error(
            "Unable to measure energy consumption. Device must be a NVIDIA Jetson."
        )

    return inference_joules


def fmt(d: dict):
    return yaml.dump(d)

def save_benchmark_results(d: dict) -> None:
    dir = os.getcwd()
    logger.info(
        f"Saving benchmark result to {dir}/benchmark.yaml"
    )
    with open(f'{dir}/benchmark.yaml', mode='w') as f:
        yaml.dump(d, f)


def benchmark(
    model: torch.nn.Module,
    input_shape: Union[list, tuple],
    num_runs: int = 100,
    warm_up_iter: int = 10,
    mode: str = 'torch',
    get_device_fn: Callable[[Any], torch.device] = get_device,
    transfer_to_device_fn=torch.Tensor.to,
    generate_input_sample_fn=generate_input_sample,
    print_fn=logger.info,
    warm_up_fn=warm_up,
    get_machine_info: str = False
):
    results = {}

    batch_size = input_shape[0]

    prevously_training = getattr(model, "training", False)
    if hasattr(model, "eval"):
        model.eval()

    if get_machine_info:
        # Get machine info
        machine_info = get_machine_info()
        results["machine_info"] = machine_info
        print_fn(fmt({"Machine info": machine_info}))


    if mode == 'onnx' or mode == 'tensorrt':
        model_device = None
    else:
        model_device = get_device_fn(model)
        assert isinstance(
            model_device, torch.device
        ), "model_device should be a `torch.device`"
        results["device"] = model_device.type
        print_fn(f"Model device: {model_device}")


    # Measure inference timing
    memory = {}
    timing = {}
    energy = {}
    with torch.no_grad():
        # Measure Allocated Memory
        # if model_device.type == "cuda":
        #     pre_mem, post_mem, max_mem = measure_allocated_memory(
        #         model, sample, model_device, transfer_to_device_fn, print_details
        #     )
        #     memory[f"batch_size_{batch_size}"] = {
        #         "pre_inference_bytes": pre_mem,
        #         "max_inference_bytes": max_mem,
        #         "post_inference_bytes": post_mem,
        #         "pre_inference": format_num(pre_mem, bytes=True),
        #         "max_inference": format_num(max_mem, bytes=True),
        #         "post_inference": format_num(post_mem, bytes=True),
        #     }
        # else:
        #     logger.warning(
        #         "Measurement of allocated memory is only available on CUDA devices"
        #     )

        # Inference timing
        warm_up_fn(
            model,
            input_shape,
            model_device,
            transfer_to_device_fn,
            generate_input_sample_fn,
            num_runs=warm_up_iter,
            mode=mode
        )

        timing[f"batch_size_{batch_size}"] = measure_repeated_inference_timing(
            model,
            input_shape,
            model_device,
            transfer_to_device_fn,
            generate_input_sample_fn,
            num_runs,
            mode=mode
        )

        # Energy measurement
        energy_joules = measure_energy(
            model,
            input_shape,
            model_device,
            transfer_to_device_fn,
            generate_input_sample_fn,
            num_runs=num_runs,
            include_transfer_costs=True,
            print_fn=print_fn,
            mode=mode
        )
        if _is_valid(energy_joules):
            energy_kwh = energy_joules / 3.6e6
            energy[f"batch_size_{batch_size}"] = {
                "joules": energy_joules,
                "kWh": energy_kwh,
            }

    results["timing"] = timing
    if memory:
        results["memory"] = memory
    if energy:
        results["energy"] = energy

    save_benchmark_results(results)
    print_fn(fmt(results))

    if prevously_training:
        model.train()

    return results
