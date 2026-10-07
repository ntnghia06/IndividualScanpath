"""DataParallel device selection and portable checkpoint handling."""
import torch


def resolve_devices(device, gpu_ids, available_count):
    device = torch.device(device)
    if device.type != "cuda":
        if gpu_ids:
            raise ValueError("--gpu_ids requires a CUDA device")
        return device, []
    if available_count < 1:
        raise ValueError("CUDA was requested but no GPU is available")
    primary = device.index if device.index is not None else 0
    ids = ([primary] + [i for i in range(available_count) if i != primary]
           if gpu_ids is None else list(gpu_ids))
    if not ids or len(set(ids)) != len(ids) or any(i < 0 or i >= available_count for i in ids):
        raise ValueError(f"Invalid --gpu_ids {ids}; visible GPU count is {available_count}")
    if ids[0] != primary:
        raise ValueError("The first --gpu_ids entry must match --device (the primary GPU)")
    return torch.device("cuda", primary), ids


def wrap_model(network, device, gpu_ids):
    network = network.to(device)
    if len(gpu_ids) > 1:
        network = torch.nn.DataParallel(network, device_ids=gpu_ids, output_device=gpu_ids[0])
    return network


def unwrap_model(network):
    return network.module if isinstance(network, torch.nn.DataParallel) else network


def model_state_dict(network):
    return unwrap_model(network).state_dict()


def load_model_state(network, state):
    # Also accept checkpoints saved directly from an older DataParallel wrapper.
    if state and all(key.startswith("module.") for key in state):
        state = {key[len("module."):]: value for key, value in state.items()}
    unwrap_model(network).load_state_dict(state)


def restore_cuda_rng(states):
    # A checkpoint may be resumed with fewer visible GPUs than the original run.
    for index, state in enumerate(states[:torch.cuda.device_count()]):
        torch.cuda.set_rng_state(state.cpu(), device=index)
