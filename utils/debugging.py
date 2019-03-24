import torch

def tensor_contains_nan(tensor):
    """Check if a tensor contains at least one NaN."""
    if not isinstance(tensor, tuple):
        return (tensor != tensor).any()
    else:
        return any(tensor_contains_nan(x) for x in tensor)



def debug_forward_pre_hook(module, input):
    """Forward pre-hook for debugging."""
    message = "NaN detected in forward pre-hook of module {}".format(module)
    assert not tensor_contains_nan(input[0]), message


def debug_forward_hook(module, input, output):
    """Forward hook for debugging."""
    message = "NaN detected in forward hook of module {}".format(module)
    assert not (tensor_contains_nan(input[0]) or tensor_contains_nan(output)), message


def debug_backward_hook(module, grad_input, grad_output):
    """Backward hook for debugging."""
    message = "NaN detected in backward hook of module {}".format(module)
    assert not (tensor_contains_nan(grad_input[0]) or tensor_contains_nan(grad_output[0])), message


def apply_hooks(model):
    """Applies hooks recursively to each child module of a model."""
    for module in model.children():
        module.register_forward_pre_hook(debug_forward_pre_hook)
        module.register_forward_hook(debug_forward_hook)
        module.register_backward_hook(debug_backward_hook)