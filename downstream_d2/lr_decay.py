from typing import List, Dict, Set, Optional, Callable, Any
import torch
import copy

from detectron2.solver.build import reduce_param_groups


def lr_factor_func(para_name: str, is_resnet50, dec: float, debug=False) -> float:
    if dec == 0:
        dec = 1.
    
    N = 5 if is_resnet50 else 11
    if '.stem.' in para_name:
        layer_id = 0
    elif '.res' in para_name:
        ls = para_name.split('.res')[1].split('.')
        if ls[0].isnumeric() and ls[1].isnumeric():
            stage_id, block_id = int(ls[0]), int(ls[1])
            if stage_id == 2:  # res2
                layer_id = 1
            elif stage_id == 3:  # res3
                layer_id = 2
            elif stage_id == 4:  # res4
                layer_id = 3 + block_id // 3  # 3, 4  or  4, 5
            else:  # res5
                layer_id = N
        else:
            assert para_name.startswith('roi_heads.res5.norm.')
            layer_id = N + 1  # roi_heads.res5.norm.weight and roi_heads.res5.norm.bias of C4
    else:
        layer_id = N + 1
    
    exp = N + 1 - layer_id
    return f'{dec:g} ** {exp}' if debug else dec ** exp


# [modification] see: https://github.com/facebookresearch/detectron2/blob/v0.6/detectron2/solver/build.py#L134
# add the `lr_factor_func` to implement lr decay
def get_default_optimizer_params(
        model: torch.nn.Module,
        base_lr: Optional[float] = None,
        weight_decay: Optional[float] = None,
        weight_decay_norm: Optional[float] = None,
        bias_lr_factor: Optional[float] = 1.0,
        weight_decay_bias: Optional[float] = None,
        lr_factor_func: Optional[Callable] = None,
        overrides: Optional[Dict[str, Dict[str, float]]] = None,
) -> List[Dict[str, Any]]:
    """
    Get default param list for optimizer, with support for a few types of
    overrides. If no overrides needed, this is equivalent to `model.parameters()`.
    
    Args:
        base_lr: lr for every group by default. Can be omitted to use the one in optimizer.
        weight_decay: weight decay for every group by default. Can be omitted to use the one
            in optimizer.
        weight_decay_norm: override weight decay for params in normalization layers
        bias_lr_factor: multiplier of lr for bias parameters.
        weight_decay_bias: override weight decay for bias parameters.
        lr_factor_func: function to calculate lr decay rate by mapping the parameter names to
            corresponding lr decay rate. Note that setting this option requires
            also setting ``base_lr``.
        overrides: if not `None`, provides values for optimizer hyperparameters
            (LR, weight decay) for module parameters with a given name; e.g.
            ``{"embedding": {"lr": 0.01, "weight_decay": 0.1}}`` will set the LR and
            weight decay values for all module parameters named `embedding`.

    For common detection models, ``weight_decay_norm`` is the only option
    needed to be set. ``bias_lr_factor,weight_decay_bias`` are legacy settings
    from Detectron1 that are not found useful.

    Example:
    ::
        torch.optim.SGD(get_default_optimizer_params(model, weight_decay_norm=0),
                       lr=0.01, weight_decay=1e-4, momentum=0.9)
    """
    if overrides is None:
        overrides = {}
    defaults = {}
    if base_lr is not None:
        defaults["lr"] = base_lr
    if weight_decay is not None:
        defaults["weight_decay"] = weight_decay
    bias_overrides = {}
    if bias_lr_factor is not None and bias_lr_factor != 1.0:
        # NOTE: unlike Detectron v1, we now by default make bias hyperparameters
        # exactly the same as regular weights.
        if base_lr is None:
            raise ValueError("bias_lr_factor requires base_lr")
        bias_overrides["lr"] = base_lr * bias_lr_factor
    if weight_decay_bias is not None:
        bias_overrides["weight_decay"] = weight_decay_bias
    if len(bias_overrides):
        if "bias" in overrides:
            raise ValueError("Conflicting overrides for 'bias'")
        overrides["bias"] = bias_overrides
    if lr_factor_func is not None:
        if base_lr is None:
            raise ValueError("lr_factor_func requires base_lr")
    norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        # NaiveSyncBatchNorm inherits from BatchNorm2d
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
    )
    params: List[Dict[str, Any]] = []
    memo: Set[torch.nn.parameter.Parameter] = set()
    for module_name, module in model.named_modules():
        for module_param_name, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            
            hyperparams = copy.copy(defaults)
            if isinstance(module, norm_module_types) and weight_decay_norm is not None:
                hyperparams["weight_decay"] = weight_decay_norm
            if lr_factor_func is not None:
                hyperparams["lr"] *= lr_factor_func(f"{module_name}.{module_param_name}")
            
            hyperparams.update(overrides.get(module_param_name, {}))
            params.append({"params": [value], **hyperparams})
    return reduce_param_groups(params)
