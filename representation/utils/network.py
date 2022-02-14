import torch
from prettytable import PrettyTable
from pathlib import Path
from utils import logger


def check_grad(depth_model, feat_model, bad_grad):
    extreme_grad = False
    with torch.no_grad():
        for name, param in depth_model.named_parameters():
            if torch.max(param) > bad_grad or torch.min(param) < -bad_grad:
                logger.warning(
                    name + f" has extreme value {torch.max(param)}, {torch.min(param)}")
            if param.grad is not None:
                if torch.max(param.grad) > bad_grad or torch.min(param.grad) < -bad_grad:
                    logger.warning(
                        name + f" has extreme gradient {torch.max(param.grad)}, {torch.min(param.grad)}")
                    extreme_grad = True
                    return extreme_grad
        #
        for name, param in feat_model.named_parameters():
            if torch.max(param) > bad_grad or torch.min(param) < -bad_grad:
                logger.warning(
                    name + f" has extreme value {torch.max(param)}, {torch.min(param)}")
            if param.grad is not None:
                if torch.max(param.grad) > bad_grad or torch.min(param.grad) < -bad_grad:
                    logger.warning(
                        name + f" has extreme gradient {torch.max(param.grad)}, {torch.min(param.grad)}")
                    extreme_grad = True
                    return extreme_grad

    return extreme_grad


def init_net(net, type="kaiming", mode="fan_in", activation_mode="relu", distribution="normal", gpu_id=0):
    assert (torch.cuda.is_available())
    net = net.cuda(gpu_id)
    if type == "glorot":
        glorot_weight_zero_bias(net, distribution=distribution)
    else:
        kaiming_weight_zero_bias(
            net, mode=mode, activation_mode=activation_mode, distribution=distribution)
    return net


def glorot_weight_zero_bias(model, distribution="uniform"):
    """
    Initalize parameters of all modules
    by initializing weights with glorot  uniform/xavier initialization,
    and setting biases to zero.
    Weights from batch norm layers are set to 1.

    Parameters
    ----------
    model: Module
    distribution: string
    """
    for module in model.modules():
        if hasattr(module, 'weight'):
            if not ('Norm' in module.__class__.__name__):
                if distribution == "uniform":
                    torch.nn.init.xavier_uniform_(module.weight, gain=1)
                else:
                    torch.nn.init.xavier_normal_(module.weight, gain=1)
            else:
                torch.nn.init.constant_(module.weight, 1)
        if hasattr(module, 'bias'):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)


def kaiming_weight_zero_bias(model, mode="fan_in", activation_mode="relu", distribution="uniform"):
    if activation_mode == "leaky_relu":
        logger.error("Leaky relu is not supported yet")
        assert False

    ignore_list = list()
    for module in model.modules():
        if hasattr(module, 'weight'):
            if (not ('BatchNorm' in module.__class__.__name__)) and (not ('GroupNorm' in module.__class__.__name__)):
                if distribution == "uniform":
                    torch.nn.init.kaiming_uniform_(
                        module.weight, mode=mode, nonlinearity=activation_mode)
                else:
                    torch.nn.init.kaiming_normal_(
                        module.weight, mode=mode, nonlinearity=activation_mode)
            else:
                torch.nn.init.constant_(module.weight, 1)
        if hasattr(module, 'bias'):
            if module.bias is not None:
                # Changed the bias value to random uniform distribution between -1 to 1
                # torch.nn.init.constant_(module.bias, 0)
                torch.nn.init.uniform_(module.bias, a=-1.0, b=1.0)

        if not hasattr(module, 'weight') and not hasattr(module, 'bias'):
            ignore_list.append(module)


def save_checkpoint(epoch, step, model, path):
    state = {
        'epoch': epoch,
        'step': step,
        'model': model.state_dict(),
    }
    torch.save(state, str(path))


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    logger.info(table)
    logger.info(f"Total Trainable Params: {total_params}")
    return total_params


def load_model(model, trained_model_path: Path, partial_load: bool):
    ignore_list = list()
    if not trained_model_path.exists():
        raise IOError("No pre-trained model detected")

    if partial_load:
        pre_trained_state = torch.load(str(trained_model_path))
        epoch = pre_trained_state['epoch'] + 1
        step = pre_trained_state['step']

        model_state = model.state_dict()
        if "model" in pre_trained_state:
            pre_trained_state = pre_trained_state["model"]
        else:
            raise IOError("no state dict found")

        trained_model_state = dict()
        for k, v in pre_trained_state.items():
            shape = pre_trained_state[k].shape
            if k in model_state:
                if model_state[k].shape == shape:
                    trained_model_state[k] = v
                else:
                    ignore_list.append(k)
            else:
                ignore_list.append(k)

        logger.info(
            f"Loading {len(trained_model_state.items())} chunk of parameters to the model to be trained which has "
            f"{len(model_state.items())}")
        model_state.update(trained_model_state)
        model.load_state_dict(model_state)

    else:
        logger.info("Loading {:s} ...".format(str(trained_model_path)))
        state = torch.load(str(trained_model_path))
        step = state['step']
        epoch = state['epoch'] + 1
        model.load_state_dict(state["model"])
        logger.info('Restored model, epoch {}, step {}'.format(epoch, step))

    return ignore_list, epoch, step
