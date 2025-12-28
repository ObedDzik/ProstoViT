import logging
import torch

def create_model(name, checkpoint=None, **kwargs):
    model_entrypoint = global_registry.get_entrypoint("model", name)

    # If the model has a pretrained_cfg parameter, use it
    if checkpoint and 'checkpoint' in inspect.signature(model_entrypoint).parameters: 
        kwargs['checkpoint'] = checkpoint
        checkpoint = None
    model = model_entrypoint(**kwargs)

    if checkpoint:
        logging.info(f"Loading checkpoint {checkpoint}")
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if 'model' in state_dict:
            state_dict = state_dict['model']
        message = model.load_state_dict(state_dict, strict=False)
        logging.info(f"Load message: {message}")

    model.model_kwargs = kwargs
    model.name = name
    model._medAI_registry_config = {
        "name": name, 
        **kwargs
    }

    return model