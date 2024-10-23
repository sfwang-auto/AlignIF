import torch
from modules.baseline import BaseLine
from modules.alignif import AlignIF
from modules.mixupif import MixUpIF


def get_model(args, device, model_path=''):
    model_name = args.model_name
    if model_name == 'baseline':
        model = BaseLine(args)
    elif model_name == 'alignif':
        model = AlignIF(args)
    elif model_name == 'mixupif':
        model = MixUpIF(args)
    model = model.to(device)
    # model.model_name = model_name
    
    print('Number of parameters: {}'.format(sum([p.numel() for p in model.parameters()])))
    
    if model_path != '':
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    return model