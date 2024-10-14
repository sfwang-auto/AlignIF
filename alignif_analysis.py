import math
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from utils import parse_args
from dataset import AlignIFDataset
from modules.models import get_model


def analysis(model, test_loader, device):
    model.eval()
    seqs, preds, masks, logits = [], [], [], []
    with torch.no_grad():
        for data in test_loader:
            data = [datum.to(device) for datum in data]
            pred, logit = model.infer(data)


def main():
    args = parse_args('alignif.yaml')

    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")

    if args.relax_label:
        name = "alignif_relax_label"
    else:
        name = "alignif"
    model = get_model(args, device, model_path=f'paras/{name}_best.h5')

    test_set = AlignIFDataset(args, 'test')
    test_loader = DataLoader(test_set, test_set, shuffle=False)

    # infer_without_msa(model, test_loader, device)
    analysis(model, test_loader, device)


if __name__ == "__main__":
    main()