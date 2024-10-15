import torch
from torch_geometric.loader import DataLoader

from trainer import alignif_train
from dataset import AlignIFDataset
from modules.models import get_model
from utils import set_seed, parse_args


def main():
    set_seed()
    args = parse_args('alignif.yaml')
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    model = get_model(args, device)

    train_set = AlignIFDataset(args, 'train', read_all=True)
    val_set = AlignIFDataset(args, 'val', read_all=True)

    # train_set = AlignIFDataset(args, 'train', read_all=False)
    # val_set = AlignIFDataset(args, 'val', read_all=False)

    train_loader = DataLoader(train_set, args.bsz, shuffle=True)
    val_loader = DataLoader(val_set, args.bsz, shuffle=False)

    alignif_train(args, model, train_loader, val_loader, device)


if __name__ == "__main__":
    main()