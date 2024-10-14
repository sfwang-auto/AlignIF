import math
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from utils import parse_args
from dataset import AlignIFDataset
from modules.models import get_model


def cal_loss(seq, mask, logits):
    seq = seq * mask
    seq_onehot = nn.functional.one_hot(seq, 4).float()

    log_probs = F.log_softmax(logits, dim=-1)
    loss = -(seq_onehot * log_probs).sum(-1)
    return torch.sum(loss * mask) / torch.sum(mask)


def infer(model, test_loader, device):
    model.eval()
    seqs, preds, masks, logits = [], [], [], []
    with torch.no_grad():
        for data in test_loader:
            data = [datum.to(device) for datum in data]
            pred, logit = model.infer(data)
            seqs.append(data[0].seq)
            preds.append(pred)
            masks.append(data[0].mask)
            logits.append(logit)
        seqs = torch.cat(seqs)
        preds = torch.cat(preds)
        masks = torch.cat(masks)
        logits = torch.cat(logits, dim=0)
    
    acc = ((seqs == preds) * masks).sum() / masks.sum() * 100
    loss = cal_loss(seqs, masks, logits)

    print('acc: %.2f' % acc, 'perplexity: %.2f' % math.exp(loss))


def infer_without_msa(model, test_loader, device):
    model.eval()
    seqs, preds, masks, logits = [], [], [], []
    with torch.no_grad():
        for data in test_loader:
            data = [data[0].to(device)]
            pred, logit = model.infer(data)
            seqs.append(data[0].seq)
            preds.append(pred)
            masks.append(data[0].mask)
            logits.append(logit)
        seqs = torch.cat(seqs)
        preds = torch.cat(preds)
        masks = torch.cat(masks)
        logits = torch.cat(logits, dim=0)
    
    acc = ((seqs == preds) * masks).sum() / masks.sum() * 100
    loss = cal_loss(seqs, masks, logits)

    print('acc: %.2f' % acc, 'perplexity: %.2f' % math.exp(loss))


def main():
    args = parse_args('alignif.yaml')

    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")

    if args.relax_label:
        name = "alignif_relax_label"
    else:
        name = "alignif"
    model = get_model(args, device, model_path=f'paras/{name}_best.h5')

    test_set = AlignIFDataset(args, 'test')
    test_loader = DataLoader(test_set, args.bsz, shuffle=False)

    infer_without_msa(model, test_loader, device)
    infer(model, test_loader, device)
    


if __name__ == "__main__":
    main()