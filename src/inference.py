import os
import torch
import argparse
from utils import set_seed
from alignif import AlignIF
from data_process import process


NUM_TO_LETTER = {0: 'A', 1: 'G', 2: 'C', 3: 'U', 4: '-'}
LETTER_TO_NUM = {'A': 0, 'G': 1, 'C': 2, 'U': 3, '-': 4}


def main(args):
    set_seed()
    device = torch.device("cuda:0")

    data = process(args, device)
    model = AlignIF().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    
    subseq = None
    if args.subseq is not None:
        if len(args.subseq) != data['num_nodes']:
            print(f"Sub-sequence length {len(subseq)} does not match the target structure length {data['num_nodes']}.")
        else:
            subseq = torch.tensor([LETTER_TO_NUM[res] for res in args.subseq], dtype=torch.int64, device=device)
    samples, logits = model.sample(
        data, n_samples=args.n_samples, temperature=args.temperature, subseq=subseq
    )

    fasta = ""
    mask = data['node_mask'][0]
    probs = torch.softmax(logits, dim=-1)
    for i in range(args.n_samples):
        probs_i = probs[i]
        sample_i = samples[i]

        real_sample = torch.ones(mask.shape, dtype=torch.int64, device=device) * 4
        real_sample[mask] = sample_i
        seq = "".join([NUM_TO_LETTER[token.item()] for token in real_sample])
        log_prob = probs_i[torch.arange(sample_i.shape[0]), sample_i].log().mean()
        fasta += f">id:{i} | averaged log probability: {log_prob:.3f}\n" + seq + "\n"
    with open(os.path.join(args.output_dir, args.name, 'out.fasta'), 'w') as f:
        f.write(fasta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_MStA', type=bool, default=True)
    parser.add_argument('--pdbs_dir', type=str, default='./pdbs/')
    parser.add_argument('--checkpoint', type=str, default='./checkpoint.h5')
    parser.add_argument('--processed_pdbs_dir', type=str, default='./processed_pdbs/')

    parser.add_argument('--n_samples', type=int, default=1)
    parser.add_argument('--subseq', type=str, default=None)
    parser.add_argument('--use_msta', type=bool, default=True)
    parser.add_argument('--max_n_aligns', type=int, default=50)     # set to prevent cuda memory overflow
    parser.add_argument('--temperature', type=float, default=0.1)

    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--input_structure_path', type=str, default='./example.pdb')
    args = parser.parse_args()

    args.name = args.input_structure_path.split('/')[-1].split('.')[0]
    os.makedirs(os.path.join(args.output_dir, args.name), exist_ok=True)

    main(args)