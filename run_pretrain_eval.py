import argparse
import time
from pathlib import Path

import pretrain
import eval_linear


if __name__ == "__main__":
    parser = argparse.ArgumentParser('DINO TRAINING PIPELINE', parents=[pretrain.get_args_parser()])
    parser.add_argument("--pipeline_mode", default=('pretrain', 'eval'), choices=['pretrain', 'eval'], type=str, nargs='+')
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if 'pretrain' in args.pipeline_mode:
        pretrain.train_dino(args)
        time.sleep(10)

    if 'eval' in args.pipeline_mode:
        # change linear specific parameters
        args.epochs = 300
        args.lr = 0.01
        args.momentum = 0.9
        args.weight_decay = 0
        args.batch_size_per_gpu = 256
        args.pretrained_weights = f"{args.output_dir}/checkpoint.pth"
        args.checkpoint_key = "teacher"
        args.n_last_blocks = 4
        args.avgpool_patchtokens = False
        args.val_freq = 1
        eval_linear.eval_linear(args, True)
