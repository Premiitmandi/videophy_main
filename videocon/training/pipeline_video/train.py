import argparse
import os
import torch
import sys
from functools import partial
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from sconf import Config
from transformers import Trainer
from transformers.training_args import TrainingArguments

from mplug_owl_video.modeling_mplug_owl import MplugOwlForConditionalGeneration
from transformers.models.llama.tokenization_llama import LlamaTokenizer
from data_utils import train_valid_test_datasets_provider
from utils import batchify, set_args

# Relative paths are relative to the script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()

# Model
parser.add_argument(
    '--pretrained-ckpt', 
    type=str, 
    default=os.path.join(SCRIPT_DIR, 'models/mplug-owl-llama-7b-video'), 
    help='Relative path to the pretrained checkpoint.'
)
parser.add_argument('--finetuned-ckpt', type=str, default=None, help='Path to the finetuned checkpoint.')
parser.add_argument('--seq-length', type=int, default=1024, help='Maximum sequence length to process.')

parser.add_argument('--use-lora', action='store_true', help='Enable LORA.')
parser.add_argument('--lora-r', type=int, default=8, help='Rank for LORA.')
parser.add_argument('--lora-alpha', type=int, default=32, help='Alpha value for LORA.')
parser.add_argument('--lora-dropout', type=float, default=0.05, help='Dropout for LORA.')
parser.add_argument('--bf16', action='store_true', default=False, help='Run model in bfloat16 mode.')

# Data
parser.add_argument(
    '--mm-config', 
    type=str, 
    default=os.path.join(SCRIPT_DIR, 'videocon/training/configs/video.yaml'), 
    help='Relative path to the multimodal config YAML file.'
)
parser.add_argument('--num-workers', type=int, default=8, help="Number of data loader workers.")

# Training
parser.add_argument('--train-epochs', type=int, default=3, help='Total number of training epochs.')
parser.add_argument('--micro-batch-size', type=int, default=1, help='Batch size per model instance.')
parser.add_argument('--gradient-accumulation-steps', type=int, default=8, help='Gradient accumulation steps.')

# Evaluation & Save
parser.add_argument(
    '--save-path', 
    type=str, 
    default=os.path.join(SCRIPT_DIR, 'output'), 
    help='Relative path to save model checkpoints.'
)
parser.add_argument('--eval-iters', type=int, default=100, help='Evaluation interval in steps.')

args = parser.parse_args()

def main():
    # Validate input arguments
    print("Starting script execution...")
    if not os.path.exists(args.mm_config):
        raise FileNotFoundError(f"Multimodal config file not found: {args.mm_config}")
    print("Configuration file validated.")
    if args.save_path:
        os.makedirs(args.save_path, exist_ok=True)
        print(f"Output directory created: {args.save_path}")

    # Load config
    print("Loading configuration...")
    config = Config(args.mm_config)
    print("Configuration loaded successfully.")

    # Load model and tokenizer
    print("Loading model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(" Model loaded successfully!")
    try:
        model = MplugOwlForConditionalGeneration.from_pretrained(
            args.pretrained-ckpt,
            torch_dtype=torch.bfloat16 if args.bf16 and device == 'cuda' else torch.float32,
        ).to(device)
        print(f"Model loaded successfully on {device}.")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        sys.exit(1)

    print("Loading tokenizer...")
    tokenizer = LlamaTokenizer.from_pretrained(args.pretrained_ckpt, legacy=True)
    print("Tokenizer loaded successfully.")

    # Load datasets
    print("Loading datasets...")
    train_data, valid_data = train_valid_test_datasets_provider(
        config.data_files, config=config, tokenizer=tokenizer, seq_length=args.seq_length
    )
    print(f"Training samples: {len(train_data)}, Validation samples: {len(valid_data)}.")

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.save_path,
            per_device_train_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            evaluation_strategy="steps",
            logging_steps=100,
            save_steps=args.eval_iters,
            num_train_epochs=args.train_epochs,
            save_total_limit=2,
        ),
        train_dataset=train_data,
        eval_dataset=valid_data,
        tokenizer=tokenizer,
    )

    # Train
    print("Starting training...")
    trainer.train()
    print(f"Training complete. Model saved to {args.save_path}")


if __name__ == "__main__":
    main()
