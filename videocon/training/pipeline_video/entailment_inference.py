# import os
# import csv
# import torch
# import argparse
# import pandas as pd
# import torch.nn as nn
# from tqdm import tqdm
# from transformers.models.llama.tokenization_llama import LlamaTokenizer
# from torch.utils.data import DataLoader
# from mplug_owl_video.modeling_mplug_owl import MplugOwlForConditionalGeneration
# from mplug_owl_video.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor
# from data_utils.xgpt3_dataset import MultiModalDataset
# from utils import batchify

# # Optional: Handle Flash-Attention Import
# try:
#     import flash_attn
# except ImportError:
#     print("Flash-Attention is not installed. Proceeding with default PyTorch behavior.")

# parser = argparse.ArgumentParser()
# parser.add_argument('--input_csv', type=str, required=True, help='input csv file')
# parser.add_argument('--output_csv', type=str, help='output csv with scores')
# parser.add_argument('--checkpoint', type=str, required=True, help='pretrained checkpoint')
# parser.add_argument('--batch_size', type=int, default=16, help='Batch size for inference')

# args = parser.parse_args()
# softmax = nn.Softmax(dim=2)

# def get_entail(logits, input_ids, tokenizer):
#     logits = softmax(logits)
#     token_id_yes = tokenizer.encode('Yes', add_special_tokens=False)[0]
#     token_id_no = tokenizer.encode('No', add_special_tokens=False)[0]
#     entailment = []
#     for j in range(len(logits)):
#         for i in range(len(input_ids[j])):
#             if input_ids[j][i] == tokenizer.pad_token_id:
#                 i = i - 1
#                 break
#             elif i == len(input_ids[j]) - 1:
#                 break
#         score = logits[j][i][token_id_yes] / (logits[j][i][token_id_yes] + logits[j][i][token_id_no])
#         entailment.append(score)
#     entailment = torch.stack(entailment)
#     return entailment

# def get_scores(model, tokenizer, dataloader):
#     with torch.no_grad():
#         for index, inputs in tqdm(enumerate(dataloader)):
#             for k, v in inputs.items():
#                 if torch.is_tensor(v):
#                     if v.dtype == torch.float:
#                         inputs[k] = v.bfloat16()
#                     inputs[k] = inputs[k].to(model.device)
#             outputs = model(pixel_values=inputs['pixel_values'], video_pixel_values=inputs['video_pixel_values'], labels=None,
#                             num_images=inputs['num_images'], num_videos=inputs['num_videos'], input_ids=inputs['input_ids'], 
#                             non_padding_mask=inputs['non_padding_mask'], non_media_mask=inputs['non_media_mask'], prompt_mask=inputs['prompt_mask'])
#             logits = outputs['logits']
#             entail_scores = get_entail(logits, inputs['input_ids'], tokenizer)
#             for m in range(len(entail_scores)):
#                 with open(args.output_csv, 'a', newline='') as f:
#                     writer = csv.writer(f)
#                     writer.writerow([inputs['videopaths'][m], inputs['captions'][m], entail_scores[m].item()])
#             print(f"Batch {index} Done")

# def main():
#     checkpoint = args.checkpoint

#     # Validate input and output paths
#     if not os.path.exists(args.input_csv):
#         raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")
#     if not os.path.isdir(checkpoint):
#         raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint}")
#     if args.output_csv:
#         os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

#     # Add header to output CSV if it doesn't exist
#     if not os.path.exists(args.output_csv):
#         with open(args.output_csv, 'w', newline='') as f:
#             writer = csv.writer(f)
#             writer.writerow(['videopath', 'caption', 'entailment_score'])

#     # Load processors and tokenizer
#     print("Loading tokenizer and processors...")
#     tokenizer = LlamaTokenizer.from_pretrained(checkpoint, legacy=True)
#     image_processor = MplugOwlImageProcessor.from_pretrained(checkpoint)
#     processor = MplugOwlProcessor(image_processor, tokenizer)
#     print("Processors loaded successfully.")

#     # Load data
#     valid_data = MultiModalDataset(args.input_csv, tokenizer, processor, max_length=256, loss_objective='sequential')
#     dataloader = DataLoader(valid_data, batch_size=args.batch_size, pin_memory=True, collate_fn=batchify)

#     # Instantiate model
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     model = MplugOwlForConditionalGeneration.from_pretrained(
#         checkpoint,
#         torch_dtype=torch.bfloat16 if device == 'cuda' else torch.float32,
#     ).to(device)
#     print(f"Model loaded on {device}")
#     model.eval()

#     # Run inference
#     try:
#         get_scores(model, tokenizer, dataloader)
#     except Exception as e:
#         print(f"Error during inference: {str(e)}")
#         raise

# if __name__ == "__main__":
#     main()
import os
import csv
import torch
import argparse
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from transformers.models.llama.tokenization_llama import LlamaTokenizer
from torch.utils.data import DataLoader
from mplug_owl_video.modeling_mplug_owl import MplugOwlForConditionalGeneration
from mplug_owl_video.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor
from data_utils.xgpt3_dataset import MultiModalDataset
from utils import batchify

# Optional: Handle Flash-Attention Import
try:
    import flash_attn
except ImportError:
    print("Flash-Attention is not installed. Proceeding with default PyTorch behavior.")

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--input_csv', type=str, required=True, help='Input CSV file')
parser.add_argument('--output_csv', type=str, help='Output CSV with scores')
parser.add_argument('--checkpoint', type=str, required=True, help='Pretrained checkpoint')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for inference')
args = parser.parse_args()

softmax = nn.Softmax(dim=2)

# Function to compute entailment score
def get_entail(logits, input_ids, tokenizer):
    logits = softmax(logits)
    token_id_yes = tokenizer.encode('Yes', add_special_tokens=False)[0]
    token_id_no = tokenizer.encode('No', add_special_tokens=False)[0]
    entailment = []
    
    for j in range(len(logits)):
        for i in range(len(input_ids[j])):
            if input_ids[j][i] == tokenizer.pad_token_id:
                i = i - 1
                break
            elif i == len(input_ids[j]) - 1:
                break
        score = logits[j][i][token_id_yes] / (logits[j][i][token_id_yes] + logits[j][i][token_id_no])
        entailment.append(score)
    
    entailment = torch.stack(entailment)
    return entailment

# Function to get entailment scores
def get_scores(model, tokenizer, dataloader):
    with torch.no_grad():
        for index, inputs in tqdm(enumerate(dataloader)):
            # Print the keys of the input sample to debug missing keys
            print(f"\n Batch {index} - Sample keys:", list(inputs.keys()))

            if 'input_ids' not in inputs:
                print(" ERROR: Missing 'input_ids' in dataset!")
                continue  # Skip this batch

            for k, v in inputs.items():
                if torch.is_tensor(v):
                    if v.dtype == torch.float:
                        inputs[k] = v.bfloat16()
                    inputs[k] = inputs[k].to(model.device)
            
            outputs = model(
                pixel_values=inputs.get('pixel_values'),
                video_pixel_values=inputs.get('video_pixel_values'),
                labels=None,
                num_images=inputs.get('num_images'),
                num_videos=inputs.get('num_videos'),
                input_ids=inputs.get('input_ids'), 
                non_padding_mask=inputs.get('non_padding_mask'),
                non_media_mask=inputs.get('non_media_mask'),
                prompt_mask=inputs.get('prompt_mask')
            )
            
            logits = outputs['logits']
            entail_scores = get_entail(logits, inputs['input_ids'], tokenizer)
            
            for m in range(len(entail_scores)):
                video_path = inputs.get('videopath', ['Path not found'])[m]
                caption = inputs.get('captions', ['Caption not found'])[m]
                score = entail_scores[m].item()

                print(f" Processed: {video_path} - Score: {score}")

                with open(args.output_csv, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([video_path, caption, score])

            print(f" Batch {index} Done")

# Main function
def main():
    checkpoint = args.checkpoint

    # Validate input and output paths
    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f" Input CSV not found: {args.input_csv}")
    if not os.path.isdir(checkpoint):
        raise FileNotFoundError(f" Checkpoint directory not found: {checkpoint}")
    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    # Load input CSV and display some sample data
    df = pd.read_csv(args.input_csv)
    print("\n Loaded CSV Data:\n", df.head())  # Display first few rows
    print(f" Total rows loaded: {len(df)}")

    # Check if CSV has required columns
    required_columns = {'videopath', 'caption'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f" CSV is missing required columns: {required_columns}")

    # Initialize output CSV
    if not os.path.exists(args.output_csv):
        with open(args.output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['videopath', 'caption', 'entailment_score'])
    print(f" Output CSV initialized at: {args.output_csv}")

    # Load processors and tokenizer
    print("\n Loading tokenizer and processors...")
    tokenizer = LlamaTokenizer.from_pretrained(checkpoint, legacy=True)
    image_processor = MplugOwlImageProcessor.from_pretrained(checkpoint)
    processor = MplugOwlProcessor(image_processor, tokenizer)
    print(" Processors loaded successfully.")

    # Load dataset
    valid_data = MultiModalDataset(args.input_csv, tokenizer, processor, max_length=256, loss_objective='sequential')

    # Debugging: Check dataset samples
    print("\n Checking dataset samples...")
    for i in range(min(3, len(valid_data))):  # Print up to 3 samples
        sample = valid_data[i]
        print(f"Sample {i}:")
        print("  Available Keys:", list(sample.keys()))
        print("  Video Path:", sample.get('videopath', 'Missing key'))
        print("  Caption:", sample.get('caption', 'Missing key'))
        print("  Input IDs:", sample.get('input_ids', 'Missing key'))  # 🔥 Check if input_ids exist

    # Create DataLoader
    dataloader = DataLoader(valid_data, batch_size=args.batch_size, pin_memory=True, collate_fn=batchify)

    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n Loading model on {device}...")
    model = MplugOwlForConditionalGeneration.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16 if device == 'cuda' else torch.float32,
    ).to(device)
    print(" Model loaded successfully.")
    model.eval()

    # Run inference
    try:
        print("\n Running inference...")
        get_scores(model, tokenizer, dataloader)
    except Exception as e:
        print(f"\n Error during inference: {str(e)}")
        raise

if __name__ == "__main__":
    main()
