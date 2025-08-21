import torch.distributed as dist
import argparse
import os
import torch
import wandb
import json
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer
import re
import nltk
import wandb
from huggingface_hub import HfApi, create_repo, login
from accelerate import Accelerator

accelerator = Accelerator()
rank_idx = accelerator.process_index
print(f"[PID {os.getpid()}, Rank {rank_idx}] Accelerator initialized. Distributed: {accelerator.distributed_type}, Device: {accelerator.device}, Num_processes: {accelerator.num_processes}", flush=True)

# ===================================================================================
# 1. REWARD FUNCTIONS
#
# ===================================================================================
# Load and prep dataset


def count_words(text):
    """Counts words in a string using a simple split."""
    if not text:
        return 0
    return len(nltk.word_tokenize(text))

def count_sentences(text):
    """Counts sentences in a string using NLTK for better accuracy."""
    if not text:
        return 0
    return len(nltk.sent_tokenize(text))

def extract_xml_answer(text: str) -> str:
    match = re.search(r"<summary>(.*?)</summary>", text, re.DOTALL)    
    if match:
        return match.group(1).strip()
    return None
def reward_word_count_normalized(completions, target_word_count, tolerance=5, **kwargs):
    """
    Calculates a NORMALIZED reward for word count with a tolerance window.
    The penalty is scaled by the target word count to keep rewards balanced.
    """
    scores = []
    FORMAT_FAILURE_PENALTY = -1.0

    for completion, target_words in zip(completions, target_word_count):
        if target_words is None:
            scores.append(0.0)
            continue

        response_text = completion[0]["content"]
        summary_text = extract_xml_answer(response_text)

        if summary_text is None:
            scores.append(FORMAT_FAILURE_PENALTY)
            continue

        num_words = count_words(summary_text)

        # Calculate the distance from the tolerance window's edge
        distance_from_window = max(0, abs(num_words - target_words) - tolerance)

        score = -distance_from_window / target_words

        scores.append(score)

    return scores
def reward_sentence_count_normalized(completions, target_sentence_count, **kwargs):
    scores = []
    FORMAT_FAILURE_PENALTY = -1.0

    for completion, target_sentences in zip(completions, target_sentence_count):
        if target_sentences is None:
            scores.append(0.0)
            continue

        response_text = completion[0]["content"]
        summary_text = extract_xml_answer(response_text)

        if summary_text is None:
            scores.append(FORMAT_FAILURE_PENALTY)
            continue

        num_sentences = count_sentences(summary_text)

        distance = abs(num_sentences - target_sentences)

        score = -distance / (target_sentences)

        scores.append(score)

    return scores

def reward_for_structure_normalized(completions, **kwargs) -> list[float]:
    scores = []
    full_pattern = re.compile(r"^\s*<reasoning>.*?</reasoning>\s*<summary>.*?</summary>\s*$", re.DOTALL)

    for completion in completions:
        text = completion[0]["content"]

        if full_pattern.search(text):
            scores.append(0.0)  # Perfect score!
            continue

        penalty = 0.0

        if "<reasoning>" not in text:
            penalty -= 0.25
        if "</reasoning>" not in text:
            penalty -= 0.25
        if "<summary>" not in text:
            penalty -= 0.25
        if "</summary>" not in text:
            penalty -= 0.25


        temp_text = text.strip()
        if not temp_text.startswith("<reasoning>") or not temp_text.endswith("</summary>"):
             penalty -= 0.2 # Add an extra penalty for extraneous text at start/end

        scores.append(max(-1.0, penalty))

    return scores
# ===================================================================================
# 2. ARGUMENT PARSING
#    This function defines the command-line arguments for our script.
# ===================================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a Llama model using GRPO with Unsloth.")

    # Model and Tokenizer arguments
    parser.add_argument("--model_name", type=str, default="meta-llama/meta-Llama-3.1-8B-Instruct", help="The base model to finetune.")
    parser.add_argument("--max_seq_length", type=int, default=4096, help="Maximum sequence length for the model.")
    parser.add_argument("--max_completion_length", type=int, default=800, help="Maximum completion length for the model.")

    # LoRA arguments
    parser.add_argument("--lora_rank", type=int, default=16, help="The rank for LoRA.")

    # Dataset arguments
    parser.add_argument("--dataset_path", type=str, default="./small_dataset.json", help="Path to the directory")

    # Training arguments
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="The learning rate for the optimizer.")
    parser.add_argument("--max_steps", type=int, default=5, help="Total number of training steps.")
    parser.add_argument("--batch_size", type=int, default=1, help="Per-device training batch size.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of steps for gradient accumulation.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log metrics every N steps.")
    parser.add_argument("--save_steps", type=int, default=50, help="Save a checkpoint every N steps.")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save model checkpoints.")
    parser.add_argument("--lora_dir", type=str, default="lora_adapters", help="Directory to save lora.")

    # W&B arguments
    parser.add_argument("--wandb_project", type=str, default="llama3-grpo-finetuning", help="The Weights & Biases project name.")
    parser.add_argument("--hf_repo_name", type=str, default="M4TT1A/my-llama3-grpo-lora", help="The name of the repository on the Hugging Face Hub (e.g., 'your-username/my-llama3-grpo-lora').")
    return parser.parse_args()

# ===================================================================================
# 3. MAIN TRAINING FUNCTION
# ===================================================================================
def main():
  
    args = parse_args()
    #if accelerator.is_main_process:
    os.environ["WANDB_PROJECT"] = args.wandb_project
    run_name = f"grpo-rank-{args.lora_rank}-lr-{args.learning_rate}-steps-{args.max_steps}"
    wandb.login(key='4c65a1c79b0c2cb47aaf9b96f87b38d2abd661b1')


    try:
      print(f"Loading dataset from {args.dataset_path}...")
      #dataset_path = '/content/drive/MyDrive/dataset_with_len_filtered.json'
      with open(args.dataset_path, "r", encoding="utf-8") as f:
         data = json.load(f)

      dataset_dict = DatasetDict({
          'train': Dataset.from_list(data['train']),
          'validation': Dataset.from_list(data['validation']),
          'test': Dataset.from_list(data['test'])
      })
      print("Successfully loaded dataset:")
    except FileNotFoundError:
        print(f"Error: Dataset file not found at '{args.dataset_path}'. Please check the path.")
        return
    except (KeyError, json.JSONDecodeError) as e:
        print(f"Error processing dataset file: {e}. Ensure it's a valid JSON with 'train', 'validation', and 'test' keys.")
        return


    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        #device_map="auto",  
        torch_dtype=torch.bfloat16,  
    )
    print(model.config.max_position_embeddings)
    print("*"*50)

    #model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=args.lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=args.lora_rank,
    )

    model = get_peft_model(model, lora_config)

    print(f"[PID {os.getpid()}, Rank {rank_idx}] Model loaded.", flush=True)
    print("*"*50)

    MAX_PROMPT_LENGTH =  args.max_seq_length- args.max_completion_length
    COMPLETION_CEILING = args.max_completion_length
    print('Befoere training')
    training_args = GRPOConfig(
        run_name=run_name,
        report_to="wandb",
        output_dir=args.output_dir,
        use_vllm=True,
        #vllm_mode="colocate",
        #vllm_tensor_parallel_size=1,
        #vllm_gpu_memory_utilization=0.4,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        max_grad_norm=0.1,
        ds3_gather_for_generation=False,

        num_generations=4,
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=COMPLETION_CEILING,
    )

    trainer = GRPOTrainer(
        model=model,
        #tokenizer=tokenizer,
        reward_funcs=[
            reward_word_count_normalized,
            reward_sentence_count_normalized,
            reward_for_structure_normalized
        ],
        args=training_args,
        train_dataset=dataset_dict['train'],
        eval_dataset=dataset_dict['validation'],
    )

    print(f"[PID {os.getpid()}, Rank {accelerator.process_index}] Starting training, will connect to vLLM server for generation...")
    trainer.train()
    print("Training finished.")

    #accelerator.wait_for_everyone()

    if args.hf_repo_name:   
        wandb.finish()
        print('before unwrap')
        unwrapped_model = accelerator.unwrap_model(trainer.model)
        print(f"Inside if")

        #print(f"Pushing LoRA adapters to Hugging Face Hub: {args.hf_repo_name}")

        # Create the repo if it doesn't exist
        #create_repo(args.hf_repo_name, exist_ok=True, private=False) # Set private=True if needed

        # Push the LoRA adapters
        #model.push_to_hub(args.hf_repo_name, use_auth_token=True)

        # Push the tokenizer
        #tokenizer.push_to_hub(args.hf_repo_name, use_auth_token=True)

        #print(f"Successfully pushed to https://huggingface.co/{args.hf_repo_name}")
    else:
        wandb.finish()

        print("No --hf_repo_name provided. Saving adapters locally to 'lora_model'.")
        model.save_pretrained("lora_model")
        tokenizer.save_pretrained("lora_model")
# ===================================================================================
# 4. SCRIPT ENTRYPOINT
# ===================================================================================
if __name__ == "__main__":
    main()