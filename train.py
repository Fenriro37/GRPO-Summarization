import argparse
import os
import torch
import wandb
import json
from datasets import load_dataset, Dataset, DatasetDict
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
import re
import nltk
import wandb
wandb.login(key='4c65a1c79b0c2cb47aaf9b96f87b38d2abd661b1')
# ===================================================================================
# 1. REWARD FUNCTIONS
# 
# ===================================================================================
# Load and prep dataset
SYSTEM_PROMPT = """You are a precise summarization expert. Your task is to create a summary that captures the essential information from the provided text while strictly adhering to the specified length constraint.

INSTRUCTIONS:
1. Analyze the text to identify the most critical information, key arguments, and main conclusions
2. Prioritize factual content over opinions unless opinions are central to the text's purpose
3. Maintain logical flow and coherence in your summary
4. Count your words/sentences carefully to meet the exact requirement

OUTPUT FORMAT:
- First, provide a brief explanation (2-3 sentences) between <explanation></explanation> tags explaining your summarization strategy and why the specified length is appropriate for capturing the key information
- Then, provide your summary between <summary></summary> tags
- Ensure your summary contains exactly the requested number of words/sentences

QUALITY REQUIREMENTS:
- Preserve the original meaning and tone
- Use clear, concise language
- Avoid redundancy and filler words
- Include specific details, numbers, or examples only if they are crucial to understanding
- Ensure each sentence (if counting sentences) or word (if counting words) adds meaningful value"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""
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
    answer = text.split("<summary>")[-1]
    answer = answer.split("</summary>")[0]
    return answer.strip()
def reward_word_count_normalized(completions, target_word_count, tolerance=5, **kwargs):
    """
    Calculates a NORMALIZED reward for word count with a tolerance window.
    The penalty is scaled by the target word count to keep rewards balanced.
    """
    scores = []
    FORMAT_FAILURE_PENALTY = -5.0

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
    FORMAT_FAILURE_PENALTY = -5.0

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

# ===================================================================================
# 2. ARGUMENT PARSING
#    This function defines the command-line arguments for our script.
# ===================================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a Llama model using GRPO with Unsloth.")

    # Model and Tokenizer arguments
    parser.add_argument("--model_name", type=str, default="meta-llama/meta-Llama-3.1-8B-Instruct", help="The base model to finetune.")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length for the model.")

    # LoRA arguments
    parser.add_argument("--lora_rank", type=int, default=64, help="The rank for LoRA.")

    # Dataset arguments
    parser.add_argument("--dataset_path", type=str, default="small_dataset.json", help="Path to the directory")

    # Training arguments
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="The learning rate for the optimizer.")
    parser.add_argument("--max_steps", type=int, default=5, help="Total number of training steps.")
    parser.add_argument("--batch_size", type=int, default=1, help="Per-device training batch size.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of steps for gradient accumulation.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log metrics every N steps.")
    parser.add_argument("--save_steps", type=int, default=50, help="Save a checkpoint every N steps.")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save model checkpoints.")

    # W&B arguments
    parser.add_argument("--wandb_project", type=str, default="llama3-grpo-finetuning", help="The Weights & Biases project name.")

    return parser.parse_args()

# ===================================================================================
# 3. MAIN TRAINING FUNCTION
# ===================================================================================
def main():
    args = parse_args()

    os.environ["WANDB_PROJECT"] = args.wandb_project
    run_name = f"grpo-rank-{args.lora_rank}-lr-{args.learning_rate}-steps-{args.max_steps}"

    try:
      print(f"Loading dataset from {args.dataset_path}...")
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

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        fast_inference=False,
        max_lora_rank=args.lora_rank,
        gpu_memory_utilization=0.6,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=args.lora_rank,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    MAX_PROMPT_LENGTH = 1536
    COMPLETION_CEILING = args.max_seq_length - MAX_PROMPT_LENGTH

    training_args = GRPOConfig(
        run_name=run_name,
        report_to="wandb",
        output_dir=args.output_dir,

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

        num_generations=4,
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=COMPLETION_CEILING,
    )

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_funcs=[
            reward_word_count_normalized,
            reward_sentence_count_normalized
        ],
        args=training_args,
        train_dataset=dataset_dict['train'],
        eval_dataset=dataset_dict['validation'],
    )

    print("Starting training...")
    trainer.train()
    print("Training finished.")

    final_model_path = os.path.join(args.output_dir, f"model_{run_name}")
    trainer.save_model(final_model_path)
    print(f"Final model saved to {final_model_path}")

    wandb.finish()

# ===================================================================================
# 4. SCRIPT ENTRYPOINT
# ===================================================================================
if __name__ == "__main__":
    main()