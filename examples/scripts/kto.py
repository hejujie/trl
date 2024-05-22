# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Run the KTO training script with the commands below. In general, the optimal configuration for KTO will be similar to that of DPO.

# Full training:
python examples/scripts/kto.py \
    --model_name_or_path=trl-lib/qwen1.5-1.8b-sft \
    --per_device_train_batch_size 16 \
    --num_train_epochs 1 \
    --learning_rate 1e-5 \
    --lr_scheduler_type=cosine \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --eval_steps 500 \
    --output_dir=kto-aligned-model \
    --warmup_ratio 0.1 \
    --report_to wandb \
    --bf16 \
    --logging_first_step

# QLoRA:
python examples/scripts/kto.py \
    --model_name_or_path=trl-lib/qwen1.5-1.8b-sft \
    --per_device_train_batch_size 8 \
    --num_train_epochs 1 \
    --learning_rate 1e-4 \
    --lr_scheduler_type=cosine \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --eval_steps 500 \
    --output_dir=kto-aligned-model-lora \
    --warmup_ratio 0.1 \
    --report_to wandb \
    --bf16 \
    --logging_first_step \
    --use_peft \
    --load_in_4bit \
    --lora_target_modules=all-linear \
    --lora_r=16 \
    --lora_alpha=16
"""

from dataclasses import dataclass

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

from trl import KTOConfig, KTOTrainer, ModelConfig, get_peft_config, setup_chat_format


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the KTO training script.
    """

    dataset_name: str = "trl-lib/kto-mix-14k"
    dataset_task_types: str = "task1 task2"  # Replace with the task names you want to filter by


# Count task occurrences before filtering
def count_tasks(dataset_split):
    task_counts = {}
    # for example in dataset_split:
    #     task = example['task']
    #     if task in task_counts:
    #         task_counts[task] += 1
    #     else:
    #         task_counts[task] = 1
    return task_counts

# Define the filter function
def filter_by_task(example, dataset_task_types):
    return example['task'] in dataset_task_types

# Apply chat template
def format_dataset(example):
    example["prompt"] = tokenizer.apply_chat_template(example["prompt"], tokenize=False)
    example["completion"] = tokenizer.apply_chat_template(example["completion"], tokenize=False)
    return example



if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, KTOConfig, ModelConfig))
    script_args, kto_args, model_args = parser.parse_args_into_dataclasses()

    # Load the dataset
    data_files = {"train": "kto_train.jsonl", "test": "kto_test.jsonl"}  # Adjust paths as needed
    dataset = load_dataset(script_args.dataset_name, data_files=data_files)

    # Print the size of the original datasets
    print(f"Original train dataset size: {dataset['train'].num_rows}")
    print(f"Original test dataset size: {dataset['test'].num_rows}")

    train_task_counts_before = count_tasks(dataset['train'])
    test_task_counts_before = count_tasks(dataset['test'])

    print("Task counts in train dataset before filtering:", train_task_counts_before)
    print("Task counts in test dataset before filtering:", test_task_counts_before)

    # Filter the dataset if task names are provided
    if len(script_args.dataset_task_types) > 0:
        data_task_types = script_args.dataset_task_types.split()
        print(f"Filtering train set with tasks: {data_task_types}")
        filtered_dataset = dataset.filter(lambda example: filter_by_task(example, data_task_types), num_proc=kto_args.dataset_num_proc)
        
        # Print the size of the filtered datasets
        print(f"Filtered train dataset size: {filtered_dataset['train'].num_rows}")
        print(f"Filtered test dataset size: {filtered_dataset['test'].num_rows}")

        train_task_counts_after = count_tasks(filtered_dataset['train'])
        test_task_counts_after = count_tasks(filtered_dataset['test'])

        print("Task counts in train dataset after filtering:", train_task_counts_after)
        print("Task counts in test dataset after filtering:", test_task_counts_after)
    else:
        filtered_dataset = dataset


    # Load a pretrained model
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)
    model_ref = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # If we are aligning a base model, we use ChatML as the default template
    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer)


    formatted_dataset = filtered_dataset.map(format_dataset, num_proc=kto_args.dataset_num_proc)

    # Initialize the KTO trainer
    kto_trainer = KTOTrainer(
        model,
        model_ref,
        args=kto_args,
        train_dataset=formatted_dataset["train"],
        eval_dataset=formatted_dataset["test"],
        tokenizer=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    # Train and push the model to the Hub
    kto_trainer.train()
    kto_trainer.save_model(kto_args.output_dir)
    # kto_trainer.push_to_hub()
