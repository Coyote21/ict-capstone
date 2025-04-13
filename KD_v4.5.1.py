# %%
#!pip install transformers
#!pip install tqdm
#!pip install evaluate
#!pip install torch
#!pip install accelerate
#!pip install numpy
#!pip install matplotlib
#!pip install tensorboardx
#!pip install scikit-learn

from transformers import AutoModelForQuestionAnswering, AutoTokenizer, Trainer, TrainingArguments, TrainerCallback
from sklearn.metrics import f1_score
from datasets import load_dataset
from pprint import pprint
from tqdm import tqdm
import evaluate
import torch
#import logging
import matplotlib.pyplot as plt
import numpy as np


# Suppress specific Hugging Face logging messages
#transformers_logger = logging.getLogger("transformers")
#transformers_logger.setLevel(logging.ERROR)

debugging = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
loss_per_epoch = []        # Loss per Epoch
exact_match_before = 0     # Exact Match score before distillation
exact_match_after = 0      # Exact Match score after distillation
f1_score_before = 0        # F1 score before distillation
f1_score_after = 0         # F1 score after distillation

# Load the SQuAD dataset
squad = load_dataset("squad")


# %% [markdown]
# # Load teacher model and tokenizer
# teacher_model_name = "csarron/roberta-base-squad-v1"
# teacher_model = AutoModelForQuestionAnswering.from_pretrained(teacher_model_name)
# teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)

# %%
# Replace original teacher model initialization
teacher_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
teacher_model = AutoModelForQuestionAnswering.from_pretrained(
    teacher_model_name,
    trust_remote_code=True  # Required for DeepSeek models
)
teacher_tokenizer = AutoTokenizer.from_pretrained(
    teacher_model_name,
    trust_remote_code=True,
    use_fast=False  # Recommended for DeepSeek models
)

# Modify preprocessing for DeepSeek's tokenization
def preprocess_teacher_train(example):
    inputs = teacher_tokenizer(
        example["question"],
        example["context"],
        truncation=True,
        max_length=512,  # Matches DeepSeek's context window
        stride=96,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        add_special_tokens=True  # Explicitly enable special tokens
    )


# %%
# Should do a training pass on SQuAD1.1 to initialise the weights in the Teacher model.
# This will also stop the warning from displaying

# Should do a training pass on SQuAD1.1 to initialise the weights in the Teacher model.
# This will also stop the warning from displaying

# New function to preprocess training data for teacher
def preprocess_teacher_train(example):
    # roberta-base-squad-v1
    #inputs = teacher_tokenizer(
    #    example["question"],
    #    example["context"],
    #    truncation=True,
    #    max_length=384,
    #    stride=128,
    #    return_overflowing_tokens=True,
    #    return_offsets_mapping=True,
    #    padding="max_length"
    #)
    # DeepSeek-R1-Distill-Qwen-1.5B
    inputs = teacher_tokenizer(
        example["question"],
        example["context"],
        truncation=True,
        max_length=512,
        stride=96,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        add_special_tokens=True  # Explicitly enable special tokens
    )
    
    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = example["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])
        
        sequence_ids = inputs.sequence_ids(i)

        # Handle empty sequence_ids case
        if not sequence_ids:
            start_positions.append(0)
            end_positions.append(0)
            continue

        # Find context start with boundary checks
        idx = 0
        while idx < len(sequence_ids) and sequence_ids[idx] != 1:
            idx += 1
        context_start = idx if idx < len(sequence_ids) else 0

        # Find context end with boundary checks
        while idx < len(sequence_ids) and sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1 if idx > 0 else 0

        # Handle answer position calculation
        if (context_start >= len(offset) or 
            context_end >= len(offset) or
            offset[context_start][0] > end_char or 
            offset[context_end][1] < start_char):
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Find start position with bounds checking
            idx = context_start
            while idx <= context_end and idx < len(offset) and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(min(idx - 1, len(offset)-1))
            
            # Find end position with bounds checking
            idx = context_end
            while idx >= context_start and idx < len(offset) and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(min(idx + 1, len(offset)-1))
    
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

# Teacher Training Arguments
teacher_training_args = TrainingArguments(
    output_dir="./teacher_train",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,  # Increased logging frequency
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    learning_rate=1e-5,
    report_to="tensorboard",
    fp16=True,  # Enable mixed precision training
    dataloader_num_workers=4,
)

# Custom progress callback
class TeacherTrainingProgress(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        print(f"🚀 Starting training with {args.num_train_epochs} epochs")
        print(f"📊 Batch size: {args.per_device_train_batch_size}")
        print(f"🔍 Evaluation every {args.eval_steps} steps")

    def on_epoch_begin(self, args, state, control, **kwargs):
        print(f"\n⏳ Starting epoch {state.epoch}/{args.num_train_epochs}")
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'loss' in logs:
            print(f"Step {state.global_step}: Loss {logs['loss']:.4f}")
        if logs and 'eval_loss' in logs:
            print(f"Validation Loss: {logs['eval_loss']:.4f}")
            print(f"Exact Match: {logs['eval_exact_match']:.2f}%")
            print(f"F1 Score: {logs['eval_f1']:.2f}%")

# Add metrics computation to Trainer
def compute_metrics(p):

    # Convert logits to predictions
    start_pred = np.argmax(p.predictions[0], axis=1)
    end_pred = np.argmax(p.predictions[1], axis=1)
    
    # Get true positions
    start_true = p.label_ids[0]
    end_true = p.label_ids[1]
    
    # Calculate exact match
    exact_matches = np.logical_and(
        start_pred == start_true,
        end_pred == end_true
    )

    # Calculate span F1
    def overlap_f1(p_start, p_end, t_start, t_end):
        pred_span = set(range(p_start, p_end+1))
        true_span = set(range(t_start, t_end+1))
        overlap = len(pred_span & true_span)
        precision = overlap / len(pred_span) if pred_span else 0
        recall = overlap / len(true_span) if true_span else 0
        return 2*(precision*recall)/(precision+recall) if (precision+recall) else 0
    
    f1_scores = [
        overlap_f1(sp, ep, st, et)
        for sp, ep, st, et in zip(start_pred, end_pred, start_true, end_true)
    ]
    
    return {
        "exact_match": np.mean(exact_matches) * 100,
        "f1": np.mean(f1_scores) * 100
    }


# Create Trainer for teacher
teacher_trainer = Trainer(
    model=teacher_model,
    args=teacher_training_args,
    train_dataset=squad["train"].map(preprocess_teacher_train, batched=True, remove_columns=squad["train"].column_names),
    eval_dataset=squad["validation"].map(preprocess_teacher_train, batched=True, remove_columns=squad["validation"].column_names),
    tokenizer=teacher_tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[TeacherTrainingProgress()]
)

# Train teacher model
print("\nTraining Teacher Model on SQuAD1.1...")
teacher_trainer.train()
teacher_model.save_pretrained("./trained_teacher")
teacher_tokenizer.save_pretrained("./trained_teacher")

print("\nRe-loading optimized teacher model")
teacher_model = AutoModelForQuestionAnswering.from_pretrained("./trained_teacher")
teacher_tokenizer = AutoTokenizer.from_pretrained("./trained_teacher")





# %%
# Load student model and tokenizer (smaller version of RoBERTa)
student_model_name = "distilroberta-base"  # Example smaller model
student_model = AutoModelForQuestionAnswering.from_pretrained(student_model_name)
student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)

# %%
# Define evaluation function
def evaluate_model(model, tokenizer, dataset):
    model.to(device)
    
    metric = evaluate.load("squad")
    predictions = []
    references = []

    for example in tqdm(dataset, desc="Evaluating"):
        # Tokenize inputs
        inputs = tokenizer(
            example["context"], example["question"], truncation=True, padding=True, return_tensors="pt"
        )

        # Move inputs to the same device as the model
        inputs = {key: value.to(device) for key, value in inputs.items()}
        
        # Get model outputs
        outputs = model(**inputs)
        start_logits, end_logits = outputs.start_logits, outputs.end_logits
        start_idx = torch.argmax(start_logits, dim=-1).item()
        end_idx = torch.argmax(end_logits, dim=-1).item()
        
        # Decode prediction
        prediction = tokenizer.decode(inputs["input_ids"][0][start_idx:end_idx + 1])
        
        # Append to predictions
        predictions.append({
            "id": example["id"],
            "prediction_text": prediction
        })

        # Append to references (ground truth)
        references.append({
            "id": example["id"],
            "answers": example["answers"]
        })

    # Compute metrics
    result = metric.compute(predictions=predictions, references=references)
    print(f"Exact Match: {result['exact_match']:.2f}%")
    print(f"F1 Score: {result['f1']:.2f}%\n")
    
    return result

# %%
def preprocess_validation_data(example):
    # Tokenize context and question
    inputs = student_tokenizer(
        example["context"],
        example["question"],
        truncation=True,
        padding="max_length",
        max_length=384,
    )
    return inputs

# %%
# Evaluate teacher model on validation set

validation_dataset = squad["validation"]

#if debugging: 
#    validation_set = validation_dataset.select(range(5000))
#    validation_dataset = validation_set

print("Teacher Model Evaluation")
result = evaluate_model(teacher_model, teacher_tokenizer, validation_dataset)

# Evaluate student model on validation set (before distillation)
print("Student Model Evaluation (Before Distillation)")
result = evaluate_model(student_model, student_tokenizer, validation_dataset)

# Save the results for use in Graphs
exact_match_before = result['exact_match']
f1_score_before = result['f1']

# %%
# Prepare data for distillation
def preprocess_data(example):

    # Move teacher model to appropriate device
    teacher_model.to(device)
    
    # Tokenize context and question
    inputs = teacher_tokenizer(
        example["context"], 
        example["question"], 
        truncation=True, 
        padding="max_length", 
        max_length=384,
        return_tensors="pt"
    )

    # Move inputs to the same device as the teacher model
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # Get logits from the teacher model
    with torch.no_grad():
        outputs = teacher_model(**inputs)
    
    # Add teacher logits to the example
    example["input_ids"] = inputs["input_ids"][0].cpu().tolist()
    example["attention_mask"] = inputs["attention_mask"][0].cpu().tolist()
    example["start_logits"] = outputs.start_logits[0].cpu().tolist()
    example["end_logits"] = outputs.end_logits[0].cpu().tolist()
    
    return example

if debugging: 
    train_dataset = squad["train"].select(range(30000)).map(preprocess_data)
else:
    train_dataset = squad["train"].map(preprocess_data)    

# Apply preprocessing to validation dataset
validation_set = validation_dataset.map(preprocess_validation_data, batched=True)

# Define training arguments for student model
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    logging_steps=10,          # Log every 10 steps need when using small datasets for testing
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
    save_total_limit=2,
    remove_unused_columns=False,
)

# %%
# Define custom loss function for knowledge distillation
class DistillationTrainer(Trainer):    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Custom loss function for knowledge distillation.
        """
        
        # Forward pass
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        
        # Compute KL Divergence loss between teacher and student logits
        start_loss = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(outputs.start_logits, dim=-1),
            torch.nn.functional.softmax(inputs["start_logits"], dim=-1),
            reduction="batchmean"
        )
        end_loss = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(outputs.end_logits, dim=-1),
            torch.nn.functional.softmax(inputs["end_logits"], dim=-1),
            reduction="batchmean"
        )
        
        # Average the start and end losses
        loss = (start_loss + end_loss) / 2
        
        return (loss, outputs) if return_outputs else loss

# %%
class FullyCustomDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        # Filter out non-tokenized fields before padding
        tokenized_features = [
            {k: v for k, v in f.items() if k in ["input_ids", "attention_mask", "token_type_ids"]}
            for f in features
        ]

        # Dynamically pad input_ids and attention_mask using the tokenizer
        batch = self.tokenizer.pad(
            tokenized_features,
            padding=True,
            max_length=None,
            return_tensors="pt",
        )

        # Add custom fields (e.g., start_logits and end_logits) to the batch
        if "start_logits" in features[0]:
            batch["start_logits"] = torch.tensor([f["start_logits"] for f in features], dtype=torch.float32)
        if "end_logits" in features[0]:
            batch["end_logits"] = torch.tensor([f["end_logits"] for f in features], dtype=torch.float32)

        return batch



# %%
data_collator = FullyCustomDataCollator(tokenizer=student_tokenizer)

# %%
# Train the student model using the custom trainer
trainer = DistillationTrainer(
    model=student_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_set,
    data_collator=data_collator,
)

#transformers_logger.setLevel(logging.INFO)

trainer.train()

# %%
# Evaluate student model on validation set after distillation
print("Student Model Evaluation after Distillation")
result = evaluate_model(student_model, student_tokenizer, validation_dataset)

# Save the results for use in Graphs
exact_match_after = result['exact_match']
f1_score_after = result['f1']

# %%

# Loss values per epoch (replace with actual values extracted from `trainer.state.log_history`)
# Generate 50 values starting at 0.8 and decaying exponentially to 0.19
factor = (0.19 / 0.8) ** (1 / 49)
loss_values = [0.19 * (factor ** i) for i in range(50)]

# Plotting loss values per epoch
plt.figure(figsize=(8, 6))
epochs = np.arange(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, marker='o', label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Per Epoch")
plt.legend()
plt.grid()
plt.show()

# Plotting Exact Match scores before and after distillation
plt.figure(figsize=(8, 6))
labels = ["Before Distillation", "After Distillation"]
exact_match_scores = [exact_match_before, exact_match_after]
plt.bar(labels, exact_match_scores, color=['blue', 'green'])
plt.ylabel("Exact Match (%)")
plt.title("Exact Match Score Before and After Distillation")
for i, v in enumerate(exact_match_scores):
    plt.text(i, v + 1, f"{v:.1f}%", ha='center', fontsize=12)
plt.show()

# Plotting F1 scores before and after distillation
plt.figure(figsize=(8, 6))
f1_scores = [f1_score_before, f1_score_after]
plt.bar(labels, f1_scores, color=['orange', 'red'])
plt.ylabel("F1 Score (%)")
plt.title("F1 Score Before and After Distillation")
for i, v in enumerate(f1_scores):
    plt.text(i, v + 1, f"{v:.1f}%", ha='center', fontsize=12)
plt.show()


# %%



