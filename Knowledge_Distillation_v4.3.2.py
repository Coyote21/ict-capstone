"""
Knowledge Distillation for Question Answering on SQuAD
------------------------------------------------------
This script demonstrates how to perform knowledge distillation to train a smaller (student) model
to mimic the predictions of a larger (teacher) model for the SQuAD question answering task.

Key Steps:
- Load SQuAD dataset and two models (teacher, student)
- Evaluate both models before distillation
- Prepare training data with teacher's soft labels (logits)
- Define custom Trainer and loss function for distillation (KL-divergence)
- Train the student model using knowledge distillation
- Evaluate and visualize results before and after distillation
"""
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, Trainer, TrainingArguments, TrainerCallback
from datasets import load_dataset
from tqdm import tqdm
import evaluate
import torch
import matplotlib.pyplot as plt

# Enable/disable debugging (smaller dataset for quick runs)
debugging = False

# Set device to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Tracking variables for loss and metrics
loss_values = []           # Stores training loss values per logging step
epoch_progress = []        # Stores epoch numbers for plotting loss

# Metrics before and after distillation
exact_match_before = 0     # Exact Match score before distillation
exact_match_after = 0      # Exact Match score after distillation
f1_score_before = 0        # F1 score before distillation
f1_score_after = 0         # F1 score after distillation

# -------------------------------
# 1. Load Dataset and Models
# -------------------------------

# Load the SQuAD dataset (train and validation splits)
squad = load_dataset("squad")

# Load teacher model (large RoBERTa fine-tuned on SQuAD) and its tokenizer
teacher_model_name = "Dingyun-Huang/roberta-large-squad1" #355M parameters
teacher_model = AutoModelForQuestionAnswering.from_pretrained(teacher_model_name)
teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)

# Load student model (smaller RoBERTa) and its tokenizer
student_model_name = "deepset/roberta-base-squad2"  # 124M parameters
student_model = AutoModelForQuestionAnswering.from_pretrained(student_model_name)
student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)

# -------------------------------
# 2. Evaluation Function
# -------------------------------

def evaluate_model(model, tokenizer, dataset):
    """
    Evaluates a QA model on the given dataset using SQuAD metrics (Exact Match, F1).

    Args:
        model: HuggingFace QA model (teacher or student)
        tokenizer: Corresponding tokenizer
        dataset: Dataset split to evaluate (e.g., squad["validation"])

    Returns:
        dict: {'exact_match': float, 'f1': float}
    """
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
    
    return result

# -------------------------------
# 3. Preprocessing Functions
# -------------------------------

# Prepare validation data for evaluation
def preprocess_validation_data(example):
    """
    Tokenizes validation data for the student model.

    Args:
        example: A single SQuAD example

    Returns:
        dict: Tokenized inputs
    """
    # Tokenize context and question
    inputs = student_tokenizer(
        example["context"],
        example["question"],
        truncation=True,
        padding="max_length",
        max_length=384,
    )
    return inputs

# Prepare training data for distillation
def preprocess_data(example):
    """
    Prepares training data for distillation by attaching teacher's logits.

    Args:
        example: A single SQuAD example

    Returns:
        dict: Example with input_ids, attention_mask, and teacher's start/end logits
    """
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

# -------------------------------
# 4. Evaluate Teacher and Student Before Distillation
# -------------------------------

validation_dataset = squad["validation"]

# Evaluate teacher model
print("Teacher Model Evaluation")
result = evaluate_model(teacher_model, teacher_tokenizer, validation_dataset)
print(f"Exact Match: {result['exact_match']:.2f}%")
print(f"F1 Score: {result['f1']:.2f}%\n")

# Evaluate student model on validation set (before distillation)
print("Student Model Evaluation (Before Distillation)")
result = evaluate_model(student_model, student_tokenizer, validation_dataset)
print(f"Exact Match: {result['exact_match']:.2f}%")
print(f"F1 Score: {result['f1']:.2f}%\n")

# Save the results for use in Graphs
exact_match_before = result['exact_match']
f1_score_before = result['f1']

# -------------------------------
# 5. Prepare Training and Validation Data
# -------------------------------

# For debugging, use a subset of training data for faster runs
# Apply preprocessing to training dataset
if debugging: 
    train_dataset = squad["train"].select(range(10000)).map(preprocess_data)
else:
    train_dataset = squad["train"].map(preprocess_data)    

# Apply preprocessing to validation dataset
validation_set = validation_dataset.map(preprocess_validation_data, batched=True)

# -------------------------------
# 6. Training Arguments
# -------------------------------

# Define training arguments for student model
training_args = TrainingArguments(
    output_dir="./trained_student",
    logging_steps=10,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=12,
    weight_decay=0.01,
    save_total_limit=2,
    save_strategy="epoch",
    remove_unused_columns=False,
)

# -------------------------------
# 7. Custom Callback for Logging
# -------------------------------

class TeacherTrainingProgress(TrainerCallback):
    """
    Custom callback to log loss and epoch progress during training.
    """
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'loss' in logs:
            loss_values.append(logs['loss'])
            epoch_progress.append(logs['epoch'])
            
# -------------------------------
# 8. Custom Trainer for Distillation
# -------------------------------

class DistillationTrainer(Trainer):    
    """
    Custom Trainer that computes the distillation loss (KL divergence) between
    teacher's and student's predicted logits.
    """
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        
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

# -------------------------------
# 9. Custom Data Collator
# -------------------------------

class FullyCustomDataCollator:
    """
    Custom data collator to pad inputs and attach teacher logits for each batch.
    """
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

data_collator = FullyCustomDataCollator(tokenizer=student_tokenizer)

# -------------------------------
# 10. Train Student Model with Distillation
# -------------------------------

trainer = DistillationTrainer(
    model=student_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_set,
    data_collator=data_collator,
    callbacks=[TeacherTrainingProgress()]
)

train_result = trainer.train()

# -------------------------------
# 11. Evaluate Student After Distillation
# -------------------------------

print("Student Model Evaluation after Distillation")
result = evaluate_model(student_model, student_tokenizer, validation_dataset)
print(f"Exact Match: {result['exact_match']:.2f}%")
print(f"F1 Score: {result['f1']:.2f}%\n")

# Save the results for use in Graphs
exact_match_after = result['exact_match']
f1_score_after = result['f1']

# -------------------------------
# 12. Visualization
# -------------------------------

# Plot loss values per epoch
plt.figure(figsize=(8, 6))
plt.plot(epoch_progress, loss_values, label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Per Epoch")
plt.legend()
plt.grid()
plt.show()

# Bar plot: Exact Match before vs after distillation
plt.figure(figsize=(8, 6))
labels = ["Before Distillation", "After Distillation"]
exact_match_scores = [exact_match_before, exact_match_after]
plt.bar(labels, exact_match_scores, color=['blue', 'green'])
plt.ylabel("Exact Match (%)")
plt.title("Exact Match Score Before and After Distillation")
for i, v in enumerate(exact_match_scores):
    plt.text(i, v + 1, f"{v:.1f}%", ha='center', fontsize=12)
plt.show()

# Bar plot: F1 score before vs after distillation
plt.figure(figsize=(8, 6))
f1_scores = [f1_score_before, f1_score_after]
plt.bar(labels, f1_scores, color=['orange', 'red'])
plt.ylabel("F1 Score (%)")
plt.title("F1 Score Before and After Distillation")
for i, v in enumerate(f1_scores):
    plt.text(i, v + 1, f"{v:.1f}%", ha='center', fontsize=12)
plt.show()

# -------------------------------
# 13. Save the Distilled Model
# -------------------------------

student_model.save_pretrained("./distilled_model")
student_tokenizer.save_pretrained("./distilled_model")


