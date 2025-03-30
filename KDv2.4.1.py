# **Initial Setup**

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict

from datasets import load_dataset
from evaluate import load
from tqdm import tqdm

# **Load Models**
# Load Teacher and Student Models
#teacher_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
teacher_model_name = "Jesujuwon/distilgpt2-squad" #82M
#student_model_name = "Locutusque/TinyMistral-248M"
student_model_name = "tniranjan/finetuned_tinystories_33M_pretrained_tinystories_ta"

teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name)
student_model = AutoModelForCausalLM.from_pretrained(student_model_name)

teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)

# **Dataset Processing**

# Preprocessing function
def preprocess_batch(batch, tokenizer, max_length=256):
    # Extract questions and contexts from the batch
    questions = [example["question"] for example in batch]
    contexts = [example["context"] for example in batch]
    
    # Tokenize context and question
    inputs = tokenizer(
        questions,
        contexts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )
    
    # Extract answer text (use the first answer for simplicity)
    answer_texts = [
        example["answers"]["text"][0] if len(example["answers"]["text"]) > 0 else "" 
        for example in batch
    ]
    
    # Tokenize answers
    labels = tokenizer(
        answer_texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )["input_ids"]
    
    # Add labels to inputs
    inputs["labels"] = labels
    
    return inputs

# **Load Dataset**

# Load SQuAD1.1 Dataset
splits = {'train': 'plain_text/train-00000-of-00001.parquet', 'validation': 'plain_text/validation-00000-of-00001.parquet'}
train_df = pd.read_parquet("hf://datasets/rajpurkar/squad/" + splits["train"])
validation_df = pd.read_parquet("hf://datasets/rajpurkar/squad/" + splits["validation"])

# **Process Dataset**

# Need to reduce the size of the train dataset to make testing much faster.
# We can use the full dataset again once we know the code is working.

##### Dataset size reduction code here #####
reduced_train_df = train_df.sample(frac=0.05, random_state=42)
reduced_validation_df = validation_df.sample(frac=0.05, random_state=42)

print(f"Original train size: {len(train_df)}")
print(f"Original valid size: {len(validation_df)}")
print(f"Reduced train size: {len(reduced_train_df)}")
print(f"Reduced valid size: {len(reduced_validation_df)}")

full_train_df = train_df
train_df = reduced_train_df

full_validation_df = validation_df
validation_df = reduced_validation_df

# Convert DataFrame to a list of dictionaries for batch processing
train_data = train_df.to_dict(orient="records")
validation_data = validation_df.to_dict(orient="records")

# Create a PyTorch Dataset
class QADataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx]
        }

# Preprocess data in batches
def process_dataset(data, tokenizer):
    processed_data = []
    batch_size = 32

    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        processed_batch = preprocess_batch(batch, tokenizer)
        processed_data.append(processed_batch)

    # Combine all processed batches into a single dataset
    input_ids = torch.cat([batch["input_ids"] for batch in processed_data])
    attention_mask = torch.cat([batch["attention_mask"] for batch in processed_data])
    labels = torch.cat([batch["labels"] for batch in processed_data])

    dataset = QADataset(input_ids, attention_mask, labels)

    return dataset

train_dataset = process_dataset(train_data, student_tokenizer)
validation_dataset = process_dataset(validation_data, student_tokenizer)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(validation_dataset, batch_size=4, shuffle=True)

# ** Evaluate the pre-distillation performance of both Teacher and Student model and display results **

#### Evaluation Code Here ####

#### Evaluation Code Here ####


# Load SQuAD metric
squad_metric = load("squad")

# Evaluation function
def evaluate_model(model, tokenizer, data, max_length=256, device="cuda"):
    # Ensure pad_token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Use EOS token as PAD token if undefined
    #model.config.pad_token_id = tokenizer.pad_token_id
    model.config.pad_token_id = tokenizer.eos_token_id
    
    model.eval()
    model.to(device)

    predictions = []
    references = []

    print(f"Evaluating on full validation set: {len(data)} examples")
    
    for example in tqdm(data):
        question = example["question"]
        context = example["context"]
        true_answers = example["answers"]["text"]

        input_text = f"{question} {tokenizer.sep_token} {context}"
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_length=512,
                do_sample=False
            )

        pred_answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        predictions.append({"id": example["id"], "prediction_text": pred_answer})
        references.append({"id": example["id"], "answers": example["answers"]})

    results = squad_metric.compute(predictions=predictions, references=references)
    print(f"Exact Match (EM): {results['exact_match']:.2f}")
    print(f"F1 Score: {results['f1']:.2f}")
    return results


# ** Evaluate the post-distillation performance of the Student model and display results **
print("==== Pre-Distillation Evaluation ====")

print("\n🔹 Teacher Model:")
evaluate_model(teacher_model, teacher_tokenizer, validation_data)

print("\n🔹 Student Model:")
evaluate_model(student_model, student_tokenizer, validation_data)
#### Results Display Code HERE ####

# **Teacher Vocab is much larger than Students so need to configure a projection layer so both tensors will align**

## This code needs a lot of memory ( > 8GB )
## So have changed to the Vocabulary Aligner code instead which uses much less RAM
# Define a projection layer to align teacher logits with student logits
# class TeacherLogitProjection(nn.Module):
#     def __init__(self, teacher_vocab_size, student_vocab_size):
#         super(TeacherLogitProjection, self).__init__()
#         self.projection = nn.Linear(teacher_vocab_size, student_vocab_size)

#     def forward(self, teacher_logits):
#         return self.projection(teacher_logits)

class VocabularyAligner:
    def __init__(self, teacher_tokenizer, student_tokenizer):
        # Create token-to-ID mappings
        teacher_vocab = teacher_tokenizer.get_vocab()
        student_vocab = student_tokenizer.get_vocab()
        
        # Build alignment mapping
        self.alignment_matrix = defaultdict(lambda: student_tokenizer.unk_token_id)
        for token, tid in teacher_vocab.items():
            if token in student_vocab:
                self.alignment_matrix[tid] = student_vocab[token]
                
    def project_logits(self, teacher_preds):
        # Initialize aligned predictions with unknown token ID
        aligned_preds = torch.full_like(teacher_preds, fill_value=self.alignment_matrix.default_factory())
        
        # Map each teacher token ID to its corresponding student token ID
        for tid in range(teacher_preds.max().item() + 1):  # Iterate over all possible teacher token IDs
            mask = (teacher_preds == tid)  # Mask where teacher predicts this token ID
            aligned_preds[mask] = self.alignment_matrix[tid]  # Map to student token ID
        
        return aligned_preds

## This code is designed for use with the Projection layer, it is not appropriate for use with the Vocab Aligner method
# Define Distillation Loss
# class DistillationLoss(nn.Module):
#     def __init__(self, alpha=0.5, temperature=2.0):
#         super(DistillationLoss, self).__init__()
#         self.alpha = alpha  # Weight for hard vs soft targets
#         self.temperature = temperature  # Temperature for softening logits
#         self.ce_loss = nn.CrossEntropyLoss()

#     def forward(self, student_logits, teacher_logits, labels):
#         # Softened teacher logits
#         teacher_probs = nn.functional.softmax(teacher_logits / self.temperature, dim=-1)
#         student_probs = nn.functional.log_softmax(student_logits / self.temperature, dim=-1)

#         # KL Divergence loss (soft targets)
#         distillation_loss = nn.functional.kl_div(student_probs, teacher_probs, reduction="batchmean") * (self.temperature ** 2)

#         # Cross-entropy loss (hard targets)
#         ce_loss = self.ce_loss(student_logits.view(-1, student_logits.size(-1)), labels.view(-1))

#         return self.alpha * ce_loss + (1 - self.alpha) * distillation_loss


# **Knowledge Distillation Training Loop**

def train_student_with_distillation(teacher_model, student_model, train_loader, epochs=3):
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    student_device = "cuda"
    teacher_device = "cuda"
    # Initialize alignment
    aligner = VocabularyAligner(teacher_tokenizer, student_tokenizer)
    
    # Move models to device
    teacher_model.to(teacher_device)
    student_model.to(student_device)
    
    # Use gradient checkpointing to save memory
    teacher_model.gradient_checkpointing_enable()
    
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=5e-5)
    
    for epoch in range(epochs):
        print(f"Starting epoch {epoch + 1}")
        student_model.train()

        batch_count = 1
        for batch in train_loader:
            print(f"\rBatch: {batch_count}/21900", end="", flush=True)
            inputs_ids = batch["input_ids"].to(student_device)
            attention_mask = batch["attention_mask"].to(student_device)
            #labels = batch["labels"].to(student_device)
            
            with torch.no_grad():
                # Get teacher predictions (no full logits)
                teacher_outputs = teacher_model(input_ids=inputs_ids, attention_mask=attention_mask)
                teacher_preds = torch.argmax(teacher_outputs.logits, dim=-1)
                
            # Project teacher predictions to student vocab
            aligned_teacher_preds = aligner.project_logits(teacher_preds).to(teacher_device)
            
            # Student forward pass
            student_outputs = student_model(input_ids=inputs_ids, attention_mask=attention_mask)
            
            # Calculate loss using aligned predictions
            # print("Logits shape:", student_outputs.logits.shape)
            # print("Flattened logits shape:", student_outputs.logits.view(-1, student_model.config.vocab_size).shape)
            # print("Teacher predictions shape:", teacher_preds.shape)
            # print("Aligned teacher predictions shape:", aligned_teacher_preds.shape)
            # print("Flattened teacher predictions shape:", aligned_teacher_preds.view(-1).shape)
            
            loss = nn.CrossEntropyLoss()(
                student_outputs.logits.view(-1, student_model.config.vocab_size),  # Flatten logits
                aligned_teacher_preds.view(-1)  # Flatten targets
            )
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            batch_count += 1
    
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")


# Run Training
train_student_with_distillation(
    teacher_model,
    student_model,
    train_loader,
    epochs=3
)

#### Results Display Code HERE ####
print("\n==== Post-Distillation Evaluation ====")

print("\n🔹 Student Model:")
evaluate_model(student_model )
