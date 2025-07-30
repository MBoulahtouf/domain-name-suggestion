#!/usr/bin/env python3
"""
Baseline model implementation for domain name suggestions.
"""

import json
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset
import random

class DomainSuggestionDataset(Dataset):
    """Custom dataset for domain suggestion training."""
    
    def __init__(self, data_path, tokenizer, max_length=128):
        """
        Initialize the dataset.
        
        Args:
            data_path (str): Path to the JSON data file
            tokenizer: Tokenizer for the model
            max_length (int): Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        with open(data_path, 'r') as f:
            self.data = json.load(f)
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get a single sample from the dataset."""
        sample = self.data[idx]
        description = sample['business_description']
        suggestions = sample['suggestions']
        
        # For baseline, we'll use the first suggestion as the target
        if suggestions:
            target_domain = suggestions[0]['domain']
        else:
            target_domain = "example.com"
        
        # Format input text
        input_text = f"Business: {description} -> Domain:"
        target_text = f" {target_domain}"
        
        # Tokenize input
        input_encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Tokenize target
        target_encoding = self.tokenizer(
            input_text + target_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = input_encoding['input_ids'].squeeze()
        attention_mask = input_encoding['attention_mask'].squeeze()
        labels = target_encoding['input_ids'].squeeze()
        
        # Mask the input part in labels
        input_len = len(input_encoding['input_ids'].squeeze())
        labels[:input_len] = -100  # Ignore input tokens in loss calculation
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

class DomainSuggestionModel:
    """Domain suggestion model based on GPT-2."""
    
    def __init__(self, model_name='gpt2'):
        """
        Initialize the model.
        
        Args:
            model_name (str): Name of the pre-trained model to use
        """
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        
        # Add padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.eos_token_id
    
    def train(self, train_data_path, eval_data_path, output_dir='./model_output'):
        """
        Train the model.
        
        Args:
            train_data_path (str): Path to training data
            eval_data_path (str): Path to evaluation data
            output_dir (str): Directory to save the trained model
        """
        # Create datasets
        train_dataset = DomainSuggestionDataset(train_data_path, self.tokenizer)
        eval_dataset = DomainSuggestionDataset(eval_data_path, self.tokenizer)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            warmup_steps=100,
            logging_steps=50,
            save_steps=500,
            evaluation_strategy="steps",
            eval_steps=500,
            prediction_loss_only=True,
            remove_unused_columns=False,
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        
        # Train the model
        trainer.train()
        
        # Save the model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
    
    def generate_suggestions(self, business_description, num_suggestions=3, max_length=20):
        """
        Generate domain suggestions for a business description.
        
        Args:
            business_description (str): Description of the business
            num_suggestions (int): Number of suggestions to generate
            max_length (int): Maximum length of generated domain name
            
        Returns:
            list: List of domain suggestions with confidence scores
        """
        # Format input text
        input_text = f"Business: {business_description} -> Domain:"
        
        # Tokenize input
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt')
        
        suggestions = []
        
        # Generate multiple suggestions
        for _ in range(num_suggestions):
            # Generate text
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    max_length=len(input_ids[0]) + max_length,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.8,
                    top_k=50,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode the generated text
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract domain suggestion
            if '-> Domain:' in generated_text:
                domain_part = generated_text.split('-> Domain:')[-1].strip()
                # Simple post-processing to extract domain-like text
                domain = ''.join(c for c in domain_part if c.isalnum() or c in '.-')
                # Ensure it has a valid extension
                if '.' not in domain:
                    domain += '.com'
            else:
                # Fallback domain
                domain = "example.com"
            
            # Simple confidence score (random for baseline)
            confidence = round(random.uniform(0.6, 0.9), 2)
            
            suggestions.append({
                'domain': domain,
                'confidence': confidence
            })
        
        return suggestions

def main():
    """Main function to train the baseline model."""
    print("Initializing baseline model...")
    model = DomainSuggestionModel()
    
    print("Training the model...")
    model.train(
        train_data_path='data/train_data.json',
        eval_data_path='data/eval_data.json',
        output_dir='./model_output'
    )
    
    print("Model training completed. Testing with sample input...")
    
    # Test the model
    test_description = "organic coffee shop in downtown area"
    suggestions = model.generate_suggestions(test_description)
    
    print(f"Business: {test_description}")
    print("Suggestions:")
    for suggestion in suggestions:
        print(f"  - {suggestion['domain']} (confidence: {suggestion['confidence']})")

if __name__ == "__main__":
    main()