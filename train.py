import os
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from transformers import RobertaTokenizer, DataCollatorWithPadding, TrainingArguments, TrainerCallback, Trainer
from peft import LoraConfig
from model import create_lora_model


# HF Trainer does not expose predictions or labels during training (not accessible via callbacks)
# So this subclass overrides `compute_loss` and `log` methods to manually accumulate them and compute training accuracy
# Reference discussion: https://discuss.huggingface.co/t/metrics-for-training-set-in-trainer/2461
# Original compute_loss method: https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py#L3778
class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        # Initialize parent Trainer class
        super().__init__(*args, **kwargs)
        
        # Create lists to accumulate predictions and labels across steps
        self.train_preds, self.train_labels = [], []

    # Override default compute loss method to reuse its logits and labels
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Perform forward pass (for logits and loss)
        outputs = model(**inputs)
        
        # Accumulate predictions and labels
        if 'labels' in inputs and 'logits' in outputs:
            labels = inputs['labels'].detach().cpu().tolist()
            preds = outputs.logits.argmax(dim=-1).detach().cpu().tolist()
            self.train_preds.extend(preds)
            self.train_labels.extend(labels)

        # Compute and return loss as usual
        loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
        return (loss, outputs) if return_outputs else loss

    # Override default log method to add training accuracy
    def log(self, logs, *args, **kwargs):
        if self.model.training and self.train_preds and self.train_labels:
            # Compute and log training accuracy for current logging interval
            logs['accuracy'] = accuracy_score(self.train_labels, self.train_preds)
            
            # Reset stored predictions and labels for next interval
            self.train_preds, self.train_labels = [], []
        
        # Call original log method to log default metrics
        super().log(logs, *args, **kwargs)


# Callback to save flattened and processed log history csvs for each checkpoint
class CustomCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        output_dir = os.path.join(args.output_dir, f'checkpoint-{state.global_step}')
        
        # Flatten training and evaluation logs by step
        flattened_log_history = {}
        for log in state.log_history:
            step = log.get('step')
            if step is None:
                continue

            if step not in flattened_log_history:
                flattened_log_history[step] = {}
            
            # Merge logs with the same step
            for key, value in log.items():
                flattened_log_history[step][key] = value

        # Convert to DataFrame and sort by step
        df = pd.DataFrame(list(flattened_log_history.values()))
        df.sort_values(by='step', inplace=True)

        # Save to csv file in the checkpoint directory
        df.to_csv(os.path.join(output_dir, 'log_history.csv'), index=False)

        # Select columns for processed logs
        desired_columns = [
            'step',         
            'loss',         
            'eval_loss',   
            'accuracy',     
            'eval_accuracy',
            'learning_rate',
            'epoch' 
        ]

        # Copy only the desired columns to a new DataFrame
        processed_logs_df = df[desired_columns].copy()

        # Rename columns for clarity and readability
        rename_mapping = {
            'step': 'Step',
            'loss': 'Train Loss',
            'eval_loss': 'Test Loss',
            'accuracy': 'Train Acc',
            'eval_accuracy': 'Test Acc',
            'learning_rate': 'Learning Rate',
            'epoch': 'Epochs'
        }

        # Rename columns in the DataFrame
        processed_logs_df.rename(columns=rename_mapping, inplace=True)

        # Add additional columns for loss spread/ratio and accuracy spread/ratio 
        if 'Train Loss' in processed_logs_df.columns and 'Test Loss' in processed_logs_df.columns:
            processed_logs_df['Loss Spread'] = processed_logs_df['Train Loss'] - processed_logs_df['Test Loss']
            processed_logs_df['Loss Ratio'] = processed_logs_df['Train Loss'] / processed_logs_df['Test Loss']

        if 'Train Acc' in processed_logs_df.columns and 'Test Acc' in processed_logs_df.columns:
            processed_logs_df['Acc Spread'] = processed_logs_df['Train Acc'] - processed_logs_df['Test Acc']
            processed_logs_df['Acc Ratio'] = processed_logs_df['Train Acc'] / processed_logs_df['Test Acc']

        # Save to csv file in the checkpoint directory
        processed_logs_df.to_csv(os.path.join(output_dir, 'processed_log_history.csv'), index=False)


def preprocess_function(examples, tokenizer):
    return tokenizer(examples['text'], truncation=True, padding=True)


def load_data(tokenizer, test_size=640, seed=42):
    # Load AG News dataset
    dataset = load_dataset('ag_news', split='train')

    # Tokenize the dataset
    tokenized_dataset = dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True,  remove_columns=['text'])
    
    # Rename 'label' column to 'labels' for compatibility with Hugging Face's Trainer
    tokenized_dataset = tokenized_dataset.rename_column('label', 'labels')

    # Split the dataset into train and test sets
    split_datasets = tokenized_dataset.train_test_split(test_size=test_size, seed=seed)

    return split_datasets['train'], split_datasets['test']


def compute_eval_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = accuracy_score(labels, preds)
    return {'accuracy': accuracy}


def main(model, training_args, checkpoint=None):
    # Load tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    # Load and preprocess dataset
    train_dataset, eval_dataset = load_data(tokenizer)

    # Create data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors='pt')

    # Initialize Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        compute_metrics=compute_eval_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[CustomCallback()]
    )

    # Train the model
    print('Starting training...')
    if checkpoint:
        print(f'Resuming from checkpoint: {checkpoint}')
    trainer.train(resume_from_checkpoint=checkpoint)


if __name__ == '__main__':
    lora_config = LoraConfig(
        r=2,
        lora_alpha=4,
        lora_dropout=0.05,
        bias='none',
        target_modules=['query', 'key', 'value'],
        task_type='SEQ_CLS'
    )

    model = create_lora_model(lora_config)
    
    training_args = TrainingArguments(
        # Core training configs
        # num_train_epochs=1,
        max_steps=100,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        optim='adamw_torch',
        learning_rate=5e-5,

        # Logging, evaluation, and checkpointing
        logging_strategy='steps',
        logging_steps=20,
        eval_strategy='steps',
        eval_steps=20,
        output_dir='saved_models',
        save_strategy='steps',
        save_steps=20,

        # Miscellaneous
        report_to='none',
        dataloader_num_workers=4,
        gradient_checkpointing=False,
        gradient_checkpointing_kwargs={'use_reentrant':True}
    )
    
    main(model, training_args)
    # main(model, training_args, checkpoint='saved_models/checkpoint-100')