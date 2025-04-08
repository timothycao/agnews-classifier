from datasets import load_dataset
from sklearn.metrics import accuracy_score
from transformers import RobertaTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer
from peft import LoraConfig
from model import create_lora_model


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


def compute_metrics(pred):
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
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
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
        max_steps=200,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        optim='adamw_torch',
        learning_rate=5e-5,

        # Logging, evaluation, and checkpointing
        logging_strategy='steps',
        logging_steps=100,
        eval_strategy='steps',
        eval_steps=100,
        output_dir='saved_models',
        save_strategy='steps',
        save_steps=100,

        # Miscellaneous
        report_to='none',
        dataloader_num_workers=4,
        gradient_checkpointing=False,
        gradient_checkpointing_kwargs={'use_reentrant':True}
    )
    
    main(model, training_args)
    # main(model, training_args, checkpoint='saved_models/checkpoint-100')