import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import RobertaTokenizer, DataCollatorWithPadding, RobertaForSequenceClassification
from peft import PeftModel


def preprocess_function(examples, tokenizer):
    return tokenizer(examples['text'], truncation=True, padding=True)


def load_data(data_path, tokenizer):    
    # Load serialized and unlabeled Hugging Face dataset
    dataset = pd.read_pickle(data_path)

    # Tokenize the dataset (same way as done during training)
    tokenized_dataset = dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True,  remove_columns=['text'])

    return tokenized_dataset


def load_model(checkpoint, device):
    # Load base BERT model
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=4)

    # Load checkpoint weights on top of base model
    lora_model = PeftModel.from_pretrained(model, checkpoint)

    # Move model to device
    lora_model.to(device)
    
    # Set model to evaluation mode
    lora_model.eval()
    
    return lora_model


def inference(dataset, data_collator, model, device, batch_size=32):
    # Create DataLoader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)

    print('Running inference...')
    predictions = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            batch_predictions = outputs.logits.argmax(dim=-1).cpu().tolist()
            predictions.extend(batch_predictions)
    
    return predictions


def save_predictions(predictions, checkpoint, output_dir):
    # Ensure given output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create output path
    checkpoint_name = os.path.basename(checkpoint)
    filename = f'predictions_{checkpoint_name}.csv'
    output_path = os.path.join(output_dir, filename)

    # Save predictions to CSV in output directory
    df = pd.DataFrame({'ID': range(len(predictions)), 'Label': predictions})
    df.to_csv(output_path, index=False)
    print(f'Predictions saved to {output_path}')


def main(data_path, checkpoint, output_dir):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    # Load and preprocess unlabeled dataset
    dataset = load_data(data_path, tokenizer)

    # Create data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors='pt')

    # Load model
    model = load_model(checkpoint, device)

    # Run inference
    predictions = inference(dataset, data_collator, model, device)

    # Save predictions
    save_predictions(predictions, checkpoint, output_dir)


if __name__ == '__main__':
    data_path = 'test_unlabeled.pkl'

    checkpoint = 'saved_models/checkpoint-100'

    output_dir = 'saved_predictions'

    main(data_path, checkpoint, output_dir)