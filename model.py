from transformers import RobertaForSequenceClassification
from peft import get_peft_model, LoraConfig


def create_lora_model(lora_config):
    # Load base BERT model
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=4)

    # Create LoRA-adapted model
    lora_model = get_peft_model(model, lora_config)
    
    # Print number of trainable parameters
    lora_model.print_trainable_parameters()
    
    return lora_model


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