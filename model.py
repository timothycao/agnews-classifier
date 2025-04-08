from transformers import RobertaForSequenceClassification
from peft import get_peft_model, LoraConfig


def create_lora_model(
    base_model='roberta-base',
    num_labels=4,
    r=2,
    alpha=4,
    dropout=0.05,
    bias='none',
    target_modules=['query']
):
    model = RobertaForSequenceClassification.from_pretrained(
        base_model,
        num_labels=num_labels
    )

    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias=bias,
        target_modules=target_modules,
        task_type='SEQ_CLS'
    )

    lora_model = get_peft_model(model, lora_config)
    lora_model.print_trainable_parameters()
    return lora_model


if __name__ == '__main__':
    # model = create_lora_model()
    model = create_lora_model(target_modules=['query', 'key', 'value'])