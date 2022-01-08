import re

from transformers import AutoTokenizer,AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from datasets import Dataset, ClassLabel, Sequence, Features, Value, load_dataset
    



def tokenize_and_align_labels(tokenizer,examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def main(dataset_name: str = "./output/conll2003-sk-ner", split:str = "test", model_name: str = "gerulata/slovakbert",  batch_size: int = 16, num_train_epochs=3) :
    task = "ner"  # only ner is supported by this dataset
    
    datasetDict = load_dataset(dataset_name)
    dataset = datasetDict[split]
    labels_count = dataset.features["ner_tags"].feature.num_classes

    tag_class_label = dataset.features["ner_tags"].feature

    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=labels_count)

    #Now tokenize and align the labels over the entire dataset with Datasets map function:
    tokenized_train_dateset = datasetDict[split].map(lambda data: tokenize_and_align_labels(tokenizer,data), batched=True)
    #tokenized_eval_dateset = datasetDict["validation"].map(lambda data: tokenize_and_align_labels(tokenizer,data), batched=True)
    data_collator = DataCollatorForTokenClassification(tokenizer)

    args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy = "epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=1e-5,
    )



    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train_dateset,
        #eval_dataset=tokenized_eval_dateset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()


if __name__ == '__main__':
    main()