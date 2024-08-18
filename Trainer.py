import csv
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset

class T5Trainer:
    def __init__(self, model_name='t5-base', training_csv_path=None):
        self.t5_model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.t5_tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.training_csv_path = training_csv_path

    def load_training_data(self):
        sources = []
        targets = []
        with open(self.training_csv_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                sources.append(row['source'])
                targets.append(row['target'])
        return sources, targets

    def prepare_dataset(self, sources, targets):
        data = {"source": sources, "target": targets}
        dataset = Dataset.from_dict(data)
        return dataset

    def preprocess_function(self, examples):
        inputs = [ex for ex in examples["source"]]
        targets = [ex for ex in examples["target"]]
        model_inputs = self.t5_tokenizer(inputs, max_length=512, truncation=True, padding="max_length")

        with self.t5_tokenizer.as_target_tokenizer():
            labels = self.t5_tokenizer(targets, max_length=512, truncation=True, padding="max_length")

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def fine_tune(self, output_dir='./fine_tuned_t5', num_train_epochs=3):
        sources, targets = self.load_training_data()
        train_dataset = self.prepare_dataset(sources, targets)
        tokenized_train_dataset = train_dataset.map(self.preprocess_function, batched=True)

        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="no",
            learning_rate=2e-5,
            per_device_train_batch_size=24,
            num_train_epochs=num_train_epochs,  # Increase number of epochs
            weight_decay=0.01,
            fp16=True,
            gradient_accumulation_steps=1,
            dataloader_num_workers=4,
            logging_steps=10,  # Log more frequently for debugging
            save_steps=10000,
            report_to="none",
        )

        data_collator = DataCollatorForSeq2Seq(self.t5_tokenizer, model=self.t5_model)

        trainer = Trainer(
            model=self.t5_model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            data_collator=data_collator,
        )

        trainer.train()

        self.t5_model.save_pretrained(output_dir)
        self.t5_tokenizer.save_pretrained(output_dir)

    def load_fine_tuned_model(self, path='./fine_tuned_t5'):
        self.t5_model = T5ForConditionalGeneration.from_pretrained(path)
        self.t5_tokenizer = T5Tokenizer.from_pretrained(path)

    def extract_entities(self, prompt):
        input_text = f"Extract entities: {prompt}"
        input_ids = self.t5_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

        output_ids = self.t5_model.generate(input_ids, max_length=150, num_beams=4, early_stopping=True)
        output_text = self.t5_tokenizer.decode(output_ids[0], skip_special_tokens=True)

        print("Raw output from T5 model:", output_text)

        entities = {
            "day_of_month": None,
            "month_of_year": None,
            "year": None,
            "ip_address": None,
            "action": None,
            "visited_page": None
        }

        entity_map = {
            "Day of month": "day_of_month",
            "Month of year": "month_of_year",
            "Year": "year",
            "IP address": "ip_address",
            "Action": "action",
            "Visited page": "visited_page"
        }

        for entity, key in entity_map.items():
            if f"{entity}:" in output_text:
                value = output_text.split(f"{entity}:")[1].split("\n")[0].strip()
                if "None" not in value:
                    entities[key] = value

        print("Extracted Entities:", entities)
        return entities
