import os

from datasets import load_dataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, default_data_collator

glue_input_template = {
    "cola": ("sentence", None),
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "rte": ("sentence1", "sentence2"),
}


class GLUEDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_workers = os.cpu_count()

    def prepare_data(self):
        # download data
        load_dataset("glue", self.config.dataset)

    def setup(self, stage=None):
        # download dataset
        dataset = load_dataset("glue", self.config.dataset)
        keys = glue_input_template[self.config.dataset]
        # tokenization
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        def preprocess_function(example):
            texts = (
                (example[keys[0]],)
                if keys[1] is None
                else (example[keys[0]], example[keys[1]])
            )
            return tokenizer(
                *texts,
                padding="max_length",
                max_length=self.config.max_seq_len,
                truncation=True,
            )

        preprocessed_dataset = dataset.map(preprocess_function, batched=True)
        self.train_dataset = preprocessed_dataset["train"]
        self.val_dataset = preprocessed_dataset["validation"]
        self.test_dataset = preprocessed_dataset["test"]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=default_data_collator,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=default_data_collator,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=default_data_collator,
            num_workers=self.num_workers,
            pin_memory=True,
        )
