import torch
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import LightningModule
from torchinfo import summary
from torchmetrics.functional import accuracy, f1_score, matthews_corrcoef
from transformers import AutoConfig, AutoModelForSequenceClassification


def build_model(model, num_classes, dropout):
    model_config = AutoConfig.from_pretrained(model)
    model_config.num_labels = num_classes
    model_config.hidden_dropout_prob = dropout
    model_config.attention_probs_dropout_prob = dropout
    model_config.classifier_dropout = dropout * 2
    model = AutoModelForSequenceClassification.from_config(model_config)
    return model


class Model(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = build_model(config.model, config.num_classes, config.dropout)

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask=attention_mask)["logits"]

    def evaluate(self, batch, stage=None):
        x, attention_mask, y = (
            batch["input_ids"],
            batch["attention_mask"],
            batch["labels"],
        )
        output = self(input_ids=x, attention_mask=attention_mask)
        loss = F.cross_entropy(output, y)
        argmax_output = torch.argmax(output, dim=1)
        acc = accuracy(argmax_output, y)
        f1 = f1_score(argmax_output, y, num_classes=self.config.num_classes)
        mc = matthews_corrcoef(
            argmax_output, y, num_classes=self.config.num_classes
        ).float()
        if stage:
            self.log_dict(
                {
                    f"{stage}_loss": loss,
                    f"{stage}_acc": acc,
                    f"{stage}_f1": f1,
                    f"{stage}_mc": mc,
                }
            )
        return loss, acc, f1, mc

    def on_train_start(self):
        # log model parameters
        model_info = summary(self, input_data=self.model.dummy_inputs, verbose=0)
        self.log_dict(
            {
                "total_params": float(model_info.total_params),
                "trainable_params": float(model_info.trainable_params),
            },
            logger=True,
        )
        # log data split sizes
        datamodule = self.trainer.datamodule
        self.log_dict(
            {
                "train_size": float(len(datamodule.train_dataset)),
                "val_size": float(len(datamodule.val_dataset)),
                "test_size": float(len(datamodule.test_dataset)),
            },
            logger=True,
        )

    def training_step(self, batch, batch_idx):
        loss, acc, f1, mc = self.evaluate(batch, "train")
        return {"loss": loss, "train_acc": acc, "train_f1": f1, "train_mc": mc}

    def training_epoch_end(self, outputs):
        loss = torch.stack([i["loss"] for i in outputs]).mean()
        acc = torch.stack([i["train_acc"] for i in outputs]).mean()
        f1 = torch.stack([i["train_f1"] for i in outputs]).mean()
        mc = torch.stack([i["train_mc"] for i in outputs]).mean()
        self.log_dict(
            {
                "avg_train_loss": loss,
                "avg_train_acc": acc,
                "avg_train_f1": f1,
                "avg_train_mc": mc,
            },
            logger=True,
            prog_bar=True,
        )

    def validation_step(self, batch, batch_idx):
        loss, acc, f1, mc = self.evaluate(batch, "val")
        return {"val_loss": loss, "val_acc": acc, "val_f1": f1, "val_mc": mc}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([i["val_loss"] for i in outputs]).mean()
        acc = torch.stack([i["val_acc"] for i in outputs]).mean()
        f1 = torch.stack([i["val_f1"] for i in outputs]).mean()
        mc = torch.stack([i["val_mc"] for i in outputs]).mean()
        self.log_dict(
            {
                "avg_val_loss": loss,
                "avg_val_acc": acc,
                "avg_val_f1": f1,
                "avg_val_mc": mc,
            },
            logger=True,
            prog_bar=True,
        )

    def test_step(self, batch, batch_idx):
        loss, acc, f1, mc = self.evaluate(batch, "test")
        return {"test_loss": loss, "test_acc": acc, "test_f1": f1, "test_mc": mc}

    def test_epoch_end(self, outputs):
        loss = torch.stack([i["test_loss"] for i in outputs]).mean()
        acc = torch.stack([i["test_acc"] for i in outputs]).mean()
        f1 = torch.stack([i["test_f1"] for i in outputs]).mean()
        mc = torch.stack([i["test_mc"] for i in outputs]).mean()
        self.log_dict(
            {
                "avg_test_loss": loss,
                "avg_test_acc": acc,
                "avg_test_f1": f1,
                "avg_test_mc": mc,
            },
            logger=True,
            prog_bar=True,
        )

    def configure_optimizers(self):
        training_steps = self.config.max_epochs * len(
            self.trainer.datamodule.train_dataloader()
        )
        warmup_steps = int(training_steps * 0.1)
        opt = optim.AdamW(
            self.parameters(), lr=self.config.lr, weight_decay=self.config.wd, eps=1e-6
        )
        sch = self.get_linear_schedule_with_warmup(opt, warmup_steps, training_steps)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sch, "interval": "step", "frequency": 1},
        }

    def get_linear_schedule_with_warmup(
        self, optimizer, num_warmup_steps, num_training_steps, last_epoch=-1
    ):
        """
        From https://github.com/uds-lsv/bert-stable-fine-tuning/blob/3cf27e8667aee9d1c822d747c63ce331b231f283/src/transformers/optimization.py

        Create a schedule with a learning rate that decreases linearly after
        linearly increasing during a warmup period.
        """

        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0,
                float(num_training_steps - current_step)
                / float(max(1, num_training_steps - num_warmup_steps)),
            )

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)
