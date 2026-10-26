"""
Continue Geneformer's masked-gene pretraining on a tokenized dataset (scripts.geneformer_tokenize).

Usage: python -m scripts.geneformer_finetune data=lung
"""
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("WANDB_MODE", "offline")

import hydra
import geneformer.perturber_utils as pu
from datasets import load_from_disk
from geneformer import TOKEN_DICTIONARY_FILE, GeneformerPretrainer
from transformers import TrainingArguments

REPO_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))


@hydra.main(config_path=str(REPO_ROOT / "config"), config_name="finetune_geneformer", version_base="1.3")
def main(cfg) -> None:
    dataset = load_from_disk(cfg.dataset)
    lengths_file = Path(cfg.dataset).with_name(f"{cfg.data.name}_lengths.pkl")
    if not lengths_file.exists():
        with open(lengths_file, "wb") as f:
            pickle.dump(list(dataset["length"]), f)
    with open(TOKEN_DICTIONARY_FILE, "rb") as f:
        token_dictionary = pickle.load(f)

    t = cfg.training
    out = Path(cfg.output_dir)
    args = TrainingArguments(
        output_dir=str(out / "training"),
        logging_dir=str(out / "logging"),
        learning_rate=t.learning_rate,
        lr_scheduler_type=t.lr_scheduler_type,
        warmup_steps=t.warmup_steps,
        weight_decay=t.weight_decay,
        per_device_train_batch_size=t.per_device_train_batch_size,
        num_train_epochs=t.num_train_epochs,
        save_strategy="steps",
        save_steps=max(1, len(dataset) // t.per_device_train_batch_size // t.saves_per_epoch),
        logging_steps=t.logging_steps,
        group_by_length=True,
        length_column_name="length",
        do_train=True,
        do_eval=False,
    )
    model = pu.load_model("Pretrained", None, cfg.model_dir, "train", quantize=False)
    trainer = GeneformerPretrainer(model=model, args=args, train_dataset=dataset,
                                   example_lengths_file=str(lengths_file), token_dictionary=token_dictionary)
    trainer.train()
    trainer.save_model(str(out))


if __name__ == "__main__":
    main()
