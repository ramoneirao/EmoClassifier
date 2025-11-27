import os
import random
import warnings
from typing import Tuple, List

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import BertModel, AutoTokenizer, get_linear_schedule_with_warmup

warnings.filterwarnings("ignore")


# -------------------------
# Utils / Seed
# -------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)


# -------------------------
# Data Processor
# -------------------------
class DataProcessor:
    """
    Carrega e prepara dados.
    Espera um CSV com coluna 'texto' e colunas one-hot das emoções na ordem:
    ['neutro','alegria','tristeza','raiva','medo','nojo','surpresa','confianca','antecipacao']
    Para multiclasse transformamos o one-hot em índice via argmax.
    """

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.emotion_columns = [
            'neutro', 'alegria', 'tristeza', 'raiva', 'medo',
            'nojo', 'surpresa', 'confianca', 'antecipacao'
        ]

    def load_data(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.data_path)
            return df
        except Exception as e:
            raise RuntimeError(f"Erro ao carregar dados: {e}")

    def prepare_data(self, df: pd.DataFrame, test_size: float = 0.3, random_state: int = 42
                    ) -> Tuple[pd.Series, pd.Series, np.ndarray, np.ndarray]:
        required_columns = ['texto'] + self.emotion_columns
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Colunas faltando no dataset: {missing}")

        X = df['texto']
        # converte one-hot para índice da classe (multiclasse)
        y = df[self.emotion_columns].values.argmax(axis=1).astype(np.int64)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        return X_train.reset_index(drop=True), X_val.reset_index(drop=True), y_train, y_val


# -------------------------
# Dataset
# -------------------------
class CustomDataset(Dataset):
    def __init__(self, texts: pd.Series, labels: np.ndarray, tokenizer: AutoTokenizer, max_len: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = str(self.texts.iloc[idx])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors='pt'
        )

        input_ids = inputs['input_ids'].squeeze(0)          # shape: (max_len,)
        attention_mask = inputs['attention_mask'].squeeze(0)
        token_type_ids = inputs.get('token_type_ids')
        if token_type_ids is not None:
            token_type_ids = token_type_ids.squeeze(0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,   # pode ser None
            'targets': torch.tensor(self.labels[idx], dtype=torch.long)
        }


# -------------------------
# Modelo
# -------------------------
class BERTClassifier(nn.Module):
    """
    BERT para classificação multiclasse usando embedding do CLS (last_hidden_state[:,0,:]).
    """

    def __init__(self, model_name: str = 'neuralmind/bert-base-portuguese-cased', num_classes: int = 9, dropout: float = 0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name, return_dict=True)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # Se token_type_ids for None, o bert aceita diretamente
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        # use CLS token embedding
        cls_emb = outputs.last_hidden_state[:, 0, :]  # shape: (batch, hidden_size)
        x = self.dropout(cls_emb)
        logits = self.classifier(x)
        return logits


# -------------------------
# Trainer
# -------------------------
class BERTTrainer:
    def __init__(
        self,
        model_name: str = 'neuralmind/bert-base-portuguese-cased',
        num_classes: int = 9,
        max_len: int = 256,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        device: torch.device = None,
        num_workers: int = 4
    ):
        self.model_name = model_name
        self.num_classes = num_classes
        self.max_len = max_len
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        # Em Windows, num_workers > 0 pode causar problemas dependendo do contexto.
        self.num_workers = num_workers if os.name != 'nt' else 0

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        # model
        self.model = BERTClassifier(model_name=self.model_name, num_classes=num_classes)
        self.model.to(self.device)

        # optimizer, criterion
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        # mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    def create_data_loaders(self, X_train, y_train, X_val, y_val):
        train_ds = CustomDataset(X_train, y_train, self.tokenizer, max_len=self.max_len)
        val_ds = CustomDataset(X_val, y_val, self.tokenizer, max_len=self.max_len)

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        return train_loader, val_loader

    def train(self, X_train, y_train, X_val, y_val, epochs: int = 3, save_dir: str = "bert_saved", patience: int = 3):
        os.makedirs(save_dir, exist_ok=True)
        # salvar tokenizer
        self.tokenizer.save_pretrained(save_dir)

        train_loader, val_loader = self.create_data_loaders(X_train, y_train, X_val, y_val)

        total_steps = len(train_loader) * epochs
        warmup_steps = int(0.1 * total_steps)
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        best_val_loss = float('inf')
        epochs_without_improve = 0
        history = {'train': [], 'val': []}

        for epoch in range(1, epochs + 1):
            print(f"\nÉpoca {epoch}/{epochs}")
            print("-" * 40)

            train_loss, train_preds, train_targets = self._train_one_epoch(train_loader, scheduler)
            val_loss, val_preds, val_targets = self._validate(val_loader)

            train_metrics = self.calculate_metrics(train_targets, train_preds)
            val_metrics = self.calculate_metrics(val_targets, val_preds)

            history['train'].append(train_metrics)
            history['val'].append(val_metrics)

            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Train F1: {train_metrics['f1']:.4f} | Val F1: {val_metrics['f1']:.4f}")
            print("Classification Report (Val):")
            print(classification_report(val_targets, val_preds, digits=4))

            # early stopping & save best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improve = 0
                model_path = os.path.join(save_dir, "best_model.pth")
                torch.save(self.model.state_dict(), model_path)
                print(f"Melhor modelo salvo em: {model_path}")
            else:
                epochs_without_improve += 1
                if epochs_without_improve >= patience:
                    print(f"Parando cedo: sem melhora por {patience} épocas.")
                    break

        return history

    def _train_one_epoch(self, train_loader, scheduler):
        self.model.train()
        total_loss = 0.0
        all_preds: List[int] = []
        all_targets: List[int] = []

        pbar = tqdm(train_loader, desc="Treinando", leave=False)
        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            token_type_ids = batch.get('token_type_ids')
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(self.device)
            targets = batch['targets'].to(self.device)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                loss = self.criterion(outputs, targets)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            scheduler.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            targs = targets.detach().cpu().numpy()

            all_preds.extend(preds.tolist())
            all_targets.extend(targs.tolist())

            pbar.set_postfix(loss=total_loss / (len(all_preds) / self.batch_size + 1e-8))

        avg_loss = total_loss / len(train_loader)
        return avg_loss, np.array(all_preds), np.array(all_targets)

    def _validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        all_preds: List[int] = []
        all_targets: List[int] = []

        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validando", leave=False)
            for batch in pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch.get('token_type_ids')
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(self.device)
                targets = batch['targets'].to(self.device)

                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                    loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                targs = targets.detach().cpu().numpy()

                all_preds.extend(preds.tolist())
                all_targets.extend(targs.tolist())

        avg_loss = total_loss / len(val_loader)
        return avg_loss, np.array(all_preds), np.array(all_targets)

    @staticmethod
    def calculate_metrics(targets: np.ndarray, predictions: np.ndarray):
        accuracy = accuracy_score(targets, predictions)
        precision = precision_score(targets, predictions, average='weighted', zero_division=0)
        recall = recall_score(targets, predictions, average='weighted', zero_division=0)
        f1 = f1_score(targets, predictions, average='weighted', zero_division=0)
        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

    def load_best_model(self, model_path: str):
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)

    def predict(self, texts: List[str], max_len: int = None, batch_size: int = 32):
        """
        Inferência em lista de textos: retorna índices de classe preditos.
        """
        self.model.eval()
        max_len = max_len or self.max_len
        ds = CustomDataset(pd.Series(texts), np.zeros(len(texts), dtype=np.int64), self.tokenizer, max_len=max_len)
        loader = DataLoader(ds, batch_size=batch_size, num_workers=self.num_workers, pin_memory=True)

        all_preds = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch.get('token_type_ids')
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                all_preds.extend(preds.tolist())
        return all_preds


# -------------------------
# Execução (main)
# -------------------------
if __name__ == "__main__":
    DATA_PATH = "data/data_balanceado.csv"
    set_seed(42)

    # Carregar e preparar
    processor = DataProcessor(DATA_PATH)
    df = processor.load_data()
    X_train, X_val, y_train, y_val = processor.prepare_data(df, test_size=0.2, random_state=42)

    trainer = BERTTrainer(
        model_name='neuralmind/bert-base-portuguese-cased',
        num_classes=9,
        max_len=256,
        batch_size=16,
        learning_rate=2e-5,
        num_workers=4
    )

    history = trainer.train(X_train, y_train, X_val, y_val, epochs=4, save_dir="models", patience=2)

    print("Treinamento finalizado. Histórico:")
    print(history)