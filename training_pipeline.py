import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import BertModel, AutoTokenizer
import warnings
warnings.filterwarnings("ignore")

class DataProcessor:
    """Classe para processar dados de entrada"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.emotion_columns = ['neutro', 'alegria', 'tristeza', 'raiva', 'medo', 
                               'nojo', 'surpresa', 'confianca', 'antecipacao']
    
    def load_data(self):
        """Carrega dados do CSV"""
        try:
            df = pd.read_csv(self.data_path)
            return df
        except Exception as e:
            print(f"Erro ao carregar dados: {e}")
            return None
    
    def prepare_data(self, df, test_size=0.3, random_state=42):
        """Prepara dados para treinamento"""
        # Verificar se as colunas necessárias existem
        required_columns = ['texto'] + self.emotion_columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Colunas faltando no dataset: {missing_columns}")
        
        # Preparar features e targets
        X = df['texto']
        # Para multiclasse: converter one-hot para índice da classe
        y = df[self.emotion_columns].values.argmax(axis=1)
        
        # Dividir em treino e validação
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        return X_train, X_val, y_train, y_val

class BERTClassifier(nn.Module):
    """Modelo BERT para classificação multilabel"""
    
    def __init__(self, num_classes=9, dropout=0.3):
        super(BERTClassifier, self).__init__()
        self.bert_model = BertModel.from_pretrained('neuralmind/bert-base-portuguese-cased', return_dict=True)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, num_classes)
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.bert_model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        output_dropout = self.dropout(output.pooler_output)
        output = self.linear(output_dropout)
        return output

class CustomDataset(torch.utils.data.Dataset):
    """Dataset customizado para BERT"""
    
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        text = str(self.texts.iloc[index])
        text = " ".join(text.split())
        
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'token_type_ids': inputs["token_type_ids"].flatten(),
            'targets': torch.LongTensor([self.labels[index]])  # Índice da classe
        }

class BERTTrainer:
    """Classe para treinar modelo BERT"""
    
    def __init__(self, num_classes=9, max_len=256, batch_size=16, learning_rate=1e-5):
        self.num_classes = num_classes
        self.max_len = max_len
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Inicializar tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased', 
            do_lower_case=False
        )
        
        # Inicializar modelo
        self.model = BERTClassifier(num_classes)
        self.model.to(self.device)
        
        # Otimizador e função de perda
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()  # Multiclasse
    
    def create_data_loaders(self, X_train, y_train, X_val, y_val):
        """Cria data loaders para treino e validação"""
        train_dataset = CustomDataset(X_train, y_train, self.tokenizer, self.max_len)
        val_dataset = CustomDataset(X_val, y_val, self.tokenizer, self.max_len)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader):
        """Treina por uma época"""
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            token_type_ids = batch['token_type_ids'].to(self.device)
            targets = batch['targets'].to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(input_ids, attention_mask, token_type_ids)
            loss = self.criterion(outputs, targets.squeeze())
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Coletar previsões e targets para métricas
            predictions = torch.argmax(outputs, dim=1).cpu().detach().numpy()
            targets_cpu = targets.squeeze().cpu().detach().numpy()
            
            all_predictions.extend(predictions)
            all_targets.extend(targets_cpu)
        
        return total_loss / len(train_loader), np.array(all_predictions), np.array(all_targets)
    
    def validate(self, val_loader):
        """Valida o modelo"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                targets = batch['targets'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask, token_type_ids)
                loss = self.criterion(outputs, targets.squeeze())
                
                total_loss += loss.item()
                
                predictions = torch.argmax(outputs, dim=1).cpu().detach().numpy()
                targets_cpu = targets.squeeze().cpu().detach().numpy()
                
                all_predictions.extend(predictions)
                all_targets.extend(targets_cpu)
        
        return total_loss / len(val_loader), np.array(all_predictions), np.array(all_targets)
    
    def train(self, X_train, y_train, X_val, y_val, epochs=3, save_path="bert_model.pth"):
        """Treina o modelo BERT"""
        train_loader, val_loader = self.create_data_loaders(X_train, y_train, X_val, y_val)
        
        best_val_loss = float('inf')
        train_history = []
        val_history = []
        
        for epoch in range(epochs):
            print(f"\nÉpoca {epoch + 1}/{epochs}")
            print("-" * 30)
            
            # Treinar
            train_loss, train_preds, train_targets = self.train_epoch(train_loader)
            
            # Validar
            val_loss, val_preds, val_targets = self.validate(val_loader)
            
            # Calcular métricas
            train_metrics = self.calculate_metrics(train_targets, train_preds)
            val_metrics = self.calculate_metrics(val_targets, val_preds)
            
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Train F1: {train_metrics['f1']:.4f} | Val F1: {val_metrics['f1']:.4f}")
            
            train_history.append(train_metrics)
            val_history.append(val_metrics)
            
            # Salvar melhor modelo
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), save_path)
                print(f"Modelo salvo em {save_path}")
        
        return train_history, val_history
    
    def calculate_metrics(self, targets, predictions):
        """Calcula métricas de avaliação"""
        accuracy = accuracy_score(targets, predictions)
        # Para multiclasse, usar 'weighted' ou 'macro'
        precision = precision_score(targets, predictions, average='weighted', zero_division=0)
        recall = recall_score(targets, predictions, average='weighted', zero_division=0)
        f1 = f1_score(targets, predictions, average='weighted', zero_division=0)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    
# Treinando o modelo BERT
if __name__ == "__main__":
    data_processor = DataProcessor(data_path='data/data_balanceado.csv')
    df = data_processor.load_data()
    
    if df is not None:
        X_train, X_val, y_train, y_val = data_processor.prepare_data(df)
        
        bert_trainer = BERTTrainer()
        train_history, val_history = bert_trainer.train(
            X_train, y_train, X_val, y_val, epochs=4, save_path="model.pth"
        )
        print("Treinamento concluído.")
        print("Histórico de Treinamento:", train_history)
        print("Histórico de Validação:", val_history)
    