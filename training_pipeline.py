"""
Pipeline de Treinamento para Classificação de Emoções
Suporte para modelos BERT e SVM
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
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
        y = df[self.emotion_columns]
        
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
            'targets': torch.FloatTensor(self.labels.iloc[index].values)
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
        self.criterion = nn.BCEWithLogitsLoss()
    
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
            loss = self.criterion(outputs, targets)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Coletar previsões e targets para métricas
            predictions = torch.sigmoid(outputs).cpu().detach().numpy()
            targets_cpu = targets.cpu().detach().numpy()
            
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
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                
                predictions = torch.sigmoid(outputs).cpu().detach().numpy()
                targets_cpu = targets.cpu().detach().numpy()
                
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
            train_metrics = self.calculate_metrics(train_targets, train_preds > 0.5)
            val_metrics = self.calculate_metrics(val_targets, val_preds > 0.5)
            
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
        precision = precision_score(targets, predictions, average='micro', zero_division=0)
        recall = recall_score(targets, predictions, average='micro', zero_division=0)
        f1 = f1_score(targets, predictions, average='micro', zero_division=0)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

class SVMTrainer:
    """Classe para treinar modelo SVM"""
    
    def __init__(self, max_features=10000, C=1.0):
        self.max_features = max_features
        self.C = C
        
        # Pipeline SVM com TF-IDF
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=max_features, stop_words='english')),
            ('classifier', MultiOutputClassifier(LinearSVC(C=C, random_state=42)))
        ])
    
    def train(self, X_train, y_train, save_path="svm_model.pkl"):
        """Treina o modelo SVM"""
        print("Treinando modelo SVM...")
        
        # Converter para array numpy se necessário
        if hasattr(X_train, 'values'):
            X_train_array = X_train.values
        else:
            X_train_array = X_train
            
        if hasattr(y_train, 'values'):
            y_train_array = y_train.values
        else:
            y_train_array = y_train
        
        # Treinar modelo
        self.model.fit(X_train_array, y_train_array)
        
        # Salvar modelo
        joblib.dump(self.model, save_path)
        print(f"Modelo SVM salvo em {save_path}")
    
    def predict(self, X):
        """Faz previsões"""
        return self.model.predict(X)
    
    def evaluate(self, X_val, y_val):
        """Avalia o modelo"""
        predictions = self.predict(X_val)
        
        accuracy = accuracy_score(y_val, predictions)
        precision = precision_score(y_val, predictions, average='micro', zero_division=0)
        recall = recall_score(y_val, predictions, average='micro', zero_division=0)
        f1 = f1_score(y_val, predictions, average='micro', zero_division=0)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

class ModelComparator:
    """Classe para comparar modelos"""
    
    def __init__(self):
        self.results = {}
    
    def compare_models(self, bert_trainer, svm_trainer, X_val, y_val):
        """Compara os dois modelos"""
        print("\n" + "="*50)
        print("COMPARAÇÃO DE MODELOS")
        print("="*50)
        
        # Avaliar BERT
        print("\nAvaliando BERT...")
        bert_val_loader = torch.utils.data.DataLoader(
            CustomDataset(X_val, y_val, bert_trainer.tokenizer),
            batch_size=bert_trainer.batch_size,
            shuffle=False
        )
        _, bert_preds, bert_targets = bert_trainer.validate(bert_val_loader)
        bert_metrics = bert_trainer.calculate_metrics(bert_targets, bert_preds > 0.5)
        
        # Avaliar SVM
        print("Avaliando SVM...")
        svm_metrics = svm_trainer.evaluate(X_val, y_val)
        
        # Armazenar resultados
        self.results = {
            'BERT': bert_metrics,
            'SVM': svm_metrics
        }
        
        # Exibir comparação
        self.display_comparison()
        
        return self.results
    
    def display_comparison(self):
        """Exibe comparação dos modelos"""
        print("\nRESULTADOS:")
        print("-" * 40)
        
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        for metric in metrics:
            print(f"\n{metric.upper()}:")
            for model_name, model_results in self.results.items():
                print(f"  {model_name}: {model_results[metric]:.4f}")
        
        # Determinar melhor modelo
        bert_f1 = self.results['BERT']['f1']
        svm_f1 = self.results['SVM']['f1']
        
        best_model = 'BERT' if bert_f1 > svm_f1 else 'SVM'
        print(f"\n🏆 MELHOR MODELO: {best_model}")

class TrainingPipeline:
    """Pipeline principal que integra tudo"""
    
    def __init__(self, data_path, test_size=0.3, random_state=42):
        self.data_processor = DataProcessor(data_path)
        self.test_size = test_size
        self.random_state = random_state
    
    def run_full_pipeline(self, bert_epochs=3, svm_C=1.0):
        """Executa pipeline completo de treinamento"""
        print("INICIANDO PIPELINE DE TREINAMENTO")
        print("="*50)
        
        # 1. Carregar e preparar dados
        print("\n1. Carregando dados...")
        df = self.data_processor.load_data()
        if df is None:
            return None
        
        print(f"Dataset carregado: {len(df)} amostras")
        
        # 2. Preparar dados
        print("\n2. Preparando dados...")
        X_train, X_val, y_train, y_val = self.data_processor.prepare_data(df, self.test_size, self.random_state)
        
        print(f"Treino: {len(X_train)} amostras")
        print(f"Validação: {len(X_val)} amostras")
        
        # 3. Treinar BERT
        print("\n3. Treinando BERT...")
        bert_trainer = BERTTrainer()
        bert_history = bert_trainer.train(X_train, y_train, X_val, y_val, epochs=bert_epochs)
        
        # 4. Treinar SVM
        print("\n4. Treinando SVM...")
        svm_trainer = SVMTrainer(C=svm_C)
        svm_trainer.train(X_train, y_train)
        
        # 5. Comparar modelos
        print("\n5. Comparando modelos...")
        comparator = ModelComparator()
        results = comparator.compare_models(bert_trainer, svm_trainer, X_val, y_val)
        
        print("\nPIPELINE CONCLUÍDO!")
        
        return {
            'bert_trainer': bert_trainer,
            'svm_trainer': svm_trainer,
            'results': results,
            'data': (X_train, X_val, y_train, y_val)
        }

# if __name__ == "__main__":
#     # Exemplo de uso
#     pipeline = TrainingPipeline("data/data.csv")
#     results = pipeline.run_full_pipeline(bert_epochs=2, svm_C=1.0)