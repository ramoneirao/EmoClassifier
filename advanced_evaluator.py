"""
Classe avançada de avaliação e comparação de modelos
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, hamming_loss
)
import torch

class AdvancedModelEvaluator:
    """Classe para avaliação avançada de modelos multilabel"""
    
    def __init__(self, emotion_labels=None):
        if emotion_labels is None:
            self.emotion_labels = ['neutro', 'alegria', 'tristeza', 'raiva', 'medo', 
                                  'nojo', 'surpresa', 'confianca', 'antecipacao']
        else:
            self.emotion_labels = emotion_labels
        
        self.results_history = {}
    
    def evaluate_bert_model(self, bert_trainer, val_loader, model_name="BERT"):
        """Avalia modelo BERT detalhadamente"""
        print(f"\nAvaliando {model_name}...")
        
        bert_trainer.model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(bert_trainer.device)
                attention_mask = batch['attention_mask'].to(bert_trainer.device)
                token_type_ids = batch['token_type_ids'].to(bert_trainer.device)
                targets = batch['targets'].to(bert_trainer.device)
                
                outputs = bert_trainer.model(input_ids, attention_mask, token_type_ids)
                probabilities = torch.sigmoid(outputs)
                
                all_probabilities.extend(probabilities.cpu().numpy())
                all_predictions.extend((probabilities > 0.5).cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        probabilities = np.array(all_probabilities)
        
        return self._calculate_detailed_metrics(
            targets, predictions, probabilities, model_name
        )
    
    def evaluate_svm_model(self, svm_trainer, X_val, y_val, model_name="SVM"):
        """Avalia modelo SVM detalhadamente"""
        print(f"\nAvaliando {model_name}...")
        
        # Fazer previsões
        predictions = svm_trainer.predict(X_val)
        
        # Para SVM, usamos as previsões como probabilidades (0 ou 1)
        probabilities = predictions.astype(float)
        
        # Converter targets para numpy se necessário
        if hasattr(y_val, 'values'):
            targets = y_val.values
        else:
            targets = y_val
        
        return self._calculate_detailed_metrics(
            targets, predictions, probabilities, model_name
        )
    
    def _calculate_detailed_metrics(self, targets, predictions, probabilities, model_name):
        """Calcula métricas detalhadas"""
        # Métricas globais
        global_metrics = {
            'accuracy': accuracy_score(targets, predictions),
            'precision_micro': precision_score(targets, predictions, average='micro', zero_division=0),
            'recall_micro': recall_score(targets, predictions, average='micro', zero_division=0),
            'f1_micro': f1_score(targets, predictions, average='micro', zero_division=0),
            'precision_macro': precision_score(targets, predictions, average='macro', zero_division=0),
            'recall_macro': recall_score(targets, predictions, average='macro', zero_division=0),
            'f1_macro': f1_score(targets, predictions, average='macro', zero_division=0),
            'hamming_loss': hamming_loss(targets, predictions)
        }
        
        # Métricas por classe
        class_metrics = {}
        for i, emotion in enumerate(self.emotion_labels):
            class_metrics[emotion] = {
                'precision': precision_score(targets[:, i], predictions[:, i], zero_division=0),
                'recall': recall_score(targets[:, i], predictions[:, i], zero_division=0),
                'f1': f1_score(targets[:, i], predictions[:, i], zero_division=0),
                'support': np.sum(targets[:, i])
            }
        
        # Armazenar resultados
        self.results_history[model_name] = {
            'global_metrics': global_metrics,
            'class_metrics': class_metrics,
            'predictions': predictions,
            'targets': targets,
            'probabilities': probabilities
        }
        
        return self.results_history[model_name]
    
    def compare_models(self, models_results):
        """Compara múltiplos modelos"""
        print("\n" + "="*60)
        print("COMPARAÇÃO DETALHADA DE MODELOS")
        print("="*60)
        
        # Comparação de métricas globais
        self._compare_global_metrics(models_results)
        
        # Comparação por classe
        self._compare_class_metrics(models_results)
        
        # Determinar melhor modelo
        self._determine_best_model(models_results)
    
    def _compare_global_metrics(self, models_results):
        """Compara métricas globais"""
        print("\nMÉTRICAS GLOBAIS:")
        print("-" * 40)
        
        # Criar DataFrame para comparação
        comparison_data = []
        for model_name, results in models_results.items():
            metrics = results['global_metrics']
            comparison_data.append({
                'Modelo': model_name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'F1-Micro': f"{metrics['f1_micro']:.4f}",
                'F1-Macro': f"{metrics['f1_macro']:.4f}",
                'Precision-Micro': f"{metrics['precision_micro']:.4f}",
                'Recall-Micro': f"{metrics['recall_micro']:.4f}",
                'Hamming Loss': f"{metrics['hamming_loss']:.4f}"
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        print(df_comparison.to_string(index=False))
    
    def _compare_class_metrics(self, models_results):
        """Compara métricas por classe"""
        print("\nMÉTRICAS POR EMOÇÃO:")
        print("-" * 40)
        
        for emotion in self.emotion_labels:
            print(f"\n{emotion.upper()}:")
            for model_name, results in models_results.items():
                metrics = results['class_metrics'][emotion]
                print(f"  {model_name:>8}: F1={metrics['f1']:.3f} | "
                      f"Precision={metrics['precision']:.3f} | "
                      f"Recall={metrics['recall']:.3f} | "
                      f"Support={int(metrics['support'])}")
    
    def _determine_best_model(self, models_results):
        """Determina o melhor modelo"""
        print("\nRANKING DOS MODELOS:")
        print("-" * 30)
        
        # Ranking por F1-micro
        f1_scores = {}
        for model_name, results in models_results.items():
            f1_scores[model_name] = results['global_metrics']['f1_micro']
        
        sorted_models = sorted(f1_scores.items(), key=lambda x: x[1], reverse=True)
        
        for i, (model_name, f1_score) in enumerate(sorted_models, 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
            print(f"{medal} {i}º lugar: {model_name} (F1-micro: {f1_score:.4f})")
    
    def generate_confusion_matrices(self, models_results, save_path=None):
        """Gera matrizes de confusão para cada modelo"""
        print("\nGerando matrizes de confusão...")
        
        n_models = len(models_results)
        n_emotions = len(self.emotion_labels)
        
        fig, axes = plt.subplots(n_models, n_emotions, figsize=(20, 6*n_models))
        
        if n_models == 1:
            axes = axes.reshape(1, -1)
        
        for model_idx, (model_name, results) in enumerate(models_results.items()):
            targets = results['targets']
            predictions = results['predictions']
            
            for emotion_idx, emotion in enumerate(self.emotion_labels):
                cm = confusion_matrix(targets[:, emotion_idx], predictions[:, emotion_idx])
                
                ax = axes[model_idx, emotion_idx]
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_title(f'{model_name} - {emotion}')
                ax.set_xlabel('Predito')
                ax.set_ylabel('Real')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Matrizes de confusão salvas em: {save_path}")
        
        plt.show()
    
    def generate_performance_report(self, models_results, save_path=None):
        """Gera relatório completo de performance"""
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("RELATÓRIO COMPLETO DE AVALIAÇÃO DE MODELOS")
        report_lines.append("="*80)
        
        for model_name, results in models_results.items():
            report_lines.append(f"\nMODELO: {model_name}")
            report_lines.append("-" * 50)
            
            # Métricas globais
            global_metrics = results['global_metrics']
            report_lines.append("\nMétricas Globais:")
            for metric, value in global_metrics.items():
                report_lines.append(f"  {metric:>15}: {value:.4f}")
            
            # Métricas por classe
            report_lines.append(f"\nMétricas por Emoção:")
            class_metrics = results['class_metrics']
            for emotion, metrics in class_metrics.items():
                report_lines.append(f"\n  {emotion.upper()}:")
                for metric, value in metrics.items():
                    if metric == 'support':
                        report_lines.append(f"    {metric:>10}: {int(value)}")
                    else:
                        report_lines.append(f"    {metric:>10}: {value:.4f}")
        
        # Comparação final
        report_lines.append("\n" + "="*80)
        report_lines.append("COMPARAÇÃO FINAL")
        report_lines.append("="*80)
        
        # Tabela comparativa
        comparison_data = []
        for model_name, results in models_results.items():
            metrics = results['global_metrics']
            comparison_data.append([
                model_name,
                f"{metrics['accuracy']:.4f}",
                f"{metrics['f1_micro']:.4f}",
                f"{metrics['f1_macro']:.4f}",
                f"{metrics['hamming_loss']:.4f}"
            ])
        
        df_comp = pd.DataFrame(comparison_data, 
                              columns=['Modelo', 'Accuracy', 'F1-Micro', 'F1-Macro', 'Hamming Loss'])
        report_lines.append("\n" + df_comp.to_string(index=False))
        
        # Melhor modelo
        best_model = max(models_results.items(), 
                        key=lambda x: x[1]['global_metrics']['f1_micro'])
        report_lines.append(f"\nMELHOR MODELO: {best_model[0]} (F1-micro: {best_model[1]['global_metrics']['f1_micro']:.4f})")
        
        # Salvar relatório
        full_report = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(full_report)
            print(f"\nRelatório salvo em: {save_path}")
        
        print(full_report)
        
        return full_report
    
    def plot_emotion_distribution(self, models_results, save_path=None):
        """Plota distribuição de emoções previstas vs reais"""
        print("\nGerando gráficos de distribuição...")
        
        n_models = len(models_results)
        fig, axes = plt.subplots(n_models, 2, figsize=(15, 5*n_models))
        
        if n_models == 1:
            axes = axes.reshape(1, -1)
        
        for model_idx, (model_name, results) in enumerate(models_results.items()):
            targets = results['targets']
            predictions = results['predictions']
            
            # Distribuição real
            real_counts = np.sum(targets, axis=0)
            pred_counts = np.sum(predictions, axis=0)
            
            # Gráfico de distribuição real
            ax1 = axes[model_idx, 0]
            bars1 = ax1.bar(self.emotion_labels, real_counts, alpha=0.7, color='blue')
            ax1.set_title(f'{model_name} - Distribuição Real')
            ax1.set_ylabel('Frequência')
            ax1.tick_params(axis='x', rotation=45)
            
            # Adicionar valores nas barras
            for bar in bars1:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom')
            
            # Gráfico de distribuição predita
            ax2 = axes[model_idx, 1]
            bars2 = ax2.bar(self.emotion_labels, pred_counts, alpha=0.7, color='red')
            ax2.set_title(f'{model_name} - Distribuição Predita')
            ax2.set_ylabel('Frequência')
            ax2.tick_params(axis='x', rotation=45)
            
            # Adicionar valores nas barras
            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gráficos de distribuição salvos em: {save_path}")
        
        plt.show()
    
    def create_comprehensive_evaluation(self, bert_trainer, svm_trainer, X_val, y_val, save_dir="evaluation_results"):
        """Cria avaliação completa dos modelos"""
        import os
        
        # Criar diretório de resultados
        os.makedirs(save_dir, exist_ok=True)
        
        print("INICIANDO AVALIAÇÃO COMPLETA DOS MODELOS")
        print("="*60)
        
        # Avaliar BERT
        bert_val_loader = torch.utils.data.DataLoader(
            CustomDataset(X_val, y_val, bert_trainer.tokenizer),
            batch_size=bert_trainer.batch_size,
            shuffle=False
        )
        bert_results = self.evaluate_bert_model(bert_trainer, bert_val_loader)
        
        # Avaliar SVM
        svm_results = self.evaluate_svm_model(svm_trainer, X_val, y_val)
        
        # Comparar modelos
        all_results = {
            'BERT': bert_results,
            'SVM': svm_results
        }
        
        # Gerar comparação
        self.compare_models(all_results)
        
        # Gerar visualizações
        self.generate_confusion_matrices(
            all_results, 
            save_path=os.path.join(save_dir, "confusion_matrices.png")
        )
        
        self.plot_emotion_distribution(
            all_results,
            save_path=os.path.join(save_dir, "emotion_distribution.png")
        )
        
        # Gerar relatório
        self.generate_performance_report(
            all_results,
            save_path=os.path.join(save_dir, "performance_report.txt")
        )
        
        print(f"\nAvaliação completa salva em: {save_dir}")
        
        return all_results

# Importar a classe CustomDataset do pipeline principal
from training_pipeline import CustomDataset