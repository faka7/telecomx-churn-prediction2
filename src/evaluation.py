"""
Módulo de Evaluación de Modelos
Telecom X - Churn Prediction

Este módulo contiene funciones para evaluar el rendimiento de los modelos
de Machine Learning y generar reportes de métricas.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    roc_auc_score, precision_recall_curve, average_precision_score
)


class ModelEvaluator:
    """
    Clase para evaluar modelos de clasificación
    """
    
    def __init__(self, model_name='Model'):
        """
        Inicializa el evaluador
        
        Args:
            model_name: Nombre del modelo a evaluar
        """
        self.model_name = model_name
        self.metrics = {}
    
    def calculate_metrics(self, y_true, y_pred, y_proba=None):
        """
        Calcula todas las métricas de evaluación
        
        Args:
            y_true: Etiquetas verdaderas
            y_pred: Predicciones del modelo
            y_proba: Probabilidades predichas (opcional)
            
        Returns:
            Diccionario con métricas
        """
        self.metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='binary', zero_division=0)
        }
        
        if y_proba is not None:
            # Si y_proba es 2D, tomar la segunda columna (probabilidad de clase positiva)
            if len(y_proba.shape) > 1:
                y_proba = y_proba[:, 1]
            self.metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            self.metrics['avg_precision'] = average_precision_score(y_true, y_proba)
        
        return self.metrics
    
    def print_metrics(self):
        """
        Imprime las métricas calculadas
        """
        if not self.metrics:
            print("⚠️ No hay métricas calculadas. Ejecuta calculate_metrics() primero.")
            return
        
        print(f"\n{'='*50}")
        print(f"📊 MÉTRICAS DE EVALUACIÓN - {self.model_name}")
        print(f"{'='*50}")
        
        for metric, value in self.metrics.items():
            print(f"{metric.upper():20s}: {value:.4f}")
        
        print(f"{'='*50}\n")
    
    def plot_confusion_matrix(self, y_true, y_pred, figsize=(8, 6), save_path=None):
        """
        Visualiza la matriz de confusión
        
        Args:
            y_true: Etiquetas verdaderas
            y_pred: Predicciones del modelo
            figsize: Tamaño de la figura
            save_path: Ruta para guardar la figura (opcional)
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['No Churn', 'Churn'],
                    yticklabels=['No Churn', 'Churn'])
        
        plt.title(f'Matriz de Confusión - {self.model_name}', fontsize=14, fontweight='bold')
        plt.ylabel('Valor Real', fontsize=12)
        plt.xlabel('Predicción', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Matriz de confusión guardada en: {save_path}")
        
        plt.show()
    
    def plot_roc_curve(self, y_true, y_proba, figsize=(8, 6), save_path=None):
        """
        Visualiza la curva ROC
        
        Args:
            y_true: Etiquetas verdaderas
            y_proba: Probabilidades predichas
            figsize: Tamaño de la figura
            save_path: Ruta para guardar la figura (opcional)
        """
        # Si y_proba es 2D, tomar la segunda columna
        if len(y_proba.shape) > 1:
            y_proba = y_proba[:, 1]
        
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'Curva ROC - {self.model_name}', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Curva ROC guardada en: {save_path}")
        
        plt.show()
    
    def plot_precision_recall_curve(self, y_true, y_proba, figsize=(8, 6), save_path=None):
        """
        Visualiza la curva Precision-Recall
        
        Args:
            y_true: Etiquetas verdaderas
            y_proba: Probabilidades predichas
            figsize: Tamaño de la figura
            save_path: Ruta para guardar la figura (opcional)
        """
        # Si y_proba es 2D, tomar la segunda columna
        if len(y_proba.shape) > 1:
            y_proba = y_proba[:, 1]
        
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        avg_precision = average_precision_score(y_true, y_proba)
        
        plt.figure(figsize=figsize)
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'Precision-Recall (AP = {avg_precision:.2f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'Curva Precision-Recall - {self.model_name}', fontsize=14, fontweight='bold')
        plt.legend(loc="lower left")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Curva Precision-Recall guardada en: {save_path}")
        
        plt.show()
    
    def generate_classification_report(self, y_true, y_pred):
        """
        Genera un reporte de clasificación detallado
        
        Args:
            y_true: Etiquetas verdaderas
            y_pred: Predicciones del modelo
            
        Returns:
            String con el reporte
        """
        report = classification_report(y_true, y_pred, 
                                      target_names=['No Churn', 'Churn'],
                                      digits=4)
        
        print(f"\n{'='*60}")
        print(f"📋 REPORTE DE CLASIFICACIÓN - {self.model_name}")
        print(f"{'='*60}")
        print(report)
        
        return report


def compare_models(models_results, metric='f1_score', figsize=(12, 6), save_path=None):
    """
    Compara múltiples modelos visualmente
    
    Args:
        models_results: Diccionario con {nombre_modelo: métricas}
        metric: Métrica a comparar
        figsize: Tamaño de la figura
        save_path: Ruta para guardar la figura (opcional)
    """
    models_names = list(models_results.keys())
    scores = [results[metric] for results in models_results.values()]
    
    # Crear gráfico de barras
    plt.figure(figsize=figsize)
    bars = plt.bar(models_names, scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    
    # Añadir valores sobre las barras
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
    plt.title(f'Comparación de Modelos - {metric.replace("_", " ").title()}', 
             fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.ylim([0, 1.1])
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Comparación guardada en: {save_path}")
    
    plt.show()


def plot_feature_importance(importances, feature_names, top_n=20, figsize=(10, 8), save_path=None):
    """
    Visualiza la importancia de las características
    
    Args:
        importances: Array con importancias
        feature_names: Lista con nombres de características
        top_n: Número de características principales a mostrar
        figsize: Tamaño de la figura
        save_path: Ruta para guardar la figura (opcional)
    """
    # Crear DataFrame y ordenar
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)
    
    # Crear gráfico
    plt.figure(figsize=figsize)
    sns.barplot(data=importance_df, y='feature', x='importance', palette='viridis')
    
    plt.title(f'Top {top_n} Características Más Importantes', fontsize=14, fontweight='bold')
    plt.xlabel('Importancia', fontsize=12)
    plt.ylabel('Característica', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Importancia de características guardada en: {save_path}")
    
    plt.show()
    
    return importance_df


if __name__ == "__main__":
    print("Módulo de evaluación cargado correctamente ✅")
    print("Clases disponibles: ModelEvaluator")
    print("Funciones disponibles: compare_models, plot_feature_importance")
