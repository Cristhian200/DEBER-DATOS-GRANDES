"""
Utilidades comunes para los proyectos de Machine Learning
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from typing import List, Tuple, Dict, Any

def set_seeds(seed: int = 42):
    """
    Establece seeds para reproducibilidad
    
    Args:
        seed: Valor del seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Para determinismo completo (puede ser más lento)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    """
    Obtiene el mejor device disponible
    
    Returns:
        torch.device: Device a usar (cuda/cpu)
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✅ GPU disponible: {torch.cuda.get_device_name(0)}")
        print(f"💾 Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device('cpu')
        print("💻 Usando CPU")
    
    return device

def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Cuenta parámetros del modelo
    
    Args:
        model: Modelo de PyTorch
        
    Returns:
        Dict con información de parámetros
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params,
        'size_mb': total_params * 4 / (1024 ** 2)  # Asumiendo float32
    }

def plot_training_history(train_losses: List[float], 
                         val_losses: List[float], 
                         train_accs: List[float], 
                         val_accs: List[float],
                         save_path: str = None):
    """
    Grafica el historial de entrenamiento
    
    Args:
        train_losses: Pérdidas de entrenamiento
        val_losses: Pérdidas de validación
        train_accs: Accuracies de entrenamiento
        val_accs: Accuracies de validación
        save_path: Ruta para guardar la figura
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Gráfico de pérdidas
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'bo-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'ro-', label='Val Loss', linewidth=2)
    ax1.set_title('📉 Pérdida Durante Entrenamiento', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfico de accuracy
    ax2.plot(epochs, train_accs, 'bo-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, val_accs, 'ro-', label='Val Acc', linewidth=2)
    ax2.set_title('📈 Precisión Durante Entrenamiento', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Gráfico guardado en: {save_path}")
    
    plt.show()

def save_checkpoint(model: torch.nn.Module, 
                   optimizer: torch.optim.Optimizer,
                   epoch: int,
                   loss: float,
                   accuracy: float,
                   filepath: str,
                   additional_info: Dict[str, Any] = None):
    """
    Guarda checkpoint del modelo
    
    Args:
        model: Modelo de PyTorch
        optimizer: Optimizador
        epoch: Época actual
        loss: Pérdida actual
        accuracy: Accuracy actual
        filepath: Ruta del archivo
        additional_info: Información adicional
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }
    
    if additional_info:
        checkpoint.update(additional_info)
    
    torch.save(checkpoint, filepath)
    print(f"💾 Checkpoint guardado: {filepath}")

def load_checkpoint(filepath: str, 
                   model: torch.nn.Module, 
                   optimizer: torch.optim.Optimizer = None):
    """
    Carga checkpoint del modelo
    
    Args:
        filepath: Ruta del archivo
        model: Modelo de PyTorch
        optimizer: Optimizador (opcional)
        
    Returns:
        Dict con información del checkpoint
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"📁 Checkpoint cargado: {filepath}")
    return checkpoint

class EarlyStopping:
    """
    Implementa early stopping para evitar overfitting
    """
    def __init__(self, patience: int = 7, min_delta: float = 0.001, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss: float, model: torch.nn.Module) -> bool:
        """
        Args:
            val_loss: Pérdida de validación actual
            model: Modelo de PyTorch
            
        Returns:
            True si debe parar el entrenamiento
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights:
                self.restore_checkpoint(model)
            return True
        return False
    
    def save_checkpoint(self, model: torch.nn.Module):
        """Guarda el mejor modelo"""
        self.best_weights = model.state_dict().copy()
    
    def restore_checkpoint(self, model: torch.nn.Module):
        """Restaura el mejor modelo"""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)

def print_model_summary(model: torch.nn.Module, input_size: Tuple[int, ...]):
    """
    Imprime resumen del modelo
    
    Args:
        model: Modelo de PyTorch
        input_size: Tamaño de entrada (sin batch dimension)
    """
    print("=" * 60)
    print("🏗️ RESUMEN DEL MODELO")
    print("=" * 60)
    
    # Información de parámetros
    param_info = count_parameters(model)
    print(f"📊 Parámetros totales: {param_info['total']:,}")
    print(f"🎯 Parámetros entrenables: {param_info['trainable']:,}")
    print(f"🔒 Parámetros no entrenables: {param_info['non_trainable']:,}")
    print(f"💾 Tamaño estimado: {param_info['size_mb']:.1f} MB")
    
    # Arquitectura del modelo
    print(f"\n🔧 Arquitectura:")
    print(model)
    
    print("=" * 60)
