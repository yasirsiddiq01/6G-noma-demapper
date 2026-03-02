"""
Evaluation Module for NOMA Demappers
Provides functions for BER calculation, performance comparison, and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import time
from tqdm import tqdm

# Import project modules
from noma_system import NOMASystem
from traditional_demappers import TraditionalDemappers
from sicnet_model import build_sicnet, compile_sicnet


def calculate_ber(true_bits, estimated_bits):
    """
    Calculate Bit Error Rate (BER).
    
    Args:
        true_bits: Ground truth bits
        estimated_bits: Estimated/detected bits
        
    Returns:
        ber: Bit Error Rate
    """
    # Ensure same length
    min_len = min(len(true_bits), len(estimated_bits))
    true_bits = true_bits[:min_len]
    estimated_bits = estimated_bits[:min_len]
    
    errors = np.sum(true_bits != estimated_bits)
    total = len(true_bits)
    
    return errors / total if total > 0 else 0.0


def evaluate_sicnet(model, X_test, y1_test, y2_test, threshold=0.5):
    """
    Evaluate SICNet model on test data.
    
    Args:
        model: Trained SICNet model
        X_test: Test features
        y1_test: User 1 true bits
        y2_test: User 2 true bits
        threshold: Probability threshold for bit decision
        
    Returns:
       
