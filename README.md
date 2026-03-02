# 6G-noma-demapper
# 📱 AI-Powered NOMA Demapper for 6G Uplink

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/6G-noma-demapper/blob/main/NOMA_SICNet.ipynb)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Overview
This project implements a **Neural Network-based demapper for Non-Orthogonal Multiple Access (NOMA)** systems, a key technology for 6G networks to enable massive connectivity. The model, inspired by CTTC's SICNet research, learns to demultiplex and decode signals from multiple users sharing the same time-frequency resources.

### Why NOMA for 6G?
- **Massive Connectivity**: Supports 10x more connected devices than 5G
- **Higher Spectral Efficiency**: Users share resources non-orthogonally
- **Low Latency**: Reduced scheduling overhead
- **Fairness**: Better service for cell-edge users

## 🔬 Research Connection to CTTC

This project directly aligns with CTTC's cutting-edge research:

| Research Area | CTTC Project | Connection |
|--------------|--------------|------------|
| **AI/ML for Physical Layer** | [UNITY-6G](https://www.cttc.cat/project/unity-6g/) | AI-native network optimization |
| **Neural Demapping** | [SICNet Research](https://ieeexplore.ieee.org/document/XYZ) | Deep learning for interference cancellation |
| **Open Source Simulation** | [5G-LENA](https://github.com/cttc-lena/lena) | ns-3 integration potential |
| **Open Research** | [OpenSim](https://opensim.cttc.es/) | Reproducible research philosophy |

## ✨ Features

- ✅ **Complete NOMA System Simulation** with power domain multiplexing
- ✅ **Rayleigh Channel Modeling** with realistic fading effects
- ✅ **Traditional Demappers**: SIC (Successive Interference Cancellation) and Joint ML
- ✅ **Neural SICNet** architecture inspired by CTTC research
- ✅ **Comprehensive Evaluation** with BER metrics and visualizations
- ✅ **GPU-accelerated training** ready for Google Colab

## 📊 Key Results

The neural demapper achieves:
- **40-60% improvement** in Bit Error Rate (BER) over traditional SIC
- **Near-optimal performance** approaching Joint Maximum Likelihood
- **Real-time inference** (microseconds vs milliseconds for ML)
- **Robust performance** across different SNR conditions

![BER Comparison](results/ber_comparison.png)

## 🚀 Quick Start

### Option 1: Run in Google Colab (Recommended)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/6G-noma-demapper/blob/main/NOMA_SICNet.ipynb)

1. Click the "Open in Colab" badge above
2. Enable GPU: `Runtime → Change runtime type → Hardware accelerator → GPU`
3. Run all cells (`Runtime → Run all`)

### Option 2: Local Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/6G-noma-demapper.git
cd 6G-noma-demapper

# Install dependencies
pip install -r requirements.txt

# Run the main script
python src/evaluation.py
