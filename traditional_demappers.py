"""
Traditional Demapping Methods for NOMA
Implements Successive Interference Cancellation (SIC) and Joint Maximum Likelihood (ML) demappers.
"""

import numpy as np


class TraditionalDemappers:
    """
    Collection of traditional demapping methods for comparison with neural approaches.
    """
    
    @staticmethod
    def sic_demapper(received, h, alpha, beta, constellation):
        """
        Successive Interference Cancellation (SIC) demapper.
        
        Strategy:
        1. Decode stronger user (higher power) first
        2. Subtract its interference from received signal
        3. Decode weaker user from remaining signal
        
        Args:
            received: Received signal array
            h: Channel coefficients matrix (2 x n_symbols)
            alpha: Power coefficient for user 1 (near user)
            beta: Power coefficient for user 2 (far user)
            constellation: Modulation constellation points
            
        Returns:
            detected_bits1: Detected bits for user 1
            detected_bits2: Detected bits for user 2
        """
        M = len(constellation)
        bits_per_symbol = int(np.log2(M))
        n_symbols = len(received)
        
        detected_bits1 = []
        detected_bits2 = []
        
        for t in range(n_symbols):
            # Step 1: Decode User 1 (stronger, with alpha)
            # Equalize channel for user 1
            y1_eq = received[t] / h[0, t]
            
            # Find closest constellation point for user 1
            distances1 = np.abs(y1_eq - constellation)
            closest_idx1 = np.argmin(distances1)
            
            # Convert index to bits
            bits1 = [(closest_idx1 >> b) & 1 for b in range(bits_per_symbol-1, -1, -1)]
            detected_bits1.extend(bits1)
            
            # Step 2: Subtract user 1's interference
            # Reconstruct user 1's signal
            s1_est = constellation[closest_idx1]
            
            # Subtract from received signal
            remaining = received[t] - h[0, t] * alpha * s1_est
            
            # Step 3: Decode User 2
            # Equalize for user 2 and remove power scaling
            y2_eq = remaining / (h[1, t] * beta)
            
            # Find closest constellation point for user 2
            distances2 = np.abs(y2_eq - constellation)
            closest_idx2 = np.argmin(distances2)
            
            # Convert index to bits
            bits2 = [(closest_idx2 >> b) & 1 for b in range(bits_per_symbol-1, -1, -1)]
            detected_bits2.extend(bits2)
        
        return np.array(detected_bits1), np.array(detected_bits2)
    
    @staticmethod
    def joint_ml_demapper(received, h, alpha, beta, constellation):
        """
        Joint Maximum Likelihood (ML) demapper.
        
        Searches over all possible symbol pairs to find the most likely combination.
        Optimal performance but computationally expensive (O(M²) per symbol).
        
        Args:
            received: Received signal array
            h: Channel coefficients matrix (2 x n_symbols)
            alpha: Power coefficient for user 1
            beta: Power coefficient for user 2
            constellation: Modulation constellation points
            
        Returns:
            detected_bits1: Detected bits for user 1
            detected_bits2: Detected bits for user 2
        """
        M = len(constellation)
        bits_per_symbol = int(np.log2(M))
        n_symbols = len(received)
        
        detected_bits1 = []
        detected_bits2 = []
        
        for t in range(n_symbols):
            min_dist = float('inf')
            best_pair = (0, 0)
            
            # Search over all possible symbol pairs
            for i, s1 in enumerate(constellation):
                for j, s2 in enumerate(constellation):
                    # Reconstruct expected received signal
                    reconstructed = h[0, t] * alpha * s1 + h[1, t] * beta * s2
                    
                    # Calculate squared distance
                    dist = np.abs(received[t] - reconstructed) ** 2
                    
                    if dist < min_dist:
                        min_dist = dist
                        best_pair = (i, j)
            
            # Convert indices to bits
            i, j = best_pair
            bits1 = [(i >> b) & 1 for b in range(bits_per_symbol-1, -1, -1)]
            bits2 = [(j >> b) & 1 for b in range(bits_per_symbol-1, -1, -1)]
            
            detected_bits1.extend(bits1)
            detected_bits2.extend(bits2)
        
        return np.array(detected_bits1), np.array(detected_bits2)
    
    @staticmethod
    def mmse_demapper(received, h, alpha, beta, constellation, noise_variance):
        """
        Minimum Mean Square Error (MMSE) demapper.
        
        Uses linear MMSE filtering before symbol detection.
        
        Args:
            received: Received signal array
            h: Channel coefficients matrix
            alpha: Power coefficient for user 1
            beta: Power coefficient for user 2
            constellation: Modulation constellation points
            noise_variance: Noise power
            
        Returns:
            detected_bits1: Detected bits for user 1
            detected_bits2: Detected bits for user 2
        """
        M = len(constellation)
        bits_per_symbol = int(np.log2(M))
        n_symbols = len(received)
        
        detected_bits1 = []
        detected_bits2 = []
        
        for t in range(n_symbols):
            # Construct channel matrix for this time instance
            H = np.array([[h[0, t] * alpha, h[1, t] * beta]])
            
            # MMSE filter for user 1
            # W_mmse = H^H * (H*H^H + noise_variance*I)^(-1)
            H_H = np.conj(H).T
            denominator = H @ H_H + noise_variance
            W1 = (H_H / denominator).flatten()
            
            # Apply MMSE filter for user 1
            y1_mmse = W1[0] * received[t]
            
            # Find closest constellation point
            distances1 = np.abs(y1_mmse - constellation)
            closest_idx1 = np.argmin(distances1)
            
            # Convert to bits
            bits1 = [(closest_idx1 >> b) & 1 for b in range(bits_per_symbol-1, -1, -1)]
            detected_bits1.extend(bits1)
            
            # SIC-like cancellation for user 2
            s1_est = constellation[closest_idx1]
            remaining = received[t] - h[0, t] * alpha * s1_est
            
            # MMSE for user 2 on remaining signal
            H2 = np.array([[h[1, t] * beta]])
            H2_H = np.conj(H2).T
            denominator2 = H2 @ H2_H + noise_variance
            W2 = (H2_H / denominator2).flatten()
            
            y2_mmse = W2[0] * remaining
            
            # Detect user 2
            distances2 = np.abs(y2_mmse - constellation)
            closest_idx2 = np.argmin(distances2)
            
            bits2 = [(closest_idx2 >> b) & 1 for b in range(bits_per_symbol-1, -1, -1)]
            detected_bits2.extend(bits2)
        
        return np.array(detected_bits1), np.array(detected_bits2)


if __name__ == "__main__":
    # Simple test
    from noma_system import NOMASystem
    
    # Create test data
    noma = NOMASystem(snr_db=20)
    constellation = noma.constellation
    
    # Simple test case
    received = np.array([0.5+0.5j, -0.3+0.8j])
    h = np.array([[0.8+0.2j, 0.7-0.1j], [0.3+0.5j, 0.4-0.3j]])
    alpha, beta = np.sqrt(0.8), np.sqrt(0.2)
    
    # Test SIC
    bits1, bits2 = TraditionalDemappers.sic_demapper(received, h, alpha, beta, constellation)
    print(f"SIC demapper test successful!")
    print(f"User 1 bits: {bits1[:4]}")
    print(f"User 2 bits: {bits2[:4]}")
