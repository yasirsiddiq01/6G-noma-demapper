"""
NOMA System Simulation Module
Implements power-domain NOMA with two users, including modulation,
power allocation, and channel effects.
"""

import numpy as np


class NOMASystem:
    """
    Simulates an uplink NOMA system with two users sharing the same
    time-frequency resources but separated by power levels.
    """
    
    def __init__(self, num_users=2, modulation='QPSK', snr_db=20):
        """
        Initialize NOMA system.
        
        Args:
            num_users: Number of users (default: 2)
            modulation: Modulation scheme ('QPSK' or '16QAM')
            snr_db: Signal-to-Noise Ratio in dB
        """
        self.num_users = num_users
        self.snr_db = snr_db
        self.set_modulation(modulation)
        
    def set_modulation(self, modulation):
        """
        Set modulation scheme and constellation.
        
        Args:
            modulation: 'QPSK' or '16QAM'
        """
        if modulation == 'QPSK':
            # QPSK constellation (4 points)
            self.constellation = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
            self.bits_per_symbol = 2
        elif modulation == '16QAM':
            # 16QAM constellation (16 points)
            symbols = []
            for i in [-3, -1, 1, 3]:
                for q in [-3, -1, 1, 3]:
                    symbols.append((i + 1j*q) / np.sqrt(10))
            self.constellation = np.array(symbols)
            self.bits_per_symbol = 4
        else:
            raise ValueError(f"Unsupported modulation: {modulation}")
            
        self.M = len(self.constellation)  # Modulation order
        
    def generate_bits(self, num_symbols):
        """
        Generate random bits for all users.
        
        Args:
            num_symbols: Number of symbols per user
            
        Returns:
            bits: Array of shape (num_users, num_symbols * bits_per_symbol)
        """
        bits_per_user = num_symbols * self.bits_per_symbol
        bits = np.random.randint(0, 2, (self.num_users, bits_per_user))
        return bits
    
    def modulate(self, bits):
        """
        Map bits to constellation symbols.
        
        Args:
            bits: Array of shape (num_users, total_bits)
            
        Returns:
            symbols: Array of shape (num_users, num_symbols)
        """
        num_symbols = bits.shape[1] // self.bits_per_symbol
        symbols = np.zeros((self.num_users, num_symbols), dtype=complex)
        
        for user in range(self.num_users):
            for i in range(num_symbols):
                # Get bits for this symbol
                start_idx = i * self.bits_per_symbol
                end_idx = (i + 1) * self.bits_per_symbol
                symbol_bits = bits[user, start_idx:end_idx]
                
                # Convert bits to index (MSB first)
                idx = 0
                for b in symbol_bits:
                    idx = (idx << 1) | b
                    
                symbols[user, i] = self.constellation[idx]
        
        return symbols
    
    def apply_noma_power(self, symbols, power_ratio=0.8):
        """
        Apply NOMA power domain multiplexing.
        
        User 1 (near user) gets more power, User 2 (far user) gets less power.
        Power allocation follows: α² + β² = 1
        
        Args:
            symbols: Array of shape (num_users, num_symbols)
            power_ratio: Power allocated to near user (α²)
            
        Returns:
            scaled_symbols: Power-scaled symbols
            alpha: Near user power coefficient
            beta: Far user power coefficient
        """
        # Power allocation coefficients
        alpha = np.sqrt(power_ratio)   # Near user (more power)
        beta = np.sqrt(1 - power_ratio) # Far user (less power)
        
        # Scale symbols
        scaled_symbols = symbols.copy()
        scaled_symbols[0] *= alpha  # Near user
        scaled_symbols[1] *= beta   # Far user
        
        return scaled_symbols, alpha, beta
    
    def add_channel_effects(self, symbols):
        """
        Add realistic channel effects:
        - Independent Rayleigh fading for each user
        - AWGN noise based on SNR
        
        Args:
            symbols: Scaled symbols of shape (num_users, num_symbols)
            
        Returns:
            received: Received signal after channel and noise
            h: Channel coefficients for each user
        """
        num_symbols = symbols.shape[1]
        
        # Channel coefficients (Rayleigh fading)
        # Complex Gaussian with unit variance
        h = (np.random.randn(self.num_users, num_symbols) + 
             1j * np.random.randn(self.num_users, num_symbols)) / np.sqrt(2)
        
        # Apply channel (received = h1*s1 + h2*s2 + noise)
        received = np.sum(h * symbols, axis=0)
        
        # Add AWGN noise
        signal_power = np.mean(np.abs(received)**2)
        noise_power = signal_power / (10**(self.snr_db/10))
        noise = (np.random.randn(num_symbols) + 
                 1j * np.random.randn(num_symbols)) * np.sqrt(noise_power/2)
        
        received += noise
        
        return received, h
    
    def generate_batch(self, batch_size=1000, symbols_per_frame=100):
        """
        Generate a batch of training data.
        
        Each sample contains:
        - Features: [received_real, received_imag, |h1|, |h2|]
        - Targets: Bits for both users
        
        Args:
            batch_size: Number of frames to generate
            symbols_per_frame: Number of symbols per frame
            
        Returns:
            X: Features array of shape (batch_size, symbols_per_frame, 4)
            y1: User 1 bits of shape (batch_size, symbols_per_frame * bits_per_symbol)
            y2: User 2 bits of shape (batch_size, symbols_per_frame * bits_per_symbol)
        """
        X = []   # Features
        y1 = []  # User 1 bits
        y2 = []  # User 2 bits
        
        for _ in range(batch_size):
            # Generate bits for both users
            bits = self.generate_bits(symbols_per_frame)
            
            # Modulate
            symbols = self.modulate(bits)
            
            # Apply NOMA power allocation
            symbols, alpha, beta = self.apply_noma_power(symbols)
            
            # Add channel effects
            received, h = self.add_channel_effects(symbols)
            
            # Features: stack real, imag, channel magnitudes
            features = np.stack([
                np.real(received),
                np.imag(received),
                np.abs(h[0]),  # Channel magnitude for user 1
                np.abs(h[1])   # Channel magnitude for user 2
            ], axis=1)
            
            X.append(features)
            y1.append(bits[0])
            y2.append(bits[1])
        
        return np.array(X), np.array(y1), np.array(y2)
    
    def set_snr(self, snr_db):
        """Update SNR value."""
        self.snr_db = snr_db


if __name__ == "__main__":
    # Quick test
    noma = NOMASystem(num_users=2, modulation='QPSK', snr_db=15)
    X, y1, y2 = noma.generate_batch(batch_size=5, symbols_per_frame=10)
    print(f"Test successful!")
    print(f"X shape: {X.shape}")
    print(f"y1 shape: {y1.shape}")
    print(f"y2 shape: {y2.shape}")
