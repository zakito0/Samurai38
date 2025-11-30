import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import logging
from typing import List, Dict, Tuple, Optional, Union, Any
import asyncio
import ccxt
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import torch.optim as optim
import torch.nn.functional as F

class MarketSentimentAnalyzer:
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        """Initialize the sentiment analysis model"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
    
    async def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of a given text"""
        try:
            result = self.sentiment_pipeline(text[:512])  # Limit to model's max length
            return {
                'sentiment': result[0]['label'],
                'score': float(result[0]['score'])
            }
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            return {'sentiment': 'NEUTRAL', 'score': 0.5}

class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        """
        Initialize LSTM model for time series prediction
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units in LSTM
            num_layers: Number of LSTM layers
            output_size: Number of output classes (e.g., buy/sell/hold)
        """
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, output_size)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better training"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LSTM model
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # Add sequence dimension if needed (for single time step)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # (batch_size, 1, input_size)
            
        batch_size = x.size(0)
        
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return F.softmax(out, dim=1)

class AITrader:
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, output_size: int = 3, model_path: str = None):
        """Initialize the AI trader with an LSTM model"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        # Initialize model components
        self.model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.scaler = None  # Will be initialized during training
        
        if model_path:
            self.load_model(model_path)
    
    def train(self, X_train, y_train, num_epochs=100, batch_size=32):
        """Train the model"""
        try:
            # Ensure inputs are numpy arrays
            if isinstance(X_train, pd.DataFrame):
                X_train = X_train.values
            if isinstance(y_train, pd.Series):
                y_train = y_train.values
                
            # Convert to PyTorch tensors
            X_train = torch.FloatTensor(X_train).to(self.device)
            y_train = torch.LongTensor(y_train).to(self.device)
            
            # Create dataset and dataloader
            dataset = TensorDataset(X_train, y_train)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Training loop
            self.model.train()
            for epoch in range(num_epochs):
                total_loss = 0
                for batch_X, batch_y in dataloader:
                    # Move batch to device
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    
                    # Backward and optimize
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                
                if (epoch + 1) % 10 == 0:
                    logging.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}')
                    
        except Exception as e:
            logging.error(f"Error during training: {e}", exc_info=True)
            raise
    
    def predict(self, data: Union[pd.DataFrame, np.ndarray, torch.Tensor]) -> Dict[str, float]:
        """Make predictions using the trained model
        
        Args:
            data: Input data of shape (batch_size, seq_len, n_features) or (seq_len, n_features)
            
        Returns:
            Dictionary with prediction probabilities for each action
        """
        try:
            if not hasattr(self, 'model') or self.model is None:
                self.load_model()
                
            self.model.eval()
            
            # Convert input to tensor if needed
            if isinstance(data, pd.DataFrame):
                features = torch.FloatTensor(data.values)
            elif isinstance(data, np.ndarray):
                features = torch.FloatTensor(data)
            elif isinstance(data, torch.Tensor):
                features = data
            else:
                raise ValueError("Input data must be a pandas DataFrame, numpy array, or PyTorch Tensor")
            
            # Ensure we have the right shape: (batch_size, seq_len, n_features)
            if len(features.shape) == 2:
                # Add sequence dimension if missing
                features = features.unsqueeze(1)  # (batch_size, 1, n_features)
            
            # Move to device
            features = features.to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(features)
                probs = F.softmax(outputs, dim=1).squeeze().cpu().numpy()
                
                # Handle single sample case
                if len(probs.shape) == 1:
                    probs = probs.reshape(1, -1)
                
                # Ensure we have 3 output classes (buy, sell, hold)
                probs = probs.reshape(-1, 3)  # Reshape to (batch_size, 3)
                
                return {
                    'buy_prob': float(probs[0][0]),
                    'sell_prob': float(probs[0][1]),
                    'hold_prob': float(probs[0][2] if probs.shape[1] > 2 else 0.0)
                }
                
        except Exception as e:
            logging.error(f"Prediction error: {e}", exc_info=True)
            # Return neutral probabilities in case of error
            return {'buy_prob': 0.33, 'sell_prob': 0.33, 'hold_prob': 0.34}
    
    def save_model(self, path: str):
        """Save the model to disk"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load_model(self, path: str):
        """Load a model from disk"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model.eval()

class MarketDataset(Dataset):
    """PyTorch Dataset for market data"""
    def __init__(self, data: Union[pd.DataFrame, np.ndarray, torch.Tensor]):
        if isinstance(data, pd.DataFrame):
            self.data = torch.FloatTensor(data.values)
        elif isinstance(data, np.ndarray):
            self.data = torch.FloatTensor(data)
        elif isinstance(data, torch.Tensor):
            self.data = data.float()
        else:
            raise ValueError("Data must be a pandas DataFrame, numpy array, or PyTorch Tensor")
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]