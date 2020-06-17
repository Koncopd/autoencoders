import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, n_vars, n_latent, **kwargs):
        super().__init__()

        self.n_vars = n_vars
        self.n_latent = n_latent
        
        self.dropout_rate = kwargs.get('dropout_rate', 0.2)

        self.mid_layers_size = kwargs.get('mid_layers_size', 600)
        self.after_mid_size = int(self.mid_layers_size/2)

        self.encoder = nn.Sequential(
            nn.Linear(self.n_vars, self.mid_layers_size),
            nn.BatchNorm1d(self.mid_layers_size),
            nn.ELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.mid_layers_size, self.after_mid_size),
            nn.BatchNorm1d(self.after_mid_size),
            nn.ELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.after_mid_size, self.n_latent)
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.n_latent, self.after_mid_size),
            nn.BatchNorm1d(self.after_mid_size),
            nn.ELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.after_mid_size, self.mid_layers_size),
            nn.BatchNorm1d(self.mid_layers_size),
            nn.ELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.mid_layers_size, self.n_vars)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return encoded, decoded
