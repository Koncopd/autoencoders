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

def index_iter(n_obs, batch_size):
    indices = torch.randperm(n_obs)
    for i in range(0, n_obs, batch_size):
        yield indices[i: min(i + batch_size, n_obs)]
        
def train_autoencoder(X, autoencoder, lr, batch_size, num_epochs, optim = torch.optim.Adam, **kwargs):
    optimizer = optim(autoencoder.parameters(), lr=lr, **kwargs)
    
    l2_loss = nn.MSELoss(reduction='sum')
    
    n_obs = X.shape[0]
    
    t_X = torch.from_numpy(X)
    
    for epoch in range(num_epochs):
        autoencoder.train()
        
        for step, selection in enumerate(index_iter(n_obs, batch_size)):
            batch = t_X[selection]
            
            encoded, decoded = autoencoder(batch)
            
            loss = l2_loss(decoded, batch)/len(selection)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        autoencoder.eval()
        
        t_encoded, t_decoded = autoencoder(t_X)
        t_loss = l2_loss(t_decoded, t_X).data.numpy()/n_obs
        
        print('Epoch:', epoch, '-- total train loss: %.4f' % t_loss)