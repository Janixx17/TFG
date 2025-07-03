import torch
import torch.nn as nn
import numpy as np
from gymnasium.spaces import Box
from torch.distributions.normal import Normal

class MLPActor(nn.Module):
    """
    Simple MLP Actor network for continuous action spaces
    """
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        
        # Build the network layers to match the saved model structure
        sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        layers = []
        for i in range(len(sizes)-1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            if i < len(sizes)-2:  # Don't add activation after last layer
                layers.append(activation())
        
        self.mu_net = nn.Sequential(*layers)  # Use mu_net to match saved model
        self.log_std = nn.Parameter(-0.5 * torch.ones(act_dim, dtype=torch.float32))

    def _distribution(self, obs):
        mu = self.mu_net(obs)  # Use mu_net instead of pi
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a

class MLPCritic(nn.Module):
    """
    Simple MLP Critic network for value function approximation
    """
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        
        # Build the network layers
        sizes = [obs_dim] + list(hidden_sizes) + [1]
        layers = []
        for i in range(len(sizes)-1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            if i < len(sizes)-2:  # Don't add activation after last layer
                layers.append(activation())
        
        self.v_net = nn.Sequential(*layers)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.

class MLPActorCritic(nn.Module):
    """
    Combined Actor-Critic network
    """
    def __init__(self, observation_space, action_space, 
                 hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        else:
            raise NotImplementedError("Only continuous action spaces supported")

        # build value function
        self.v  = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]

# ======== PARÁMETROS MANUALES PARA TEST ========
obs_dim = 1009  # Correct observation dimension from the saved model
act_dim = 84    # Correct action dimension from the saved model

# Simulamos una observación de prueba
fake_obs = np.random.randn(obs_dim).astype(np.float32)
obs_tensor = torch.tensor(fake_obs).unsqueeze(0)  # Añadir batch dim

# Crear instancia del modelo (como en el entrenamiento)
observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
action_space = Box(low=-1, high=1, shape=(act_dim,), dtype=np.float32)

model = MLPActorCritic(
    observation_space=observation_space,
    action_space=action_space,
    hidden_sizes=(512, 512),
    activation=torch.nn.ReLU
)

try:
    # Cargar pesos entrenados
    model.load_state_dict(torch.load("agent_cppo_deepseek_100_epochs_20k_steps_01.pth", map_location="cpu"))
    model.eval()
    print("Model loaded successfully!")

    # Obtener acción
    with torch.no_grad():
        action = model.act(obs_tensor)

    print("Observación de prueba:")
    print(fake_obs)
    print("\nAcción predicha por el modelo:")
    print(action)
    
except FileNotFoundError:
    print("Model file 'agent_cppo_deepseek_100_epochs_20k_steps_01.pth' not found.")
    print("Please make sure the file exists in the current directory.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Testing with random model weights instead...")
    
    # Test with random weights
    with torch.no_grad():
        action = model.act(obs_tensor)

    print("Observación de prueba:")
    print(fake_obs)
    print("\nAcción predicha por el modelo (pesos aleatorios):")
    print(action)
