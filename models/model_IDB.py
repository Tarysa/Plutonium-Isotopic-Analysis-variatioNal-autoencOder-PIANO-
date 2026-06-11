import torch
import torch.nn as nn
import numpy as np
from models.GRL import ReverseLayerF

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.
    Applies affine transformation to features based on conditioning parameters.
    """
    def __init__(self, input_neurons, output_neurons):
        super().__init__()
        
        # Linear layers for gamma (scale) and beta (shift) parameters
        self.linear_gamma = nn.Linear(input_neurons, output_neurons)
        self.linear_beta = nn.Linear(input_neurons, output_neurons)
        
        # Initialize weights using Xavier uniform distribution
        nn.init.xavier_uniform_(self.linear_gamma.weight)
        nn.init.xavier_uniform_(self.linear_beta.weight)
    
    def forward(self, x, conditioning_params):
        """
        Apply FiLM transformation: x * gamma + beta
        
        Args:
            x: Input features [batch_size, features, sequence_length]
            conditioning_params: Conditioning parameters [batch_size, input_neurons]
        
        Returns:
            Modulated features
        """
        gamma = self.linear_gamma(conditioning_params).unsqueeze(-1)  # [batch, features, 1]
        beta = self.linear_beta(conditioning_params).unsqueeze(-1)    # [batch, features, 1]
        
        return x * gamma + beta


class RelativeMSELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        rel = (y_pred - y_true) / (torch.abs(y_true) + self.eps)
        return torch.mean(rel ** 2)
    
def compute_loss(predictions, targets):
    """
    Compute loss based on data types (qualitative or quantitative).
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets    
    Returns:
        Combined loss value
    """
    
    total_loss = nn.MSELoss()(predictions, targets)
    
    return total_loss

class RelativeHuberLoss(nn.Module):
    def __init__(self, delta=1.0, eps=1e-4):
        super().__init__()
        self.delta = delta
        self.eps = eps

    def forward(self, pred, target):
        error = pred - target
        abs_err = torch.abs(error)

        # Huber part
        quadratic = torch.minimum(abs_err, torch.tensor(self.delta))
        linear = abs_err - quadratic

        huber = 0.5 * quadratic**2 + self.delta * linear

        # Relative normalization — crucial
        rel = huber / (torch.abs(target) + self.eps)

        # Mean over batch + features
        return rel.mean()


class CLRLoss(nn.Module):
    """
    Centered Log-Ratio (CLR) Loss for compositional data.
    
    This loss function is specifically designed for data constrained to the simplex
    (i.e., data where components sum to 1, like isotopic compositions).
    
    The CLR transformation maps compositional data from the simplex to Euclidean space
    where standard distance metrics are appropriate.
    
    Mathematical formulation:
        clr(x) = [log(x_1/g(x)), log(x_2/g(x)), ..., log(x_N/g(x))]
        where g(x) = (x_1 * x_2 * ... * x_N)^(1/N) is the geometric mean
    
    Args:
        eps (float): Small constant to avoid log(0). Default: 1e-10
                    For isotopic data, this should be smaller than your minimum
                    expected concentration.
        reduction (str): Specifies the reduction to apply to the output:
                        'mean' | 'sum' | 'none'. Default: 'mean'
        use_huber (bool): If True, apply Huber loss in CLR space instead of MSE.
                         This makes the loss more robust to outliers. Default: False
        huber_delta (float): Delta parameter for Huber loss. Default: 1.0
    
    Compatibility with Softmax:
        ✅ YES - This loss is FULLY COMPATIBLE with Softmax outputs.
        The CLR transformation is applied to both predictions and targets,
        so both must be valid compositions (positive, sum to 1).
        Softmax guarantees this for predictions.
    
    Usage:
        criterion = CLRLoss(eps=1e-10)
        
        # predictions from model with Softmax output
        predictions = model(x)  # shape: [batch, num_isotopes]
        targets = y             # shape: [batch, num_isotopes]
        
        loss = criterion(predictions, targets)
    
    Note on zero handling:
        If your data contains true zeros (impossible in reality for isotopes, but
        possible in datasets), you need to handle them before applying CLR.
        Options:
        1. Use multiplicative replacement (replace zeros with eps)
        2. Use additive log-ratio (ALR) with a reference isotope instead
        3. Ensure your measurement process doesn't produce exact zeros
    """
    
    def __init__(self, eps=1e-10, reduction='mean', use_huber=False, huber_delta=1.0):
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.use_huber = use_huber
        self.huber_delta = huber_delta
        
        if use_huber:
            self.base_loss = nn.HuberLoss(reduction='none', delta=huber_delta)
        else:
            self.base_loss = nn.MSELoss(reduction='none')
    
    def _clr_transform(self, x):
        log_x = torch.log(x + self.eps)
        # clr = log(x) - log(geom_mean)
        # log(geom_mean) est simplement la moyenne des log(x)
        return log_x - torch.mean(log_x, dim=-1, keepdim=True)
    
    def forward(self, pred, target):
        """
        Compute CLR loss between predictions and targets.
        
        Args:
            pred: Predicted compositions [batch, num_components]
                  Should come from Softmax (positive, sum to 1)
            target: Target compositions [batch, num_components]
                    Should be normalized (positive, sum to 1)
        
        Returns:
            Scalar loss value (if reduction='mean' or 'sum')
            or per-sample losses (if reduction='none')
        """
        # Verify inputs are valid compositions (optional, remove in production)
        if torch.any(pred < 0) or torch.any(target < 0):
            raise ValueError("CLR Loss expects positive values (compositions)")
        

        # Apply CLR transformation to both pred and target
        clr_pred = self._clr_transform(pred)
        clr_target = self._clr_transform(target)
        
        # Compute loss in CLR space
        if self.use_huber:
            loss = self.base_loss(clr_pred, clr_target)
        else:
            loss = self.base_loss(clr_pred, clr_target)
        
        # Reduce across components (mean over isotopes)
        loss = torch.mean(loss, dim=-1)
        
        # Apply final reduction
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")

class DenseBlock(nn.Module):
    """
    Dense (fully connected) block with optional normalization and activation.
    """
    def __init__(self, config, input_neurons, output_neurons, 
                 is_last_layer=False, use_norm=True, activation_function="leakyReLU"):
        super().__init__()
        self.config = config
        
        # Build activation sequence
        activation_layers = []
        
        if not is_last_layer:
            if use_norm:
                activation_layers.append(nn.BatchNorm1d(output_neurons))
                
                # Add specified activation function
                if activation_function == "leakyReLU":
                    activation_layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
                elif activation_function == "ReLU":
                    activation_layers.append(nn.ReLU())
                elif activation_function == "Mish":
                    activation_layers.append(nn.Mish())
                elif activation_function == "SiLU":
                    activation_layers.append(nn.SiLU())
        
        self.linear = nn.Linear(input_neurons, output_neurons)
        nn.init.xavier_uniform_(self.linear.weight)
        
        self.activation = nn.Sequential(*activation_layers)
    
    def forward(self, x):
        x = self.linear(x)
        return self.activation(x)


class ConvBlock(nn.Module):
    """
    1D Convolutional block with optional normalization, activation, and dropout.
    """
    def __init__(self, config, input_channels, output_channels, kernel_size=4, 
                 use_activation=True, use_stride=True, use_norm=True, use_bias=True):
        super().__init__()
        self.config = config
        
        # Dropout configuration
        self.use_dropout = getattr(config, 'dropout', False)
        self.dropout_rate = getattr(config, 'drop_rate', 0.2)
        
        activation_layers = []
        
        if use_norm:
            # Choose normalization type
            if config.type == "batchnorm":
                activation_layers.append(nn.BatchNorm1d(output_channels))
            elif config.type == "instancenorm":
                activation_layers.append(nn.InstanceNorm1d(output_channels, affine=True))
            
            if use_activation:
                activation_layers.append(nn.ReLU(inplace=True))
        
        # Configure convolution layer
        if use_stride:
            self.conv = nn.Conv1d(input_channels, output_channels, 
                                kernel_size=kernel_size, stride=2, padding=1, bias=use_bias)
        else:
            self.conv = nn.Conv1d(input_channels, output_channels, 
                                kernel_size=kernel_size, stride=1, padding="same", bias=use_bias)
        
        if self.use_dropout:
            activation_layers.append(nn.Dropout(p=self.dropout_rate))
        
        nn.init.xavier_uniform_(self.conv.weight)
        self.activation = nn.Sequential(*activation_layers)
    
    def forward(self, x, condition=None):
        x = self.conv(x)
        return self.activation(x)


class ResidualConvBlock(nn.Module):
    """
    Residual convolutional block with optional FiLM conditioning.
    """
    def __init__(self, config, input_channels, output_channels, kernel_size, downsample):
        super().__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.downsample = downsample
        self.use_film = getattr(config, 'film_layer', False)
        
        # Normalization layers
        norm_cls = (nn.BatchNorm1d if config.type == "batchnorm" 
                   else lambda c: nn.InstanceNorm1d(c, affine=True))
        self.norm1 = norm_cls(input_channels)
        self.norm2 = norm_cls(output_channels)
        
        self.activation = nn.ReLU()
        
        nb_condition = 3
        # FiLM layers for conditional generation
        if self.use_film:
            self.film_layer_1 = FiLMLayer(nb_condition, output_channels)
            self.film_layer_2 = FiLMLayer(nb_condition, output_channels)
        
        # Residual connection handling
        if downsample:
            self.downsample_conv1 = ConvBlock(config, input_channels, output_channels, 
                                            kernel_size=1, use_stride=False, use_bias=False, 
                                            use_activation=False, use_norm=False)
            self.downsample_conv2 = ConvBlock(config, output_channels, output_channels, 
                                            kernel_size=kernel_size, use_stride=downsample, 
                                            use_bias=False, use_activation=False, use_norm=False)
        elif input_channels != output_channels:
            self.channel_match_conv = ConvBlock(config, input_channels, output_channels, 
                                              kernel_size=kernel_size, use_stride=False, 
                                              use_bias=False, use_activation=False, use_norm=False)
        else:
            self.channel_match_conv = None
        
        # Main convolution layers
        self.conv1 = ConvBlock(config, input_channels, output_channels, 
                             kernel_size=kernel_size, use_stride=downsample, 
                             use_activation=False, use_norm=False)
        self.conv2 = ConvBlock(config, output_channels, output_channels, 
                             kernel_size=kernel_size, use_stride=False, 
                             use_activation=False, use_norm=False)
    
    def forward(self, x, condition=None):
        # Store input for residual connection
        residual = x.clone()
        
        # First conv block
        if self.use_film and condition is not None:
            x = self.film_layer_1(x, condition)
        else:
            x = self.norm1(x)
        x = self.activation(x)
        x = self.conv1(x)
        
        # Second conv block
        if self.use_film and condition is not None:
            x = self.film_layer_2(x, condition)
        else:
            x = self.norm2(x)
        x = self.activation(x)
        x = self.conv2(x)
        
        # Handle residual connection
        if self.downsample:
            residual = self.downsample_conv1(residual)
            residual = self.downsample_conv2(residual)
        elif self.channel_match_conv:
            residual = self.channel_match_conv(residual)
        
        return x + residual


class ConvolutionalEncoder(nn.Module):
    """
    Stack of convolutional blocks for encoding.
    """
    def __init__(self, config, num_conv_layers):
        super().__init__()
        self.config = config
        
        # Build encoder blocks
        conv_blocks = []
        
        # First layer
        conv_blocks.append(ConvBlock(config, 1, 4))
        
        # Intermediate layers with exponentially increasing channels
        for i in range(1, num_conv_layers):
            in_channels = min(64, 2**(i+1))
            out_channels = min(64, 2**(i+2))
            conv_blocks.append(ConvBlock(config, in_channels, out_channels))
        
        # Optional residual blocks
        if config.resnet:
            final_channels = min(64, 2**(num_conv_layers+1))
            for _ in range(2):
                conv_blocks.append(
                    ResidualConvBlock(config, final_channels, final_channels, 
                                    kernel_size=4, downsample=False)
                )
        
        self.conv_blocks = nn.Sequential(*conv_blocks)
    
    def forward(self, x, condition=None):
        for layer in self.conv_blocks:
            x = layer(x, condition)
        return x


class DeconvolutionalBlock(nn.Module):
    """
    Transposed convolution block for decoding.
    """
    def __init__(self, config, input_channels, output_channels, 
                 use_stride=True, is_last_layer=False):
        super().__init__()
        self.config = config
        
        # Dropout configuration
        self.use_dropout = getattr(config, 'dropout', False)
        self.dropout_rate = getattr(config, 'drop_rate', 0.2)
        
        activation_layers = []
        
        # Configure transposed convolution
        if use_stride:
            self.conv = nn.ConvTranspose1d(input_channels, output_channels, 
                                         kernel_size=4, stride=2, padding=1)
        else:
            self.conv = nn.Conv1d(input_channels, output_channels, 
                                kernel_size=4, stride=1, padding="same")
        
        # Configure activation for last layer
        if is_last_layer:
            if config.last_act == "identity":
                activation_layers.append(nn.Identity())
            elif config.last_act == "sigmoid":
                activation_layers.append(nn.Sigmoid())
        else:
            # Standard intermediate layer configuration
            if config.type == "batchnorm":
                activation_layers.append(nn.BatchNorm1d(output_channels))
            elif config.type == "instancenorm":
                activation_layers.append(nn.InstanceNorm1d(output_channels, affine=True))
            
            activation_layers.append(nn.LeakyReLU(negative_slope=0.2))
            
            if self.use_dropout:
                activation_layers.append(nn.Dropout(p=self.dropout_rate))
        
        nn.init.xavier_uniform_(self.conv.weight)
        self.activation = nn.Sequential(*activation_layers)
    
    def forward(self, x, condition=None):
        x = self.conv(x)
        return self.activation(x)


class ConvolutionalDecoder(nn.Module):
    """
    Stack of deconvolutional blocks for decoding.
    """
    def __init__(self, config, num_conv_layers):
        super().__init__()
        self.config = config
        
        deconv_blocks = []
        
        # Optional residual blocks at the beginning
        if config.resnet:
            final_channels = min(64, 2**(num_conv_layers+1))
            for _ in range(2):
                deconv_blocks.append(
                    ResidualConvBlock(config, final_channels, final_channels, 
                                    kernel_size=4, downsample=False)
                )
        
        # Intermediate deconvolution layers
        for i in range(num_conv_layers-1, 0, -1):
            in_channels = min(64, 2**(2+i))
            out_channels = min(64, 2**(2+i-1))
            deconv_blocks.append(DeconvolutionalBlock(config, in_channels, out_channels))
        
        # Penultimate layer
        deconv_blocks.append(DeconvolutionalBlock(config, 4, 4))
        
        self.deconv_blocks = nn.Sequential(*deconv_blocks)
        
        # Final layer to single channel output
        self.final_layer = DeconvolutionalBlock(config, 4, 1, use_stride=False, is_last_layer=True)
    
    def forward(self, x, condition=None):
        for layer in self.deconv_blocks:
            x = layer(x, condition)
        return self.final_layer(x)


class SharedEncoder(nn.Module):
    """
    Shared encoder that maps input to latent space parameters (mu, log_sigma^2).
    """
    def __init__(self, config, input_dim):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        self.latent_dim = config.latent_dim
        self.num_conv_layers = config.nb_conv_layer
        
        # Convolutional feature extractor
        self.conv_net = ConvolutionalEncoder(config, self.num_conv_layers)
        
        # Linear layer to latent parameters
        feature_dim = min(64, 2**(1+self.num_conv_layers)) * (input_dim // (2**self.num_conv_layers))
        self.linear = nn.Linear(feature_dim, 2 * self.latent_dim)  # mu and log_sigma^2
    
    def forward(self, x, condition=None):
        # Extract features using convolutions
        x = self.conv_net(x, condition)
        
        # Flatten and project to latent parameters
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        
        # Split into mean and log variance
        mu = x[:, :self.latent_dim]
        log_sigma_2 = x[:, self.latent_dim:]
        
        return mu, log_sigma_2


class SharedDecoder(nn.Module):
    """
    Shared decoder that maps from latent space back to input space.
    """
    def __init__(self, config, input_dim):
        super().__init__()
        self.config = config
        self.latent_dim = config.latent_dim
        self.num_conv_layers = config.nb_conv_layer
        self.input_dim = input_dim
        
        # Deconvolutional generator
        self.deconv_net = ConvolutionalDecoder(config, self.num_conv_layers)
        
        # Linear projection from latent space
        feature_dim = min(64, 2**(1+self.num_conv_layers)) * (input_dim // (2**self.num_conv_layers))
        
        self.linear = DenseBlock(config, self.latent_dim, feature_dim)
        
        # Shape for reshaping flattened features back to conv format
        self.reshape_dims = (-1, min(64, 2**(1+self.num_conv_layers)), 
                           input_dim // (2**self.num_conv_layers))
    
    def forward(self, x, condition=None):
        # Project to feature space
        x = self.linear(x)
        
        # Reshape for deconvolution
        x = x.view(self.reshape_dims)
        
        # Generate output using deconvolutional layers
        return self.deconv_net(x, condition)


class Regressor(nn.Module):
    """
    Regression head for predicting target values from latent representations.
    """
    def __init__(self, config, num_dense_layers, use_physical_data, 
                 input_class_dim=3, num_classes=1, use_softmax = False):
        super().__init__()
        
        self.use_physical_data = use_physical_data
        
        # Determine input dimension
        if use_physical_data:
            input_dim = config.latent_dim + input_class_dim
        else:
            input_dim = config.latent_dim
        
        # Determine output dimension based on dataset
        
        num_outputs = num_classes
        
        # Build regression layers
        layers = []
        if num_dense_layers == 1:
            layers = [DenseBlock(config, input_dim, num_outputs, is_last_layer=True)]
        elif num_dense_layers == 2:
            layers = [DenseBlock(config, input_dim, 64)]
            
            if config.dropout_cls:
                layers.append(nn.Dropout(0.2))
            
            layers.append(DenseBlock(config, 64, num_outputs, is_last_layer=True))
        elif num_dense_layers == 3:
            layers = [DenseBlock(config, input_dim, 128)]
            
            if config.dropout_cls:
                layers.append(nn.Dropout(0.2))
            
            layers.append(DenseBlock(config, 128, 64))

            if config.dropout_cls:
                layers.append(nn.Dropout(0.2))
            
            layers.append(DenseBlock(config, 64, num_outputs, is_last_layer=True))

        layers.append(nn.Softmax(dim=1))


        self.regressor = nn.Sequential(*layers)
    
    def forward(self, x):
                
        return self.regressor(x)


class VAE(nn.Module):
    """
    Conditional Variational Autoencoder with optional domain adaptation and regression.
    """
    def __init__(self, config, device, input_dim, input_class_dim=3, num_outputs_pu=5, num_outputs_u=4):
        super().__init__()
        
        self.config = config
        self.device = device
        self.input_dim = input_dim
        self.input_class_dim = input_class_dim

        self.use_softmax = True
     
        if hasattr(config, 'train_with_condition'):
            self.train_with_condition = config.train_with_condition
        else:
            self.train_with_condition = False
        
        if hasattr(config, 'use_decoder'):
            self.use_decoder = config.use_decoder
        else:
            self.use_decoder = True
        
        # Core VAE components
        self.encoder = SharedEncoder(config, input_dim)

        if self.use_decoder:
            self.decoder = SharedDecoder(config, input_dim)
        
        # Conditioning embeddings
        self.class_embedding_encoder = DenseBlock(config, input_class_dim, input_dim, use_norm=False)
        self.class_embedding_decoder = DenseBlock(config, input_class_dim, config.latent_dim, use_norm=False)
        
        # Latent regressor
        
        regressor_input_dim = config.latent_dim
        
        # layers = []
        # if config.nb_dense_layer == 1:
        #     layers = [DenseBlock(config, regressor_input_dim, num_outputs, is_last_layer=True)]
        # elif config.nb_dense_layer == 2:
        #     layers = [DenseBlock(config, regressor_input_dim, 20), 
        #                 nn.Dropout(0.2),
        #                 DenseBlock(config, 20, num_outputs, is_last_layer=True)]
        
        # self.regressor = nn.Sequential(*layers)

        self.pu_regressor = Regressor(config, num_dense_layers= config.nb_dense_layer, use_physical_data=self.train_with_condition, 
                 input_class_dim=input_class_dim, num_classes=num_outputs_pu)
        
        self.u_regressor = Regressor(config, num_dense_layers= config.nb_dense_layer, use_physical_data=self.train_with_condition, 
                 input_class_dim=input_class_dim, num_classes=num_outputs_u)
    
    def forward(self, x, u_label, pu_label, condition=None):
        """
        Forward pass through the conditional VAE.
        
        Args:
            x: Input data [batch_size, sequence_length]
            condition: Conditioning information [batch_size, condition_dim]
            type: Types of data for loss computation
        
        Returns:
            Reconstruction and latent parameters, with optional auxiliary outputs
        """
        # Encode to latent space
        mu, log_sigma_2 = self.encoder(x.unsqueeze(1))
        z = self.reparameterize(mu, log_sigma_2)

        if self.train_with_condition:
            reg_input = torch.cat((mu, condition), dim=1)
        else:
            reg_input = mu.clone()

        # Regression    
        mask_u = (torch.sum(u_label, dim=1) > 0.95) & (torch.sum(u_label, dim=1) < 1.05)
        mask_pu = (torch.sum(pu_label, dim=1) > 0.95) & (torch.sum(pu_label, dim=1) < 1.05)

        if mask_u.sum() >= 2:            
            preds_u = self.u_regressor(reg_input[mask_u])
        else:
            preds_u = torch.zeros(0, u_label.shape[1], device=self.device)

        if mask_pu.sum() >= 2:            
            preds_pu = self.pu_regressor(reg_input[mask_pu])
        else:
            preds_pu = torch.zeros(0, pu_label.shape[1], device=self.device)
       
        if self.use_decoder:

            # Decode from latent space
            reconstruction = self.decoder(z, condition)[:, 0]
            
            # Return appropriate outputs based on configuration
            return (reconstruction, mu, log_sigma_2), preds_u, mask_u, preds_pu, mask_pu
        else:
            return None, preds_u, mask_u, preds_pu, mask_pu
        
    
    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick for VAE sampling.
        
        Args:
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution
        
        Returns:
            Sampled latent code
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def _compute_regressor_loss(self, regressor_pred, truth):
        """
        Compute regression loss for isotope prediction.
        
        Args:
            z: Latent representation
        
        Returns:
            Domain classification loss
        """
                
        return compute_loss(regressor_pred, truth)
    
    def exact_reconstruction(self, x, u_label, pu_label, condition=None):
        """
        Perform reconstruction using mean of latent distribution (no sampling).
        
        Args:
            x: Input data
        
        Returns:
            Exact reconstruction and latent parameters
        """
        # Encode using mean only
        
        mu, log_sigma_2 = self.encoder(x.unsqueeze(1))
        
        # regression_pred = torch.clamp(regression_pred, 0, 1)

        mask_u =  abs(torch.sum(u_label, dim=1) - 1) < 0.1
        mask_pu =  abs(torch.sum(pu_label, dim=1) - 1) < 0.1

        if self.train_with_condition:
            reg_input = torch.cat((mu, condition), dim=1)
        else:
            reg_input = mu.clone()

        # Regression    
        if torch.sum(mask_u) > 0:
            regression_pred_u = self.u_regressor(reg_input[mask_u])
            normalized_preds_u = regression_pred_u
            
        else:
            normalized_preds_u = torch.zeros(0, self.u_regressor.regressor[-1].out_features if isinstance(self.u_regressor.regressor[-1], nn.Linear) else 4).to(self.device)
        if torch.sum(mask_pu) > 0:
            regression_pred_pu = self.pu_regressor(reg_input[mask_pu])

            normalized_preds_pu = regression_pred_pu
           
        else:
            normalized_preds_pu = torch.zeros(0, self.pu_regressor.regressor[-1].out_features if isinstance(self.pu_regressor.regressor[-1], nn.Linear) else 5).to(self.device)

       
        if self.use_decoder:
            # Decode using mean
            reconstruction = self.decoder(mu, condition = condition)[:, 0]
            
            return reconstruction, mu, log_sigma_2, mask_u, mask_pu, normalized_preds_u, normalized_preds_pu
        else:
            return mask_u, mask_pu, normalized_preds_u, normalized_preds_pu


class KLDivergenceLoss(nn.Module):
    """
    KL divergence loss for VAE training.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, mu, log_sigma_2):
        """
        Compute KL divergence between learned distribution and unit Gaussian.
        
        Args:
            mu: Mean of learned distribution
            log_sigma_2: Log variance of learned distribution
        
        Returns:
            KL divergence loss
        """
        kl_loss = 1 + log_sigma_2 - torch.exp(log_sigma_2) - torch.square(mu)
        kl_loss = -0.5 * torch.sum(kl_loss, dim=1)
        return torch.mean(kl_loss)


class VAELoss(nn.Module):
    """
    Standard VAE loss combining reconstruction and KL divergence terms.
    """
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.reconstruction_loss = nn.MSELoss(reduction="none")
        self.kl_loss = KLDivergenceLoss()
    
    def forward(self, epoch, vae_output, target):
        """
        Compute total VAE loss.
        
        Args:
            epoch: Current training epoch (unused in this version)
            vae_output: Tuple of (reconstruction, mu, log_sigma_2)
            target: Target data
        
        Returns:
            Total loss, reconstruction loss, KL loss
        """
        reconstruction, mu, log_sigma_2 = vae_output
        
        batch_size = reconstruction.size(0)
        recon_loss = self.reconstruction_loss(reconstruction, target) / batch_size
        kl_loss = self.kl_loss(mu, log_sigma_2)
        
        total_loss = recon_loss + self.beta * kl_loss

        return total_loss, recon_loss, kl_loss


def create_cyclic_schedule(num_iterations, start=0.0, stop=1.0, num_cycles=4, ratio=0.2):
    """
    Create a cyclic annealing schedule for beta parameter.
    
    Args:
        num_iterations: Total number of training iterations
        start: Starting value for schedule
        stop: Maximum value for schedule
        num_cycles: Number of annealing cycles
        ratio: Fraction of cycle spent increasing
    
    Returns:
        Array of beta values for each iteration
    """
    schedule = np.ones(num_iterations) * stop
    period = num_iterations / num_cycles
    step = (stop - start) / (period * ratio)
    
    for cycle in range(num_cycles):
        value, iteration = start, 0
        while value <= stop and (int(iteration + cycle * period) < num_iterations):
            schedule[int(iteration + cycle * period)] = value
            value += step
            iteration += 1
    
    return schedule


class VAELossWithSchedule(nn.Module):
    """
    VAE loss with cyclic annealing schedule for beta parameter.
    """
    def __init__(self, num_iterations, num_cycles=1, ratio=0.3, start=0.0, stop=1.0, 
                 loss="mse"):
        super().__init__()
        
        # Loss functions
        self.bce_loss = nn.BCELoss()
        
        if loss == "mse":
            self.reconstruction_loss = nn.MSELoss(reduction="none")
        elif loss == "mae":
            self.reconstruction_loss = nn.L1Loss(reduction="none")
        else:
            self.reconstruction_loss = nn.BCELoss(reduction="none")
        
        self.kl_loss = KLDivergenceLoss()
        
        # Create beta annealing schedule
        self.beta_schedule = create_cyclic_schedule(
            num_iterations, num_cycles=num_cycles, ratio=ratio, start=start, stop=stop
        )
            
    def forward(self, epoch, vae_output, target):
        """
        Compute VAE loss with scheduled beta parameter.
        
        Args:
            epoch: Current training epoch
            vae_output: Tuple of (reconstruction, mu, log_sigma_2)
            target: Target data
        
        Returns:
            Total loss, reconstruction loss, KL loss
        """
        reconstruction, mu, log_sigma_2 = vae_output
        
        batch_size = reconstruction.size(0)
        beta = self.beta_schedule[epoch]
        
        # Weighted reconstruction loss (emphasizes higher magnitude targets)
        reconstruction_loss = (torch.sum(torch.exp(target + 1) * 
                                       self.reconstruction_loss(reconstruction, target)) / 
                             batch_size)
        
        kl_loss = self.kl_loss(mu, log_sigma_2)
        total_loss = reconstruction_loss + beta * kl_loss
        
        return total_loss, reconstruction_loss, kl_loss