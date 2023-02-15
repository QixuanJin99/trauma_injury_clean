import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

## ARCHITECTURES
class Encoder(tf.keras.Model): 
    def __init__(self, input_shape, latent_dim, layer_sizes): 
        super(Encoder, self).__init__()
        self.latent_dim = int(latent_dim)
        
        self.encode_nn = tf.keras.Sequential()
        
        self.encode_nn.add(tf.keras.layers.Dense(layer_sizes[0], input_shape = input_shape, activation=tf.nn.relu, dtype=tf.float32))
        self.encode_nn.add(tf.keras.layers.Dense(layer_sizes[1], activation = tf.nn.relu, dtype=tf.float32))
        
        self.mean_estimator = tf.keras.layers.Dense(self.latent_dim, dtype=tf.float32)
        self.covar_estimator = tf.keras.layers.Dense(self.latent_dim, dtype=tf.float32)
        
    def __call__(self, x): 
        x_encode = self.encode_nn(x)
        z_mean = self.mean_estimator(x_encode)
        z_covar = tf.nn.softplus(self.covar_estimator(x_encode))
        
        return z_mean, z_covar, tfd.MultivariateNormalDiag(loc=z_mean, scale_diag=z_covar)

class Decoder(tf.keras.Model): 
    def __init__(self, output_shape, latent_dim, layer_sizes): 
        super(Decoder, self).__init__()
        self.latent_dim = int(latent_dim)
        
        self.decode_nn = tf.keras.Sequential()
        self.decode_nn.add(tf.keras.layers.Dense(layer_sizes[1], activation = tf.nn.relu, dtype=tf.float32))
        self.decode_nn.add(tf.keras.layers.Dense(layer_sizes[0], activation = tf.nn.relu, dtype=tf.float32))
        self.decode_nn.add(tf.keras.layers.Dense(output_shape, activation = tf.nn.sigmoid, dtype=tf.float32))
    
    def __call__(self, z): 
        return self.decode_nn(z)

class Classifier(tf.keras.Model): 
    def __init__(self, layer_size, num_classes, classifier_activation): 
        super(Classifier, self).__init__()
        
        self.nn = tf.keras.Sequential()
        self.nn.add(tf.keras.layers.Dense(layer_size, activation = tf.nn.relu, dtype=tf.float32))
        if classifier_activation == "softmax":
            self.nn.add(tf.keras.layers.Dense(num_classes, activation = tf.nn.softmax, dtype=tf.float32))
        elif classifier_activation == "sigmoid":
            self.nn.add(tf.keras.layers.Dense(num_classes, activation = tf.nn.sigmoid, dtype=tf.float32))
        elif classifier_activation == "dense": 
            self.nn.add(tf.keras.layers.Dense(num_classes, dtype=tf.float32))
    
    def __call__(self, z): 
        return self.nn(z)

## MODELS 
class BetaVAE(tf.keras.Model): 
    def __init__(self, encoder, decoder, beta): 
        super(BetaVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder 
        self.beta = beta 
        self.latent_dim = encoder.latent_dim 
        
        self.total_loss_tracker = tf.keras.metrics.Mean(name = 'total_loss')
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name = 'reconstruction_loss')
        self.kl_loss_tracker = tf.keras.metrics.Mean(name = 'kl_loss')
    
    def __call__(self, x):
        _, _, pz = self.encoder(x)
        z = pz.sample()
        x_recon = self.decoder(z)
        return x_recon
    
    @property 
    def metrics(self): 
        return [self.total_loss_tracker, 
               self.reconstruction_loss_tracker, 
               self.kl_loss_tracker]
    
    @tf.function
    def train_step(self, x): 
        with tf.GradientTape() as tape: 
            z_mean, z_covar , pz = self.encoder(x)
            z = pz.sample()
            x_recon = self.decoder(z)
            
            kl_loss = -0.5 * (1 + z_covar - tf.square(z_mean) - tf.exp(z_covar))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            
            reconstruct_loss = tf.reduce_mean(tf.losses.binary_crossentropy(x, x_recon))

            total_loss = reconstruct_loss + self.beta * kl_loss
            
            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            
            self.total_loss_tracker.update_state(total_loss)
            self.reconstruction_loss_tracker.update_state(reconstruct_loss)
            self.kl_loss_tracker.update_state(kl_loss)
            
        return {
            "total loss": self.total_loss_tracker.result(), 
            "reconstruction loss": self.reconstruction_loss_tracker.result(), 
            "kl loss": self.kl_loss_tracker.result()
        }

class BetaVAE_Classifier(tf.keras.Model): 
    def __init__(self, encoder, decoder, classifier, beta, gamma, 
                 classifier_loss, classifier_activation, 
                 num_classes,
                 unweighted_classifier_loss = True,
                 demo_idx = None, gcs_idx = None, mech_idx = None, risk_idx = None,
                 reg_demo = 1., reg_gcs = 1., reg_mech = 1., reg_risk = 1.,
                 ): 
        super(BetaVAE_Classifier, self).__init__()
        self.encoder = encoder
        self.decoder = decoder 
        self.classifier = classifier
        self.classifier_loss = classifier_loss
        self.classifier_activation = classifier_activation
        self.num_classes = num_classes
        self.beta = beta 
        self.gamma = gamma
        self.latent_dim = encoder.latent_dim 
        
        self.unweighted_classifier_loss = unweighted_classifier_loss
        self.demo_idx = demo_idx
        self.reg_demo = reg_demo
        self.gcs_idx = gcs_idx
        self.reg_gcs = reg_gcs
        self.mech_idx = mech_idx
        self.reg_mech = reg_mech
        self.risk_idx = risk_idx
        self.reg_risk = reg_risk

        self.total_loss_tracker = tf.keras.metrics.Mean(name = 'total_loss')
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name = 'reconstruction_loss')
        self.kl_loss_tracker = tf.keras.metrics.Mean(name = 'kl_loss')
        self.classifier_loss_tracker = tf.keras.metrics.Mean(name = 'classifier_loss')
    
    def __call__(self, inputs):
        x, c = inputs[:, :-self.num_classes], inputs[:, -self.num_classes:]
        _, _, pz = self.encoder(x)
        z = pz.sample()
        x_recon = self.decoder(z)
        class_pred = self.classifier(z)
        return x_recon, class_pred
    
    @property 
    def metrics(self): 
        return [self.total_loss_tracker, 
               self.reconstruction_loss_tracker, 
               self.kl_loss_tracker, 
               self.classifier_loss_tracker]
    
    @tf.function
    def train_step(self, inputs): 
        with tf.GradientTape() as tape: 
            x, c = inputs[:, :-self.num_classes], inputs[:, -self.num_classes:]
            z_mean, z_covar , pz = self.encoder(x)
            z = pz.sample()
            x_recon = self.decoder(z)
            class_pred = self.classifier(z)
            
            kl_loss = -0.5 * (1 + z_covar - tf.square(z_mean) - tf.exp(z_covar))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            
            reconstruct_loss = tf.reduce_mean(tf.losses.binary_crossentropy(x, x_recon))
            
            if self.classifier_loss == "binary crossentropy":
                if self.unweighted_classifier_loss: 
                    classifier_loss = tf.reduce_mean(tf.losses.binary_crossentropy(c, class_pred))
                else: 
               
                    # Given flags, add classifier loss as appropriate 
                    classifier_loss = 0
                    if self.demo_idx: 
                        # regularization demographics: age (normalized)
                        classifier_loss += self.reg_demo * tf.reduce_mean(tf.losses.binary_crossentropy(c[:, self.demo_idx[0]:self.demo_idx[1]], class_pred[:, self.demo_idx[0]:self.demo_idx[1]]))
                    if self.gcs_idx: 
                        # Multiclass: sigmoid instead of softmax 
                        classifier_loss += self.reg_gcs * tf.reduce_mean(tf.losses.binary_crossentropy(c[:, self.gcs_idx[0]:self.gcs_idx[1]], class_pred[:, self.gcs_idx[0]:self.gcs_idx[1]]))
                    if self.mech_idx: 
                        # Multilabel 
                        classifier_loss += self.reg_mech * tf.reduce_mean(tf.losses.binary_crossentropy(c[:, self.mech_idx[0]:self.mech_idx[1]], class_pred[:, self.mech_idx[0]:self.mech_idx[1]]))
                    if self.risk_idx: 
                        # 0-1 high risk or not
                        classifier_loss += self.reg_risk * tf.reduce_mean(tf.losses.binary_crossentropy(c[:, self.risk_idx[0]:self.risk_idx[1]], class_pred[:, self.risk_idx[0]:self.risk_idx[1]]))
            elif self.classifier_loss == "mse": 
                classifier_loss = tf.reduce_mean(tf.losses.MeanSquaredError(c, class_pred))

            total_loss = reconstruct_loss + self.beta * kl_loss + self.gamma * classifier_loss
            
            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            
            self.total_loss_tracker.update_state(total_loss)
            self.reconstruction_loss_tracker.update_state(reconstruct_loss)
            self.kl_loss_tracker.update_state(kl_loss)
            self.classifier_loss_tracker.update_state(classifier_loss)
            
        return {
            "total loss": self.total_loss_tracker.result(), 
            "reconstruction loss": self.reconstruction_loss_tracker.result(), 
            "kl loss": self.kl_loss_tracker.result(),
            "classifier loss": self.classifier_loss_tracker.result(),
        }


class BetaCVAE(tf.keras.Model):
    def __init__(self, encoder, decoder, beta, num_classes): 
        super(BetaCVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder 
        self.beta = beta 
        self.latent_dim = encoder.latent_dim 
        self.num_classes = num_classes
        
        self.total_loss_tracker = tf.keras.metrics.Mean(name = 'total_loss')
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name = 'reconstruction_loss')
        self.kl_loss_tracker = tf.keras.metrics.Mean(name = 'kl_loss')
    
    def __call__(self, inputs):
        x, c = inputs[:, :-self.num_classes], inputs[:, -self.num_classes:]
        _, _, pz = self.encoder(inputs)
        z = pz.sample()
        z = tf.cast(z, tf.float32)
        c = tf.cast(c, tf.float32)
        new_z = tf.concat([z, c], axis = 1)
        x_recon = self.decoder(new_z)
        return x_recon
    
    @property 
    def metrics(self): 
        return [self.total_loss_tracker, 
               self.reconstruction_loss_tracker, 
               self.kl_loss_tracker]
    
    @tf.function
    def train_step(self, inputs): 
        with tf.GradientTape() as tape: 
            x, c = inputs[:, :-self.num_classes], inputs[:, -self.num_classes:]
            z_mean, z_covar , pz = self.encoder(inputs)
            z = pz.sample()
            z = tf.cast(z, tf.float32)
            c = tf.cast(c, tf.float32)
            new_z = tf.concat([z, c], axis = 1)
            x_recon = self.decoder(new_z)
            
            kl_loss = -0.5 * (1 + z_covar - tf.square(z_mean) - tf.exp(z_covar))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            
            reconstruct_loss = tf.reduce_mean(tf.losses.binary_crossentropy(x, x_recon))

            total_loss = reconstruct_loss + self.beta * kl_loss
            
            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            
            self.total_loss_tracker.update_state(total_loss)
            self.reconstruction_loss_tracker.update_state(reconstruct_loss)
            self.kl_loss_tracker.update_state(kl_loss)
            
        return {
            "total loss": self.total_loss_tracker.result(), 
            "reconstruction loss": self.reconstruction_loss_tracker.result(), 
            "kl loss": self.kl_loss_tracker.result()
        }

class GMVAE_Classifier(tf.keras.Model): 
    def __init__(self, encoder_y, decoder, prior_gmm, encoder_gmm, classifier, 
                 classifier_loss,  
                 mix_components, latent_dim, num_classes, batch_size, 
                 reg_cat_loss = 1., reg_gauss_loss= 1., reg_recon_loss= 1., gamma = 1., 
                 pretrain = False): 
        super(GMVAE_Classifier, self).__init__()
        self.encoder_y = encoder_y
        self.encoder_gmm = encoder_gmm 
        self.prior_gmm = prior_gmm
        self.decoder = decoder 
        
        self.classifier = classifier
        self.classifier_loss = classifier_loss
        self.gamma = gamma
        
        self.latent_dim = latent_dim 
        self.batch_size = int(batch_size)
        self.mix_components = int(mix_components)
        
        self.reg_cat_loss = reg_cat_loss
        self.reg_gauss_loss = reg_gauss_loss
        self.reg_recon_loss = reg_recon_loss
        
        self.num_classes = tf.cast(num_classes, tf.int32)
        
        # Flag for whether currently in pretraining mode or not
        self.pretrain = pretrain
        
        self.total_loss_tracker = tf.keras.metrics.Mean(name = 'total loss')
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name = 'reconstruction loss')
        self.kl_div_z_loss_tracker = tf.keras.metrics.Mean(name = 'kl gaussian loss')
        self.nent_loss_tracker = tf.keras.metrics.Mean(name = 'kl categorical loss')
        self.classifier_loss_tracker = tf.keras.metrics.Mean(name = 'kl categorical loss')
    
    def __call__(self, inputs):
        x, c = inputs[:, :-self.num_classes], inputs[:, -self.num_classes:]
        
        q_y = self.encoder_y(x)
        y = q_y.sample()
        if self.pretrain:
            y = tf.fill(tf.shape(y), 0.5)

        _, _, q_z = self.encoder_gmm(tf.concat([x, y], axis = 1))
        z = q_z.sample()
        
        recon = self.decoder(z)
        return recon
    
    @property 
    def metrics(self): 
        return [self.total_loss_tracker, 
               self.reconstruction_loss_tracker, 
               self.kl_div_z_loss_tracker, 
               self.nent_loss_tracker,
               self.classifier_loss_tracker]
    
    def encode(self, x):
        """Transform 'inputs' to yield mean latent code."""
        q_y = self.encoder_y(x)
        y = q_y.sample()
        if self.pretrain:
            y = tf.fill(tf.shape(y), 0.5)

        _, _, q_z = self.encoder_gmm(tf.concat([x, y], axis = 1))
        z = q_z.sample()
        return z
    
    def generate_samples_prior(self, num_samples, clusters=None):
        """Samples components from the static latent GMM prior.
        Args:
            num_samples: Number of samples to draw from the static GMM prior.
            clusters: If desired, can sample from a specific batch of clusters.
        
        Returns:
            z: A float Tensor of shape [num_samples, mix_components, latent_size]
                representing samples drawn from each component of the GMM if 
                clusters is None. Else the Tensor is of shape [num_samples, batch_size,
                latent_size] where batch_size is the first dimension of
                clusters, depending on how many were supplied.
        """

        # If no specific clusters supplied, sample from each component in GMM
        if clusters ==  None:
            # Generate outputs over each component in GMM
            k = tf.range(0, self.mix_components)
            y = tf.one_hot(k, self.mix_components)

            _, _, p_z_given_y = self.prior_gmm(y)

            # Draw 'num_samples' samples from each cluster
            # Return shape: [num_samples, mix_components, latent_size]
            z = p_z_given_y.sample(num_samples)
            z = tf.reshape(z, [num_samples*self.mix_components, -1])
        else:
            y = tf.one_hot(clusters, self.mix_components)
            _, _, p_z_given_y = self.prior_gmm(y)

            # Draw samples from the supplied clusters
            # Shape: [num_samples, batch_size, latent_size]
            z = p_z_given_y.sample(num_samples)
            z = tf.reshape(z, [num_samples*tf.shape(clusters)[0], -1])

        return z
    
    @tf.function
    def train_step(self, inputs): 
        with tf.GradientTape() as tape: 
            x, c = inputs[:, :-self.num_classes], inputs[:, -self.num_classes:]    
            
            q_y = self.encoder_y(x)
            y = q_y.sample()
            if self.pretrain:
                y = tf.fill(tf.shape(y), 0.5)
                
            # Prior p(z | y)
            _, _, p_z_given_y = self.prior_gmm(y)
            
            # Encoder for q(z | x, y)
            _, _, q_z = self.encoder_gmm(tf.concat([x, y], axis = 1))
            z = q_z.sample()
            
            # Decoder p(x | z) 
            x_recon = self.decoder(z)
            # Classifier 
            class_pred = self.classifier(z)
            
            # Reconstruction loss 
            reconstruct_loss = tf.reduce_mean(tf.losses.binary_crossentropy(x, x_recon))
            
            # Gaussian KL: Latent loss between approximate posterior and prior for z
            kl_div_z = tf.reduce_mean(q_z.log_prob(z) - p_z_given_y.log_prob(z))

            # Categorical KL: Conditional entropy loss
            if self.pretrain:
                nent = 0.
            else:
                nent = -tf.reduce_mean(entropy(
                    q_y.distribution.logits, tf.nn.softmax(q_y.distribution.logits)))
            
            # Classifier Loss 
            if self.classifier_loss == "binary crossentropy":
                classifier_loss = tf.reduce_mean(tf.losses.binary_crossentropy(c, class_pred))
            elif self.classifier_loss == "mse": 
                classifier_loss = tf.reduce_mean(tf.losses.MeanSquaredError(c, class_pred))
            else: 
                raise Exception("Invalid classifier loss")
            
            if self.pretrain: 
                total_loss = self.reg_recon_loss * reconstruct_loss + \
                             self.reg_gauss_loss * kl_div_z + \
                             self.gamma * classifier_loss
#                 total_loss = self.reg_recon_loss * reconstruct_loss + \
#                              self.gamma * classifier_loss
            else: 
                total_loss = self.reg_recon_loss * reconstruct_loss + \
                             self.reg_gauss_loss * kl_div_z + \
                             self.reg_cat_loss * nent + \
                             self.gamma * classifier_loss
            
            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            
            self.total_loss_tracker.update_state(total_loss)
            self.reconstruction_loss_tracker.update_state(reconstruct_loss)
            self.kl_div_z_loss_tracker.update_state(kl_div_z)
            self.nent_loss_tracker.update_state(nent)
            self.classifier_loss_tracker.update_state(classifier_loss)
            
        return {
            "total loss": self.total_loss_tracker.result(), 
            "reconstruction loss": self.reconstruction_loss_tracker.result(), 
            "kl gaussian loss": self.kl_div_z_loss_tracker.result(), 
            "kl categorical loss": self.nent_loss_tracker.result(), 
            "classifier loss": self.classifier_loss_tracker.result(),
        }