import numpy as np
import tensorflow as tf

# Define DNA alphabet
DNA_ALPHABET = 'ACGT'

# Function to generate synthetic DNA sequences
def generate_synthetic_sequences(num_sequences, seq_length):
    return [''.join(np.random.choice(list(DNA_ALPHABET), seq_length)) for _ in range(num_sequences)]

# Function to encode DNA sequences into numerical representations
def encode_sequence(sequence):
    return [DNA_ALPHABET.index(char) for char in sequence]

# Function to decode numerical representations back to DNA sequences
def decode_sequence(encoded_sequence):
    return ''.join([DNA_ALPHABET[i] for i in encoded_sequence])

# Generator model
def build_generator(latent_dim, seq_length):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, input_dim=latent_dim, activation='relu'),
        tf.keras.layers.Reshape((8, 8)),
        tf.keras.layers.Conv1D(32, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.Conv1D(4, kernel_size=3, padding='same', activation='sigmoid')
    ])
    return model

# Discriminator model
def build_discriminator(seq_length):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(32, kernel_size=3, input_shape=(seq_length, 4), activation='relu'),
        tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu'),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# GAN model
def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = tf.keras.Sequential([
        generator,
        discriminator
    ])
    return model

# Define hyperparameters
latent_dim = 100
seq_length = 50
num_sequences = 10000
epochs = 50
batch_size = 32

# Generate real DNA sequences
real_sequences = generate_synthetic_sequences(num_sequences, seq_length)
encoded_sequences = [encode_sequence(seq) for seq in real_sequences]

# Reshape sequences for training
encoded_sequences = np.array(encoded_sequences).reshape(-1, seq_length, 1)

# Build and compile discriminator
discriminator = build_discriminator(seq_length)
discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Build and compile generator
generator = build_generator(latent_dim, seq_length)
gan = build_gan(generator, discriminator)
gan.compile(optimizer='adam', loss='binary_crossentropy')

# Training loop
for epoch in range(epochs):
    for _ in range(num_sequences // batch_size):
        # Train discriminator on real sequences
        real_seq_batch = encoded_sequences[np.random.randint(0, len(encoded_sequences), batch_size)]
        real_labels = np.ones((batch_size, 1))
        discriminator_loss_real = discriminator.train_on_batch(real_seq_batch, real_labels)

        # Train discriminator on generated sequences
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        generated_sequences = generator.predict(noise)
        fake_labels = np.zeros((batch_size, 1))
        discriminator_loss_fake = discriminator.train_on_batch(generated_sequences, fake_labels)

        # Train generator (via GAN model)
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        gan_labels = np.ones((batch_size, 1))
        gan_loss = gan.train_on_batch(noise, gan_labels)

    print(f'Epoch: {epoch+1}, Discriminator Loss Real: {discriminator_loss_real[0]}, Discriminator Loss Fake: {discriminator_loss_fake[0]}, GAN Loss: {gan_loss}')

# Generate synthetic DNA sequences
num_synthetic_sequences = 1000
noise = np.random.normal(0, 1, (num_synthetic_sequences, latent_dim))
synthetic_sequences = generator.predict(noise)

# Decode synthetic sequences
decoded_sequences = [decode_sequence(np.argmax(seq, axis=1)) for seq in synthetic_sequences]
