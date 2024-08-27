# Generative Adversarial Network (GAN) for MNIST Dataset

This project focuses on leveraging the MNIST dataset within an unsupervised learning framework using PyTorch, with a specific emphasis on implementing a Generative Adversarial Network (GAN) architecture. GANs are a powerful tool for generating synthetic data that closely resembles real data distributions. The architecture of a GAN consists of two key components: a generator and a discriminator.

## Project Overview

### Generator Network
The generator network is responsible for creating synthetic data samples, in this case, generating images that resemble handwritten digits from the MNIST dataset. It takes random noise as input and transforms it into images that are intended to be indistinguishable from real MNIST images.

### Discriminator Network
The discriminator network is trained to differentiate between real MNIST images and the synthetic images produced by the generator.

### Adversarial Training
During training, the generator and discriminator networks engage in a competitive game. The generator aims to produce images that fool the discriminator into classifying them as real, while the discriminator aims to correctly classify images as either real or synthetic. Through this adversarial training process, both networks improve iteratively, with the generator gradually learning to produce more realistic images and the discriminator becoming more adept at distinguishing between real and fake images.

### Hyperparameters
The success of the GAN model is highly dependent on the selection and tuning of hyperparameters, including:
- **Batch Size:** Number of samples processed in each training iteration.
- **Learning Rate:** Controls the step size during gradient descent optimization.
- **Number of Epochs:** Number of times the entire dataset is passed through the model during training.
- **Input and Output Dimensions:** Size of input noise vectors and output images.
- **Noise Dimension:** Dimensionality of the random noise input to the generator.

Once the GAN model is trained, the resulting generator network is capable of producing new synthetic images that closely resemble the original MNIST dataset. These trained models can be saved for later use and evaluation.

## Set of Experiments

### Experiment 1: Initial Model
The initial generator and discriminator models were simplistic, employing sequential and linear layers. Experimentation revealed that integrating activation functions like LeakyReLU, Sigmoid, and Tanh enhanced the models' capabilities substantially. These non-linear functions introduced more expressive power, enabling the networks to capture intricate patterns in the data distribution.

#### Generator Architecture:
- Input: Random noise vector (input_size = 100).
- Layers: Four fully connected (FC) layers, with ReLU activation functions.
- Output: Image of 28x28 pixels (flattened to 784), with Tanh activation to normalize pixel values.

#### Discriminator Architecture:
- Input: Flattened image vector (input_size = 784).
- Layers: Three FC layers, with LeakyReLU activation functions.
- Output: Single scalar (probability), with Sigmoid activation to represent the discriminator's confidence.

### Experiment 2: Improved Model
In this experiment, LeakyReLU was used in both generator and discriminator, with a higher number of epochs for training. The generator and discriminator architectures were enhanced, utilizing LeakyReLU and dropout layers to prevent overfitting. The models were optimized using the Adam optimizer with Binary Cross Entropy Loss.

## Special Skills

The project relies on several key hyperparameters such as:
- **Number of Epochs**
- **Batch Size**
- **Learning Rate**
- **Noise Dimension**
- **Generator Output Dimension**
- **Discriminator Input Dimension**
- **Activation Functions:** LeakyReLU, Sigmoid, and Tanh.

### Adjustments Made:
- Reduced batch size to 100 due to resource constraints.
- Reduced the number of epochs to 200 for training efficiency.
- Adjusted the learning rate to 0.0002 to optimize model performance.
- Replaced the Tanh activation function with Sigmoid in the generator's last layer to match MNIST pixel value range.

## Visualization

Efforts were made to enhance result reproducibility by initializing the random seed and reloading saved models before each training session. This project visualizes the training progress and generated synthetic images resembling MNIST digits.
![image](https://github.com/user-attachments/assets/560a48a5-e9ff-4eb0-9f3a-f257e9dcbbe9)


## Conclusion

This project successfully implemented a Generative Adversarial Network (GAN) using PyTorch to generate lifelike images resembling handwritten digits from the MNIST dataset. Through meticulous experimentation and adjustments to the model's architecture and hyperparameters, the GAN was optimized for better performance. This project underscores the potential of GANs in generating synthetic data for various unsupervised learning tasks and highlights avenues for further exploration in artificial intelligence and machine learning techniques.
