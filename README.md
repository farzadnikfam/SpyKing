# SpyKing  
## A Privacy-Preserving Framework for Spiking Neural Networks

This is the source code of **SpyKing**, a framework to compare Spiking Neural Networks (SNNs) and Deep Neural Networks (DNNs) in privacy-preserving settings using Homomorphic Encryption (HE). The aim is to analyze accuracy, latency, and performance of encrypted neural computation in neuromorphic and conventional models.

For more details, refer to the main paper published in [*Frontiers in Neuroscience*](https://doi.org/10.3389/fnins.2025.1551143) (2025): DOI - 10.3389/fnins.2025.1551143

## Requirements and usage

The main file is `spyking.py`. To use the code properly, you need to manually install the following Python packages:

```bash
pip install torch pyfhel norse
```
Please make sure your Python version is 3.9 or higher for compatibility with Pyfhel and Norse.

⚠️ Note: The current version does not include a .yml environment file or requirements.txt. You must install the packages manually as shown above.
If you want to obtain the results directly on your phone you have to create a telegram bot and compile the `telegram_bot.txt` file before starting.

To cite this work, please use:
```
F. Nikfam, A. Marchisio, M. Martina and M. Shafique, "SpyKing: A Privacy-Preserving Framework for Spiking Neural Networks," in Frontiers in Neuroscience, vol. 19, pp. 1551143, May 2025, doi: 10.3389/fnins.2025.1551143.
```
```
@ARTICLE{10.3389/fnins.2025.1551143,  
AUTHOR={Nikfam, Farzad  and Marchisio, Alberto  and Martina, Maurizio  and Shafique, Muhammad },         
TITLE={SpyKing—Privacy-preserving framework for Spiking Neural Networks},        
JOURNAL={Frontiers in Neuroscience},        
VOLUME={Volume 19 - 2025},
YEAR={2025},
URL={https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1551143},
DOI={10.3389/fnins.2025.1551143},
ISSN={1662-453X},
ABSTRACT={Artificial intelligence (AI) models, frequently built using deep neural networks (DNNs), have become integral to many aspects of modern life. However, the vast amount of data they process is not always secure, posing potential risks to privacy and safety. Fully Homomorphic Encryption (FHE) enables computations on encrypted data while preserving its confidentiality, making it a promising approach for privacy-preserving AI. This study evaluates the performance of FHE when applied to DNNs and compares it with Spiking Neural Networks (SNNs), which more closely resemble biological neurons and, under certain conditions, may achieve superior results. Using the SpyKing framework, we analyze key challenges in encrypted neural computations, particularly the limitations of FHE in handling non-linear operations. To ensure a comprehensive evaluation, we conducted experiments on the MNIST, FashionMNIST, and CIFAR10 datasets while systematically varying encryption parameters to optimize SNN performance. Our results show that FHE significantly increases computational costs but remains viable in terms of accuracy and data security. Furthermore, SNNs achieved up to 35% higher absolute accuracy than DNNs on encrypted data with low values of the plaintext modulus t. These findings highlight the potential of SNNs in privacy-preserving AI and underscore the growing need for secure yet efficient neural computing solutions.}}
```
