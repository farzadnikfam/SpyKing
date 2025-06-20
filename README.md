# SpyKing  
## A Privacy-Preserving Framework for Spiking Neural Networks

This is the source code of **SpyKing**, a framework to compare Spiking Neural Networks (SNNs) and Deep Neural Networks (DNNs) in privacy-preserving settings using Homomorphic Encryption (HE). The aim is to analyze accuracy, latency, and performance of encrypted neural computation in neuromorphic and conventional models.

For more details, refer to the main paper published in *Frontiers in Neuroscience* (2025): [DOI link](https://doi.org/10.3389/fnins.2025.1551143)

## Requirements and usage

The main file is `spyking.py`. To use the code properly, you need to manually install the following Python packages:

```bash
pip install torch pyfhel norse
Please make sure your Python version is 3.9 or higher for compatibility with Pyfhel and Norse.

⚠️ Note: The current version does not include a .yml environment file or requirements.txt. You must install the packages manually as shown above.
If you want to obtain the results directly on your phone you have to create a telegram bot and compile the `telegram_bot.txt` file before starting.

To cite this work, please use:
```
F. Nikfam, A. Marchisio, M. Martina and M. Shafique, "SpyKing: A Privacy-Preserving Framework for Spiking Neural Networks," in Frontiers in Neuroscience, vol. 19, pp. 1551143, May 2025, doi: 10.3389/fnins.2025.1551143.
```
```
@article{spyking2025,
  author  = {Nikfam, Farzad and Marchisio, Andrea and Martina, Marco and Shafique, Muhammad},
  title   = {SpyKing: A Privacy-Preserving Framework for Spiking Neural Networks},
  journal = {Frontiers in Neuroscience},
  volume  = {19},
  pages   = {1551143},
  year    = {2025},
  doi     = {10.3389/fnins.2025.1551143}
}
```
