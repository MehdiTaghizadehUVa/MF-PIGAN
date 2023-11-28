# Multi-fidelity Physics-informed Generative Adversarial Networks (MF-PIGANs)

## Overview
This repository hosts the code for the MF-PIGANs framework developed in our paper "Multi-fidelity Physics-informed Generative Adversarial Network for Solving Partial Differential Equations." The paper presents a novel method that integrates physics-informed neural networks (PINNs) with generative adversarial networks (GANs) for solving partial differential equations (PDEs).

## Key Features
- **Multi-Fidelity Modeling:** Harnesses low- and high-fidelity data to optimize learning and accuracy\.
- **Conditional GANs:** Uses cGANs architecture to generate new data samples conditioned on additional inputs, enhancing versatility and application scope.
- **Physics-Informed Structure:** Incorporates physical constraints in the training process of GANs, ensuring physically plausible outputs.
- **Innovative Architecture:** Our MF-PIGAN framework uniquely guides the learning of both the generator and discriminator models with physics, enhancing stability and realism.

## Repository Structure
- `src/`: Source code of the MF-PIGAN framework.
- `examples/`: Example implementations demonstrating the use of MF-PIGANs in solving different PDEs, including Burgers' Equation and Wave Equation.
- `data/`: Sample datasets used for training and testing the models.
- `docs/`: Documentation and additional resources.
- `LICENSE`: Licensing information.


## Citation
If you use this code or adapt the MF-PIGAN framework in your work, please cite our paper. The paper can be accessed [here](https://www.researchgate.net/publication/371340182_IMPROVING_ACCURACY_AND_COMPUTATIONAL_EFFICIENCY_OF_OPTIMAL_DESIGN_OF_EXPERIMENTS_VIA_GREEDY_BACKWARD_APPROACH):

Taghizadeh, M., Nabian, M. A., & Alemazkoor, N. (2023). Multi-fidelity Physics-informed Generative Adversarial Network for Solving Partial Differential Equations. Journal of Computing and Information Science in Engineering. DOI: 10.1115/1.4063986

## License
This project is licensed under the MIT License - see the `LICENSE` file for details.
