# Multi-fidelity Physics-informed Generative Adversarial Networks (MF-PIGANs)

## Overview
This repository hosts the code for the MF-PIGANs framework developed in our paper "Multi-fidelity Physics-informed Generative Adversarial Network for Solving Partial Differential Equations." The paper presents a novel method that integrates physics-informed neural networks (PINNs) with generative adversarial networks (GANs) for solving partial differential equations (PDEs)&#8203;``【oaicite:5】``&#8203;.

## Key Features
- **Multi-Fidelity Modeling:** Harnesses low- and high-fidelity data to optimize learning and accuracy&#8203;``【oaicite:4】``&#8203;.
- **Conditional GANs:** Uses cGANs architecture to generate new data samples conditioned on additional inputs, enhancing versatility and application scope&#8203;``【oaicite:3】``&#8203;.
- **Physics-Informed Structure:** Incorporates physical constraints in the training process of GANs, ensuring physically plausible outputs&#8203;``【oaicite:2】``&#8203;.
- **Innovative Architecture:** Our MF-PIGAN framework uniquely guides the learning of both the generator and discriminator models with physics, enhancing stability and realism&#8203;``【oaicite:1】``&#8203;.

## Repository Structure
- `src/`: Source code of the MF-PIGAN framework.
- `examples/`: Example implementations demonstrating the use of MF-PIGANs in solving different PDEs, including Burgers' Equation and Wave Equation&#8203;``【oaicite:0】``&#8203;.
- `data/`: Sample datasets used for training and testing the models.
- `docs/`: Documentation and additional resources.
- `LICENSE`: Licensing information.


## Citation
If you use this code or adapt the MF-PIGAN framework in your work, please cite our paper:

Taghizadeh, M., Nabian, M. A., & Alemazkoor, N. (2023). Multi-fidelity Physics-informed Generative Adversarial Network for Solving Partial Differential Equations. Journal of Computing and Information Science in Engineering. DOI: 10.1115/1.4063986

## License
This project is licensed under the MIT License - see the `LICENSE` file for details.
