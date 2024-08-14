# Client Security Alone Fails in Federated Learning: 2D and 3D Attack Insights

## Abstract
Federated learning (FL) plays a vital role in boosting both accuracy and privacy in the collaborative medical imaging field. The importance of privacy increases with the diverse security standards across nations and corporations, particularly in healthcare and global FL initiatives. Current research on privacy attacks in federated medical imaging focuses on sophisticated gradient inversion attacks that can reconstruct images from FL communications. These methods demonstrate potential worst-case scenarios, highlighting the need for effective security measures and the adoption of comprehensive zero-trust security frameworks. Our paper introduces a novel method for performing precise reconstruction attacks on the private data of participating clients in FL settings using a malicious server. We conducted experiments on brain tumor MRI and chest CT data sets, implementing  existing 2D and novel 3D reconstruction techniques. Our results reveal significant privacy breaches: 35.19\% of data reconstructed with 6 clients, 37.21\% with 12 clients in 2D, and 62.92\% in 3D with 12 clients. This underscores the urgent need for enhanced privacy protections in FL systems. To address these issues, we suggests effective measures to counteract such vulnerabilities by securing gradient, analytic, and linear layers. Our contributions aim to strengthen the security framework of FL in medical imaging, promoting the safe advancement of collaborative healthcare research. The source code is available at: 

## Installation

To reproduce the experiments and run the code associated with this paper, you'll need to set up the environment using Conda. The environment can be created using the provided `environment.yml` file.

### Steps to create the environment:
1. Clone this repository:
    ```bash
    git clone <repo url>
    cd [Repository Directory]
    ```
2. Create the environment:
    ```bash
    conda env create -f environment.yml
    ```
3. Activate the environment:
    ```bash
    conda activate [Environment Name]
    ```
## Example Notebooks

We provide example Jupyter notebooks to demonstrate the attack of our methods to both 2D and 3D medical image data. These can be found in the `example_notebooks` folder.

- **2D Medical Image Attack:** This notebook walks through the process of attacking 2D medical images using the techniques discussed in our paper. You can find it at `example_notebooks/2D_Attack.ipynb`.
- **3D Medical Image Attack:** This notebook provides an example of attacking 3D medical images, showcasing the methods and results for 3D data. The notebook is located at `example_notebooks/3D_Attack.ipynb`.


## License
This project is licensed under the [Apache Software License 2.0](https://www.apache.org/licenses/LICENSE-2.0). Please see the license file for more details.

## Acknowledgments
This work was built upon the  framework provided by (https://github.com/JonasGeiping/breaching). We would like to extend our thanks to the authors of this framework and the following papers for their significant contributions, which greatly inspired our work:

1. [Robbing the Fed: Directly Obtaining Private Data in Federated Learning with Modified Models ](https://arxiv.org/abs/2110.13057)
2. [When the Curious Abandon Honesty: Federated Learning Is Not Private](https://ieeexplore.ieee.org/abstract/document/10190537)



---

If you encounter any problems, please file an issue along with a detailed description.
