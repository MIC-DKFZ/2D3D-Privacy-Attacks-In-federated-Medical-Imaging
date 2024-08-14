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

## License
This project is licensed under the Apache Software License 2.0 ([License URL](https://www.apache.org/licenses/LICENSE-2.0)). Please see the license file for more details.

## Acknowledgments
This work was built upon the base framework provided by [Repository Name](Repository URL). We would like to extend our thanks to the authors of the following papers for their significant contributions, which greatly inspired our work:

1. [Paper Title 1](DOI or URL)
2. [Paper Title 2](DOI or URL)



---

If you encounter any problems, please file an issue along with a detailed description.
