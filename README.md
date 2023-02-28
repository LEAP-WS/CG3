# CG3
This is the repository for the AAAI 21 paper [Contrastive and Generative Graph Convolutional Networks for Graph-based Semi-Supervised Learning].

Abstract: Graph-based Semi-Supervised Learning (SSL) aims to transfer the labels of a handful of labeled data to the remaining massive unlabeled data via a graph. As one of the most popular graph-based SSL approaches, the recently proposed Graph Convolutional Networks (GCNs) have gained remarkable progress by combining the sound expressiveness of neural networks with graph structure. Nevertheless, the existing graph-based methods do not directly address the core problem of SSL, i.e., the shortage of supervision, and thus their performances are still very limited. To accommodate this issue, a novel GCN-based SSL algorithm is presented in this paper to enrich the supervision signals by utilizing both data similarities and graph structure. Firstly, by designing a semi-supervised contrastive loss, improved node representations can be generated via maximizing the agreement between different views of the same data or the data from the same class. Therefore, the rich unlabeled data and the scarce yet valuable labeled data can jointly provide abundant supervision information for learning discriminative node representations, which helps improve the subsequent classification result. Secondly, the underlying determinative relationship between the data features and input graph topology is extracted as supplementary supervision signals for SSL via using a graph generative loss related to the input features. Intensive experimental results on a variety of real-world datasets firmly verify the effectiveness of our algorithm compared with other state-of-the-art methods.

## Requirements

- Tensorflow (1.14.0)

## Usage

You can conduct node classification experiments on benchmark datasets (e.g., CiteSeer) by running the 'main.py' file.

## Cite
Please cite our paper if you use this code in your own work:

```
@inproceedings{Wan2021Contrastive,
  title={Contrastive and Generative Graph Convolutional Networks for Graph-based Semi-Supervised Learning},
  author={Wan, Sheng and Pan, Shirui and Yang, Jian and Gong, Chen},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={11},
  pages={10049-10057},
  year={2021}
}
```
