# LaptSNE: Laplacian-based Cluster-Contractive t-SNE

LaptSNE is a new graph-layout nonlinear dimensionality reduction method based on t-SNE for visualizing high-dimensional data as 2D scatter plots. Specifically, LAPtSNE leverages the eigenvalue information of the graph Laplacian to shrink the potential clusters in the low-dimensional embedding when learning to preserve the local and global structure from high-dimensional space to low-dimensional space.

We evaluate our method by a formal comparison with state-of-the-art methods, both visually and via established quantitative measurements. The results demonstrate the superiority of our method over baselines such as t-SNE and UMAP. We also extend our method to spectral clustering and establish an accurate and parameter-free clustering algorithm.

## How to use

There are two implementation of LaptSNE. First, we implement the LaptSNE based on the scikit-learn package. Another extended version was tsne-torch, which support GPU-acceleration. You can test the code by runing `test-TSNE.py` for the sklearn version and running `test-torch.py` for the tsne-torch version. Data are available in the folder `./data`, where COIL20 dataset is provided.