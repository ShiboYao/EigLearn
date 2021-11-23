# Perturbing Eigenvalues with Residual Learning in Graph Convolutional Neural Networks #

**Link to paper:[download](https://github.com/ShiboYao/shiboyao.github.io/blob/master/yao21.pdf)**

### Usage ###

```
cd EigLearnGCN

python train.py
```

## Citation ##

```
@InProceedings{eiglearn,
  title = {Perturbing Eigenvalues with Residual Learning in Graph Convolutional Neural Networks},
  author = {Yao, Shibo and Yu, Dantong and Jiao, Xiangmin},
  booktitle = 	 {Proceedings of The 13th Asian Conference on Machine Learning},
  year = 	 {2021},
  series = 	 {Proceedings of Machine Learning Research},
  publisher =    {PMLR},
  abstract = 	 {Network structured data is ubiquitous in natural and social science applications. Graph Convolutional Neural Network (GCN) has attracted significant attention recently due to its success in representing, modeling, and predicting large-scale network data. Various types of graph convolutional filters were proposed to process graph signals to boost the performance of graph-based semi-supervised learning. This paper introduces a novel spectral learning technique called EigLearn, which uses residual learning to perturb the eigenvalues of the graph filter matrix to optimize its capability. EigLearn is relatively easy to implement, and yet thorough experimental studies reveal that it is more effective and efficient than the prior works on the specific issue, such as LanczosNet and FisherGCN. EigLearn only perturbs a small number of eigenvalues and does not require a complete eigendecomposition. Our investigation shows that EigLearn reaches the maximal performance improvement by perturbing about 30 to 40 eigenvalues, and the EigLearn-based GCN has comparable efficiency as the standard GCN. Furthermore, EigLearn bears a clear explanation in the spectral domain of the graph filter and shows aggregation effects in performance improvement when coupled with different graph filters. Hence, we anticipate that EigLearn may serve as a useful neural unit in various graph-involved neural net architectures.}
}
```

Contact
======

* Shibo Yao ([espoyao@gmail.com](espoyao@gmail.com))
