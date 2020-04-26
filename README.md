# CVPR2020-QEBA
This is the code repository accompanying the paper: QEBA: Query-Efﬁcient Boundary-Based Blackbox Attack.

In this work, we propose the QEBA method that can perform adversarial attack based only on the final prediction labels of a victim model. 
We theoretically show why previous boundary-based attack with gradient estimation on the whole gradient space is not efﬁcient in terms of query numbers, and provide optimality analysis for our dimension reduction-based gradient estimation. Extensive experiments on ImageNet and CelebA show that compared with the state-of-the-art blackbox attacks, QEBA is able to use a smaller number of queries to achieve a lower magnitude of perturbation with 100% attack success rate.

The code is based on the foolbox project (https://github.com/bethgelab/foolbox).

