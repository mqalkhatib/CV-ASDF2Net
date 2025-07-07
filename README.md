# CV-ASDF2Net
This is an implementation of "PolSAR Image Classification using Shallow to Deep Feature Fusion Network with Complex Valued Attention" Accepted for Publication on Scientific Reports. The paper can be accessed through:
https://www.nature.com/articles/s41598-025-10475-3
![image](https://github.com/user-attachments/assets/52261d53-1451-4cb9-9503-3da89b1b1c6a)

# Datasets
Three benchmark datasets were used in this paper, namely Flevoland, San Francisco and Oberpfaffenhofen, dataset can be downloaded from:
https://mega.nz/folder/WhgT1L4S#PnMttCUpjtwkD8qTEdwZsw

# Requirement
Python 3.9.18, Tensorflow (and Keras) 2.10.0, cvnn 2.0, Tensorflow Probability 0.18.0

# Results
To quantitatively measure the proposed CV-ASDF2Net model, three evaluation metrics are employed to verify the effectiveness of the algorithm, Overall Accuracy (OA), Average Accuracy (AA) and Cohen's Kappa (k). Also, Each class accuracy has been reported
![image](https://github.com/user-attachments/assets/33cb2b25-b5f1-4277-a475-56e8831339e0)

Model was qualitatively evaluated by visually comparing the resulting class maps.
![image](https://github.com/user-attachments/assets/b3e767e3-59f0-4401-920b-2efd2014bfa6)

# Citation
@ARTICLE{Alkhatib2025-kz,
  title     = "{PolSAR} image classification using shallow to deep feature
               fusion network with complex valued attention",
  author    = "Alkhatib, Mohammed Q and Zitouni, M Sami and Al-Saad, Mina and
               Aburaed, Nour and Al-Ahmad, Hussain",
  journal   = "Sci. Rep.",
  publisher = "Nature Publishing Group",
  volume    =  15,
  number    =  1,
  pages     = "1--19",
  month     =  jul,
  year      =  2025,
  language  = "en"
}

