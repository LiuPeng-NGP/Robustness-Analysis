# Robustness-Analysis
Code for ICLR 2026 [Robustness of Probabilistic Models to Low-Quality Data: A Multi-Perspective Analysis](https://openreview.net/forum?id=ZFZhV7Snf4)

# Environment
## Image
conda create -n data_quality python=3.10 pip
conda activate data_quality
pip3 install torch torchvision pyyaml tqdm pytorch-fid

## Text
conda create -n data_quality_text python=3.9 pip
conda activate data_quality_text
pip3 install torch tqdm tiktoken wandb==0.9.4 numpy==1.23 datasets==2.0.0 importlib-metadata
pip install importlib-metadata

## Machine Translation and Text Summarization
conda create -n data_quality_mtts python=3.10.18 pip -y
conda activate data_quality_mtts
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda==12.1 datasets==4.0.0 tokenizers==0.21.0 numpy==2.2.6 matplotlib==3.10.6 -c pytorch -c nvidia -c conda-forge -y
pip install torchtext==0.18.0 bert-score==0.3.13 sacrebleu==2.5.1 rouge-score==0.1.2

# Experiment
For details about the experiment, please refer to the README files in the subdirectories.

# Citation
```bib
@inproceedings{Liu2026robustness,
  title={Robustness of Probabilistic Models to Low-Quality Data: A Multi-Perspective Analysis},
  author={Liu, Peng and Jin, Yaochu},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=ZFZhV7Snf4}
}
```