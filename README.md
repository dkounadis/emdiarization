# EMdiarisation

MATLAB code for An EM algorithm for simultaneous Audio Source Separation and Diarization using NMF in STFT domain implementing this paper

```
D. Kounades-Bastian, L. Girin, X. Alameda-Pineda, S. Gannot and R. Horaud, 
"An EM algorithm for joint source separation and diarisation of multichannel convolutive speech mixtures," 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), New Orleans, LA,
pp. 16-20, doi: 10.1109/ICASSP.2017.7951789.
```

## DEMO

```python
# In MATLAB console
>>> example

# `example.m` generates a stereo mix of 3 sources (by loading trueSrc1.wav, ..)
# Then calls `initNMF.m` to provide initialisation params via binary masking.
# Then applies `emd.m` to separate the sources. 
# Separated sources will be saved as .wav files (estimatedSrc1.wav, etc.) in the directory `./results/`. 
```



## PAPER
  - [pdf](https://inria.hal.science/hal-01430761/file/diarisation_camready.pdf)
  - [Slides](https://team.inria.fr/perception/files/2018/05/slidiar.pdf)




