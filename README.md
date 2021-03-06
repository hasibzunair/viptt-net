# ViPTT-Net: Video pretraining of spatio-temporal model for tuberculosis type classification from chest CT scans

ViPTT-Net is a method that pretrains a hybrid CNN-RNN based model on realistic videos for human activity recognition task. It is then fine-tuned on a dataset of chest CT scans for the task of tuberculosis type classification.

ViPTT-Net achieved 2nd place (Kappa score of 0.2)  in the ImageCLEF 2021 Tuberculosis - TBT Classification Challenge.

<p align="center">
  <a href="#"><img src="./media/vipttnet.png"></a> <br />
  <em> 
    Figure 1. Schematic layout of ViPTT-Net.
    </em>
</p>

## Resources

* Paper ([Published](http://ceur-ws.org/Vol-2936/paper-121.pdf), [arXiv](https://arxiv.org/abs/2105.12810))
* Task details are [here](https://www.imageclef.org/2021/medical/tuberculosis)
* [Leaderboard results](https://www.aicrowd.com/challenges/imageclef-2021-tuberculosis-tbt-classification/leaderboards)

## Citation

If you use this code or models in your scientific work, please cite the
following paper:

```
H. Zunair, A. Rahman, N. Mohammed, ViPTT-Net: Video pretraining of spatio-temporal
model for tuberculosis type classification from chest CT scans, in: CLEF2021 Working
Notes, CEUR Workshop Proceedings, CEUR-WS.org <http://ceur-ws.org>, Bucharest,
Romania, 2021.
```

## Installation

This code requires:

* Python 3.7
* TensorFlow 2.4.1
* Nibabel

This research code will not be maintained, unless we decide to do a follow up work. If you have trouble running this code ONLY with the requirements mentioned above, file and issue and we'll look at it tomorrow.  

## Preparing training and test datasets
See `notebooks/`.

## Training scripts
See `notebooks/`.

## Evaluation scripts
See `notebooks/`.

## Pretrained models
We provide pretrained models:

| Models | Description | Weights|
|:---:|:---:|:---:|
| ViPTT-Net ImageCLEF | Fine-tunes `ViPTT-Net UCF50` on ImageCLEF 2021 Tuberculosis - TBT dataset.| [ViPTT-Net-CLEF-TBT.h5](https://github.com/hasibzunair/ViPTT-Net/releases/latest/download/ViPTT-Net-CLEF-TBT.h5) |
| ViPTT-Net UCF50 | Trains ViPTT-Net on a subset of the UCF50 dataset | [ViPTT-Net-UCF50.h5](https://github.com/hasibzunair/ViPTT-Net/releases/latest/download/ViPTT-Net-UCF50.h5) |

## Results
See paper for details!

## License
MIT

