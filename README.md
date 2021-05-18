# ViPTT-Net: Video pretraining of spatio-temporal models for tuberculosis type classification

ViPTT-Net is a method that pretrains a hybrid CNN-RNN based model on human activity recognition task, and then fine-tunes the model for the task of tuberculosis type classification from chest CT scans. 

It achieved 2nd place in the ImageCLEF 2021 Tuberculosis - TBT classification task.

## Resources
* Paper (arXiv)
* [Leaderboard results](https://www.aicrowd.com/challenges/imageclef-2021-tuberculosis-tbt-classification/leaderboards)

## Citation

If you use this code or models in your scientific work, please cite the
following paper:

```bibtex
@article{zunair2020melanoma,
  title={Melanoma detection using adversarial training and deep transfer learning},
  author={Zunair, Hasib and Hamza, A Ben},
  journal={Physics in Medicine \& Biology},
  year={2020},
  publisher={IOP Publishing}
}
```

## Installation

This code requires:

* Python 3.7
* TensorFlow 2.4.1
* Nibabel

## Preparing training and test datasets

See `notebooks/`.

## Training scripts

See `notebooks/`.

## Evaluation scripts

See `notebooks/`.

## Pre-trained models

We provide pre-trained models:

| Models | Weights|
|:---:|:---:|
| ViPTT-Net (PT+CW+AUG) | [ViPTT-Net-CLEF-TBT.h5](https://github.com/hasibzunair/ViPTT-Net/releases/latest/download/ViPTT-Net-CLEF-TBT.h5) |
| ViPTT-Net UCF50 | [ViPTT-Net-UCF50.h5](https://github.com/hasibzunair/ViPTT-Net/releases/latest/download/ViPTT-Net-UCF50.h5) |

## Results
Table will be added. See paper for details.

## License 

MIT

