# ViPTT-Net: Video pretraining of spatio-temporal models for tuberculosis type classification

ViPTT-Net is a method that pretrains a hybrid CNN-RNN based model on realistic videos for human activity recognition task. It is then fine-tuned on a dataset of chest CT scans for the task of tuberculosis type classification. 

ViPTT-Net achieved 2nd place in the ImageCLEF 2021 Tuberculosis - TBT classification task.

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

## Pretrained models

We provide pretrained models:

| Models | Description | Weights|
|:---:|:---:|:---:|
| ViPTT-Net ImageCLEF | Fine-tunes `ViPTT-Net UCF50` on ImageCLEF 2021 Tuberculosis - TBT dataset.| [ViPTT-Net-CLEF-TBT.h5](https://github.com/hasibzunair/ViPTT-Net/releases/latest/download/ViPTT-Net-CLEF-TBT.h5) |
| ViPTT-Net UCF50 | Trains ViPTT-Net on a subset of the UCF50 dataset | [ViPTT-Net-UCF50.h5](https://github.com/hasibzunair/ViPTT-Net/releases/latest/download/ViPTT-Net-UCF50.h5) |

## Results
Table will be added. See paper in the meantime!

## License 

MIT

