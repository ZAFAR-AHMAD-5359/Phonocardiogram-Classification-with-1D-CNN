# Five-class heart-sound classification with a 1D CNN

Publication materials and historical source excerpts for **Phonocardiogram Classification Based on 1D CNN with Pitch-Shifting and Signal Uniformity Techniques**, published at EUSIPCO 2024. Zafar Ahmad is the first author.

The study evaluates five heart-sound classes: **normal, aortic stenosis, mitral stenosis, mitral regurgitation and mitral valve prolapse**. It is separate from the [ICAI paediatric normal/abnormal study](https://github.com/ZAFAR-AHMAD-5359/PCG-Pediatric-CHD-Detection). The paper does not establish a paediatric population for this five-class dataset.

## Publication

Zafar Ahmad, Muhammad Salman Khan, Muhammad E. H. Chowdhury, Susu Zughaier and Wanis Hamad Ibrahim. “Phonocardiogram Classification Based on 1D CNN with Pitch-Shifting and Signal Uniformity Techniques.” *2024 32nd European Signal Processing Conference (EUSIPCO)*, pp. 1536–1540, 2024.

- [Publisher record and DOI](https://doi.org/10.23919/EUSIPCO63174.2024.10715037)
- [Open proceedings paper — EURASIP](https://eurasip.org/Proceedings/Eusipco/Eusipco2024/pdfs/0001536.pdf)
- [Official conference program and author list](https://cmsworkshops.com/EUSIPCO2024/view_paper.php?PaperNum=1705)
- [Existing paper file](Phonocardiogram_Classification_Based_on_1D_CNN_with_Pitch-Shifting_and_Signal_Uniformity_Techniques.pdf)

The author spelling above follows the official conference program.

## Research scope

The paper describes a public dataset with 200 recordings per class, sampled at 8 kHz. Its method standardises recordings to three seconds, applies pitch-shifting augmentation, extracts mel-frequency cepstral coefficients and evaluates a one-dimensional convolutional neural network.

Read the paper for the model configurations, evaluation procedure and reported results. These are research dataset results; they do not establish clinical deployment, prospective patient-level performance or measured real-time device operation. No new reproduction or clinical validation is reported by this repository update.

## What is available here

| File | Contents |
|---|---|
| Paper PDF linked above | Existing copy of the published paper |
| [code/README.md](code/README.md) | Historical Python/Colab model-training and evaluation source excerpt |
| [data/README.md](data/README.md) | Link to the dataset source used in the study |
| [requirements.txt](requirements.txt) | Existing dependency list; no tested reproduction environment is supplied |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Existing general contribution guidance |

The source excerpt mounts Google Drive and expects WAV recordings and spreadsheet labels in personal training/test folders. Those inputs, prepared datasets and trained model weights are not included. There is no packaged classifier API or runnable inference demo in the current inventory. Installing the dependency list alone therefore does not reproduce the publication. The excerpt is preserved as historical material; execution and exact correspondence with every published experiment have not been verified here.

## Data access and reuse

The existing [data note](data/README.md) points to the [heart-sound dataset source](https://github.com/yaseen21khan/Classification-of-Heart-Sound-Signal-Using-Multiple-Features-/blob/master/README.md). Obtain data from that source under its applicable access and reuse terms; recordings are not bundled here. Check the original source documentation and the paper when preparing an independent reproduction.

A `LICENSE` file is not currently included. For repository reuse permissions, contact the maintainer. The publication and external dataset remain subject to their respective terms.

## Citation

```bibtex
@inproceedings{ahmad2024phonocardiogram,
  author = {Ahmad, Zafar and Khan, Muhammad Salman and
            Chowdhury, Muhammad E. H. and Zughaier, Susu and
            Ibrahim, Wanis Hamad},
  title = {Phonocardiogram Classification Based on 1D CNN with Pitch-Shifting and Signal Uniformity Techniques},
  booktitle = {2024 32nd European Signal Processing Conference (EUSIPCO)},
  year = {2024},
  pages = {1536--1540},
  doi = {10.23919/EUSIPCO63174.2024.10715037}
}
```

## Contact

Zafar Ahmad — [ahmadzafar577@gmail.com](mailto:ahmadzafar577@gmail.com)

[Cardiovascular research portfolio](https://zafar-ahmad-5359.github.io/research.html) · [Google Scholar](https://scholar.google.com/citations?user=D5W9TVwAAAAJ&hl=en) · [LinkedIn](https://www.linkedin.com/in/zafar-ahmad-ab87b6183/)

Documentation curated on 10 October 2026. Published paper, existing source excerpt, data note and repository history are preserved.
