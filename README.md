# Project Overview

This project contains the main program, model architecture, training pipeline, demonstration scripts, and associated five-fold data files. It is designed for multimodal/text classification tasks.

## File Descriptions

- `main.py`: **Main execution script** that integrates data loading, model instantiation, training, and evaluation.
- `RAMF.py`: **Core model implementation**, containing the primary model architecture and forward logic.
- `trainner.py`: **Training manager**, handling the training loop, loss computation, optimization, and evaluation.

## Demo Files

- `Demo_MF`: Demonstration script for running the MF model.
- `Demo_RAMF`: Demonstration script for running the RAMF model.

## Data Files

Five-fold cross-validation datasets:

- `five_fold_MHC.pkl`
- `five_fold_MHC_E.pkl`
- `five_folds_HateMM.pickle`

## Notes

- The current release focuses on the RAMF model and its associated training pipeline.
- The implementations of **LGCF** (Local-Global Context Fusion), **SCA** (Semantic Coross Attention) modules and detailed configuration code will be made publicly available **after the review process**.
- The relevant parameter settings for the experiment can be found in the appendix.
