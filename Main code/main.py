import os
import pickle
import copy
import math
from os import truncate
from decimal import Decimal, ROUND_DOWN
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.utils.seed import set_seed
from configs.DMTFusion_LLM import *
from src.data.dataset_DMTFusion_LLM import load_features, MultiModalDataset
from src.modules.DMTFusion_LLM import *
from src.training.trainer_DMTFuisonLLM import train_epoch, evaluate_val_loss, evaluate, evaluate_test

def main():
    print("CUDA available:", torch.cuda.is_available())


    set_seed(RANDOM_SEED)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)


    SELECTED_FOLD = 0  # Change to 1~5 to run a certain fold, 0 to run all


    textData, audData, vidData, turn1_data, turn2_data, turn3_data = load_features()

    # Read external 5 folds data
    five_fold_pickle = os.path.join(DATA_PATH, "five_folds.pickle")
    with open(five_fold_pickle, "rb") as f:
        fold_data = pickle.load(f)


    test_accs, test_macro_f1s = [], []
    test_f1_hs, test_aucs = [], []
    test_precision_hs, test_recall_hs = [], []

    fold_range = range(1, 6) if SELECTED_FOLD == 0 else [SELECTED_FOLD]
    print("Start training：")


    for fold_id in fold_range:
        print(f"\n======== Fold {fold_id} ========")


        train_list, train_label = fold_data[f'Fold_{fold_id}']['train']
        val_list, val_label = fold_data[f'Fold_{fold_id}']['val']
        test_list, test_label = fold_data[f'Fold_{fold_id}']['test']


        train_dataset = MultiModalDataset(train_list, train_label, textData, audData, vidData,
                                          turn1_data, turn2_data, turn3_data)
        val_dataset = MultiModalDataset(val_list, val_label, textData, audData, vidData,
                                        turn1_data, turn2_data, turn3_data)
        test_dataset = MultiModalDataset(test_list, test_label, textData, audData, vidData,
                                         turn1_data, turn2_data, turn3_data)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


        model = MultiModalClassifier(
            text_dim=TEXT_FEATURE_DIM,
            audio_dim=AUDIO_FEATURE_DIM,
            video_dim=VIDEO_FEATURE_DIM,
            turn_dim=TURN_FEATURE_DIM,
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        best_val_acc = float('-inf')
        best_val_macro_f1 = float('-inf')
        best_model_weights = None

        for epoch in range(NUM_EPOCHS):

            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

            val_loss = evaluate_val_loss(model, val_loader, criterion, device)
            val_acc, val_macro_f1, val_weighted_f1 = evaluate(model, val_loader, device)
            test_acc, test_macro_f1, test_weighted_f1 = evaluate(model, test_loader, device)

            print(f"Epoch {epoch + 1:02d}: "
                  f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                  f"Val Accuracy={val_acc:.4f}, Macro F1={val_macro_f1:.4f}, Weighted F1={val_weighted_f1:.4f} "
                  f"Test Accuracy={test_acc:.4f}, Test MF1={test_macro_f1:.4f}"
            )

            # Determine whether to update the optimal model based on the macro F1 of the validation set.
            if val_macro_f1 > best_val_macro_f1:
                best_val_macro_f1 = val_macro_f1
                print("Best model updated based on val_macro_f1!")
                best_model_weights = copy.deepcopy(model.state_dict())

        # ========== Training complete, save the best model for this fold ==========
        os.makedirs(SAVE_DIR, exist_ok=True)
        best_model_path = os.path.join(SAVE_DIR, f"hierarchical_llm_fusion_best_model_fold_{fold_id}.pth")
        torch.save(best_model_weights, best_model_path)
        print(f"[*] Best model for fold {fold_id} saved to {best_model_path}")

        # Restore the optimal model for this folder
        model.load_state_dict(best_model_weights)

        # Evaluate on the test set in this fold
        test_acc, test_macro_f1, test_f1_h, test_auc, test_ph, test_rh = evaluate_test(model, test_loader, device)
        print(f"==> [Fold {fold_id} Test] Acc={test_acc:.4f}, MacroF1={test_macro_f1:.4f}, "
              f"F1(H)={test_f1_h:.4f}, AUC={test_auc:.4f}, Prec(H)={test_ph:.4f}, Recall(H)={test_rh:.4f}")

        test_accs.append(test_acc)
        test_macro_f1s.append(test_macro_f1)
        test_f1_hs.append(test_f1_h)
        test_aucs.append(test_auc)
        test_precision_hs.append(test_ph)
        test_recall_hs.append(test_rh)


    avg_acc = np.mean(test_accs)
    std_acc = np.std(test_accs, ddof=1)
    avg_macro_f1 = np.mean(test_macro_f1s)
    std_macro_f1 = np.std(test_macro_f1s, ddof=1)
    avg_f1_h = np.mean(test_f1_hs)
    std_f1_h = np.std(test_f1_hs, ddof=1)
    avg_auc = np.mean(test_aucs)
    std_auc = np.std(test_aucs, ddof=1)
    avg_precision_h = np.mean(test_precision_hs)
    std_precision_h = np.std(test_precision_hs, ddof=1)
    avg_recall_h = np.mean(test_recall_hs)
    std_recall_h = np.std(test_recall_hs, ddof=1)

    print("\n================ Cross-Validation ================")
    print(f"Accuracy       : {avg_acc:.4f} ± {std_acc:.4f}")
    print(f"Macro F1       : {avg_macro_f1:.4f} ± {std_macro_f1:.4f}")
    print(f"F1 (H)         : {avg_f1_h:.4f} ± {std_f1_h:.4f}")
    print(f"AUC            : {avg_auc:.4f} ± {std_auc:.4f}")
    print(f"Precision (H)  : {avg_precision_h:.4f} ± {std_precision_h:.4f}")
    print(f"Recall (H)     : {avg_recall_h:.4f} ± {std_recall_h:.4f}")


if __name__ == "__main__":
    main()  # train_DMTFusion.py