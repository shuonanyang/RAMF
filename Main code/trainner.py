

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix
)


def train_epoch(model, train_loader, criterion, optimizer, device):

    model.train()


    total_loss = 0.0
    num_batches = 0


    for batch in train_loader:
        # Unpacking the new data format: contains the original 3 modalities + 3 turn features
        # text_feat: Original text features [batch_size, seq_len, text_dim]
        # audio_feat: Audio features [batch_size, seq_len, audio_dim]
        # video_feat: Video features [batch_size, seq_len, video_dim]
        # turn1_feat: Turn1 objective description features [batch_size, turn_dim]
        # turn2_feat: Turn2 hatred hypothesis features [batch_size, turn_dim]
        # turn3_feat: Turn3 non-hatred hypothesis features [batch_size, turn_dim]
        # labels: Labels [batch_size]
        text_feat, audio_feat, video_feat, turn1_feat, turn2_feat, turn3_feat, labels = batch


        text_feat = text_feat.to(device)
        audio_feat = audio_feat.to(device)
        video_feat = video_feat.to(device)
        turn1_feat = turn1_feat.to(device)
        turn2_feat = turn2_feat.to(device)
        turn3_feat = turn3_feat.to(device)
        labels = labels.to(device)


        optimizer.zero_grad()


        outputs = model(text_feat, audio_feat, video_feat,
                        turn1_feat, turn2_feat, turn3_feat)


        loss = criterion(outputs, labels)


        loss.backward()


        optimizer.step()


        total_loss += loss.item()
        num_batches += 1


    return total_loss / num_batches


def evaluate_val_loss(model, val_loader, criterion, device):

    model.eval()

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:

            text_feat, audio_feat, video_feat, turn1_feat, turn2_feat, turn3_feat, labels = batch


            text_feat = text_feat.to(device)
            audio_feat = audio_feat.to(device)
            video_feat = video_feat.to(device)
            turn1_feat = turn1_feat.to(device)
            turn2_feat = turn2_feat.to(device)
            turn3_feat = turn3_feat.to(device)
            labels = labels.to(device)


            outputs = model(text_feat, audio_feat, video_feat,
                            turn1_feat, turn2_feat, turn3_feat)


            loss = criterion(outputs, labels)

            total_loss += loss.item()
            num_batches += 1


    return total_loss / num_batches


def evaluate(model, data_loader, device):

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:

            text_feat, audio_feat, video_feat, turn1_feat, turn2_feat, turn3_feat, labels = batch


            text_feat = text_feat.to(device)
            audio_feat = audio_feat.to(device)
            video_feat = video_feat.to(device)
            turn1_feat = turn1_feat.to(device)
            turn2_feat = turn2_feat.to(device)
            turn3_feat = turn3_feat.to(device)
            labels = labels.to(device)


            outputs = model(text_feat, audio_feat, video_feat,
                            turn1_feat, turn2_feat, turn3_feat)


            _, preds = torch.max(outputs, 1)


            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')

    return accuracy, macro_f1, weighted_f1


def evaluate_test(model, data_loader, device):

    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in data_loader:

            text_feat, audio_feat, video_feat, turn1_feat, turn2_feat, turn3_feat, labels = batch


            text_feat = text_feat.to(device)
            audio_feat = audio_feat.to(device)
            video_feat = video_feat.to(device)
            turn1_feat = turn1_feat.to(device)
            turn2_feat = turn2_feat.to(device)
            turn3_feat = turn3_feat.to(device)
            labels = labels.to(device)


            outputs = model(text_feat, audio_feat, video_feat,
                            turn1_feat, turn2_feat, turn3_feat)


            probs = torch.softmax(outputs, dim=1)


            _, preds = torch.max(outputs, 1)


            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())


    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)


    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')


    f1_hate = f1_score(all_labels, all_preds, pos_label=1)


    if all_probs.shape[1] == 2:

        auc = roc_auc_score(all_labels, all_probs[:, 1])
    else:

        auc = 0.0


    precision_hate = precision_score(all_labels, all_preds, pos_label=1, zero_division=0)
    recall_hate = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)

    return accuracy, macro_f1, f1_hate, auc, precision_hate, recall_hate