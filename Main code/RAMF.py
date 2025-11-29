
import torch
import torch.nn as nn
from .DMTFusion_MLP_LLM import TextMLP, AudioMLP, VideoMLP, TurnMLP
from .Dynamic_Temporal_Feature_Modelling import DynamicTemporalFusion
from .Crosshead_Multimodal_Topology_Attention import MultiTokenAttention


class MultiModalClassifier(nn.Module):
    def __init__(self, text_dim, audio_dim, video_dim, turn_dim, hidden_dim=256, num_classes=2,
                 num_heads=4, num_layers=2, dropout=0.1, classifier_dropout=0.3):
        super().__init__()

        # Three-modal MLP projector + turn1 MLP projector
        self.text_mlp = TextMLP(text_dim, hidden_dim, dropout)
        self.audio_mlp = AudioMLP(audio_dim, hidden_dim, dropout)
        self.video_mlp = VideoMLP(video_dim, hidden_dim, dropout)
        self.turn1_mlp = TurnMLP(turn_dim, hidden_dim, dropout)

        # MLP projectors for turn2 and turn3
        self.turn2_mlp = TurnMLP(turn_dim, hidden_dim, dropout)
        self.turn3_mlp = TurnMLP(turn_dim, hidden_dim, dropout)

        # Local-Gobal Context Fusion
        self.seq_pool = DynamicTemporalFusion(hidden_dim)

        # First layer CMTA: integration of text + audio + video + turn1 (four modalities)
        self.first_fusion = MultiTokenAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)

        # Second layer CMTA: Combine the output of the first layer + turn2 + turn3
        self.second_fusion = MultiTokenAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, text_input, audio_input, video_input, turn1_input, turn2_input, turn3_input):
        # Step 1: Project the original three modalities + turn1 to hidden_dim
        text_projected = self.text_mlp(text_input)  # [B, seq_len, hidden_dim]
        audio_projected = self.audio_mlp(audio_input)  # [B, seq_len, hidden_dim]
        video_projected = self.video_mlp(video_input)  # [B, seq_len, hidden_dim]
        turn1_projected = self.turn1_mlp(turn1_input)  # [B, hidden_dim]

        # Step 2: Pool sequence features into a single vector
        text_pooled = self.seq_pool(text_projected)  # [B, hidden_dim]
        audio_pooled = self.seq_pool(audio_projected)  # [B, hidden_dim]
        video_pooled = self.seq_pool(video_projected)  # [B, hidden_dim]


        # Step 3: First layer CMTA fusion - text + audio + video + turn1 (four modalities)
        first_stage_tokens = torch.stack([text_pooled, audio_pooled, video_pooled, turn1_projected],
                                         dim=1)  # [B, 4, hidden_dim]
        first_fused = self.first_fusion(first_stage_tokens)  # [B, 4, hidden_dim]
        first_fused_pooled = first_fused.mean(dim=1)  # [B, hidden_dim]

        # Step 4: Project turn2 and turn3
        turn2_projected = self.turn2_mlp(turn2_input)  # [B, hidden_dim]
        turn3_projected = self.turn3_mlp(turn3_input)  # [B, hidden_dim]

        # Step 5: Second layer CMTA fusion - First layer fusion result + turn2 + turn3
        second_stage_tokens = torch.stack([first_fused_pooled, turn2_projected, turn3_projected],
                                          dim=1)  # [B, 3, hidden_dim]
        second_fused = self.second_fusion(second_stage_tokens)  # [B, 3, hidden_dim]
        final_fused = second_fused.mean(dim=1)  # [B, hidden_dim]

        # Step 6: Classification
        logits = self.classifier(final_fused)

        return logits


class MultiModalClassifier_SimpleSingleStep(nn.Module):
    """
    Simplified single-step fusion multimodal classifier: directly fuses six modalities with a simpler structure.
    """

    def __init__(self, text_dim, audio_dim, video_dim, turn_dim, hidden_dim=256, num_classes=2,
                 num_heads=4, dropout=0.1, classifier_dropout=0.3):
        super().__init__()

        # Six-modal MLP projector
        self.text_mlp = TextMLP(text_dim, hidden_dim, dropout)
        self.audio_mlp = AudioMLP(audio_dim, hidden_dim, dropout)
        self.video_mlp = VideoMLP(video_dim, hidden_dim, dropout)
        self.turn1_mlp = TurnMLP(turn_dim, hidden_dim, dropout)
        self.turn2_mlp = TurnMLP(turn_dim, hidden_dim, dropout)
        self.turn3_mlp = TurnMLP(turn_dim, hidden_dim, dropout)

        # Time-series pooling layer
        self.seq_pool = DynamicTemporalFusion(hidden_dim)

        # Simple six-modal fusion: a single layer of attention mechanism
        self.six_modal_fusion = MultiTokenAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, text_input, audio_input, video_input, turn1_input, turn2_input, turn3_input):


        # Step 1: Project all modes
        text_projected = self.text_mlp(text_input)
        audio_projected = self.audio_mlp(audio_input)
        video_projected = self.video_mlp(video_input)
        turn1_projected = self.turn1_mlp(turn1_input)
        turn2_projected = self.turn2_mlp(turn2_input)
        turn3_projected = self.turn3_mlp(turn3_input)

        # Step 2: Pooling sequence features
        text_pooled = self.seq_pool(text_projected)
        audio_pooled = self.seq_pool(audio_projected)
        video_pooled = self.seq_pool(video_projected)

        # Step 3: Construct a 6-modal token sequence and fuse it directly
        six_modal_tokens = torch.stack([
            text_pooled, audio_pooled, video_pooled,
            turn1_projected, turn2_projected, turn3_projected
        ], dim=1)  # [B, 6, hidden_dim]

        # Step 4: Single-step fusion
        fused_tokens = self.six_modal_fusion(six_modal_tokens)  # [B, 6, hidden_dim]
        final_fused = fused_tokens.mean(dim=1)  # [B, hidden_dim]

        # Step 5: Clasifier
        logits = self.classifier(final_fused)

        return logits


class MultiModalClassifierAblation1(nn.Module):
    """
    Fusion experiment 1: The first layer of CMTA does not include turn1, only merging text+audio+video, then the second layer of CMTA includes turn1+turn2+turn3.
    """
    def __init__(self, text_dim, audio_dim, video_dim, turn_dim, hidden_dim=256, num_classes=2,
                 num_heads=4, num_layers=2, dropout=0.1, classifier_dropout=0.3):
        super().__init__()

        # 原有的三个模态MLP投影器 + turn MLP投影器
        self.text_mlp = TextMLP(text_dim, hidden_dim, dropout)
        self.audio_mlp = AudioMLP(audio_dim, hidden_dim, dropout)
        self.video_mlp = VideoMLP(video_dim, hidden_dim, dropout)
        self.turn1_mlp = TurnMLP(turn_dim, hidden_dim, dropout)
        self.turn2_mlp = TurnMLP(turn_dim, hidden_dim, dropout)
        self.turn3_mlp = TurnMLP(turn_dim, hidden_dim, dropout)

        # 时序池化层
        self.seq_pool = DynamicTemporalFusion(hidden_dim)

        # 第一层CMTA：只融合text + audio + video (三个模态)
        self.first_fusion = MultiTokenAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)

        # 第二层CMTA：融合第一层输出 + turn1 + turn2 + turn3 (四个输入)
        self.second_fusion = MultiTokenAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, text_input, audio_input, video_input, turn1_input, turn2_input, turn3_input):

        text_projected = self.text_mlp(text_input)  # [B, seq_len, hidden_dim]
        audio_projected = self.audio_mlp(audio_input)  # [B, seq_len, hidden_dim]
        video_projected = self.video_mlp(video_input)  # [B, seq_len, hidden_dim]


        text_pooled = self.seq_pool(text_projected)  # [B, hidden_dim]
        audio_pooled = self.seq_pool(audio_projected)  # [B, hidden_dim]
        video_pooled = self.seq_pool(video_projected)  # [B, hidden_dim]


        first_stage_tokens = torch.stack([text_pooled, audio_pooled, video_pooled],
                                         dim=1)  # [B, 3, hidden_dim]
        first_fused = self.first_fusion(first_stage_tokens)  # [B, 3, hidden_dim]
        first_fused_pooled = first_fused.mean(dim=1)  # [B, hidden_dim]


        turn1_projected = self.turn1_mlp(turn1_input)  # [B, hidden_dim]
        turn2_projected = self.turn2_mlp(turn2_input)  # [B, hidden_dim]
        turn3_projected = self.turn3_mlp(turn3_input)  # [B, hidden_dim]


        second_stage_tokens = torch.stack([first_fused_pooled, turn1_projected, turn2_projected, turn3_projected],
                                          dim=1)  # [B, 4, hidden_dim]
        second_fused = self.second_fusion(second_stage_tokens)  # [B, 4, hidden_dim]
        final_fused = second_fused.mean(dim=1)  # [B, hidden_dim]


        logits = self.classifier(final_fused)

        return logits


class MultiModalClassifierAblation2(nn.Module):
    """
    Fusion experiment 2: The first layer of CMTA only fuses three modalities (text + audio + video), while the second layer of CMTA fuses the output of the first layer + turn2 + turn3 (excluding turn1).
    """
    def __init__(self, text_dim, audio_dim, video_dim, turn_dim, hidden_dim=256, num_classes=2,
                 num_heads=4, num_layers=2, dropout=0.1, classifier_dropout=0.3):
        super().__init__()


        self.text_mlp = TextMLP(text_dim, hidden_dim, dropout)
        self.audio_mlp = AudioMLP(audio_dim, hidden_dim, dropout)
        self.video_mlp = VideoMLP(video_dim, hidden_dim, dropout)
        self.turn1_mlp = TurnMLP(turn_dim, hidden_dim, dropout)
        self.turn2_mlp = TurnMLP(turn_dim, hidden_dim, dropout)
        self.turn3_mlp = TurnMLP(turn_dim, hidden_dim, dropout)


        self.seq_pool = DynamicTemporalFusion(hidden_dim)


        self.first_fusion = MultiTokenAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)


        self.second_fusion = MultiTokenAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)


        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, text_input, audio_input, video_input, turn1_input, turn2_input, turn3_input):

        text_projected = self.text_mlp(text_input)  # [B, seq_len, hidden_dim]
        audio_projected = self.audio_mlp(audio_input)  # [B, seq_len, hidden_dim]
        video_projected = self.video_mlp(video_input)  # [B, seq_len, hidden_dim]


        # turn1_projected = self.turn1_mlp(turn1_input)  # not ues


        text_pooled = self.seq_pool(text_projected)  # [B, hidden_dim]
        audio_pooled = self.seq_pool(audio_projected)  # [B, hidden_dim]
        video_pooled = self.seq_pool(video_projected)  # [B, hidden_dim]


        first_stage_tokens = torch.stack([text_pooled, audio_pooled, video_pooled],
                                         dim=1)  # [B, 3, hidden_dim]
        first_fused = self.first_fusion(first_stage_tokens)  # [B, 3, hidden_dim]
        first_fused_pooled = first_fused.mean(dim=1)  # [B, hidden_dim]


        turn2_projected = self.turn2_mlp(turn2_input)  # [B, hidden_dim]
        turn3_projected = self.turn3_mlp(turn3_input)  # [B, hidden_dim]


        second_stage_tokens = torch.stack([first_fused_pooled, turn2_projected, turn3_projected],
                                          dim=1)  # [B, 3, hidden_dim]
        second_fused = self.second_fusion(second_stage_tokens)  # [B, 3, hidden_dim]
        final_fused = second_fused.mean(dim=1)  # [B, hidden_dim]


        logits = self.classifier(final_fused)

        return logits


class MultiModalClassifierAblation3(nn.Module):
    """
    Melting experiment 3: Only the first layer of CMTA (four modalities: text+audio+video+turn1), direct output classification
    """

    def __init__(self, text_dim, audio_dim, video_dim, turn_dim, hidden_dim=256, num_classes=2,
                 num_heads=4, num_layers=2, dropout=0.1, classifier_dropout=0.3):
        super().__init__()


        self.text_mlp = TextMLP(text_dim, hidden_dim, dropout)
        self.audio_mlp = AudioMLP(audio_dim, hidden_dim, dropout)
        self.video_mlp = VideoMLP(video_dim, hidden_dim, dropout)
        self.turn1_mlp = TurnMLP(turn_dim, hidden_dim, dropout)


        self.turn2_mlp = TurnMLP(turn_dim, hidden_dim, dropout)
        self.turn3_mlp = TurnMLP(turn_dim, hidden_dim, dropout)


        self.seq_pool = DynamicTemporalFusion(hidden_dim)


        self.first_fusion = MultiTokenAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)


        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, text_input, audio_input, video_input, turn1_input, turn2_input, turn3_input):

        text_projected = self.text_mlp(text_input)  # [B, seq_len, hidden_dim]
        audio_projected = self.audio_mlp(audio_input)  # [B, seq_len, hidden_dim]
        video_projected = self.video_mlp(video_input)  # [B, seq_len, hidden_dim]
        turn1_projected = self.turn1_mlp(turn1_input)  # [B, hidden_dim]


        # turn2_projected = self.turn2_mlp(turn2_input)  # not ues
        # turn3_projected = self.turn3_mlp(turn3_input)  # not ues


        text_pooled = self.seq_pool(text_projected)  # [B, hidden_dim]
        audio_pooled = self.seq_pool(audio_projected)  # [B, hidden_dim]
        video_pooled = self.seq_pool(video_projected)  # [B, hidden_dim]


        first_stage_tokens = torch.stack([text_pooled, audio_pooled, video_pooled, turn1_projected],
                                         dim=1)  # [B, 4, hidden_dim]
        first_fused = self.first_fusion(first_stage_tokens)  # [B, 4, hidden_dim]
        final_fused = first_fused.mean(dim=1)  # [B, hidden_dim]


        logits = self.classifier(final_fused)

        return logits