import torch.nn as nn
import torch


class bertModel(nn.Module):
    def __init__(self, transformer):
        super(bertModel, self).__init__()
        self.transformer = transformer
        self.fc_middle = nn.Sequential(nn.ReLU(),
                                nn.Linear(8, 8),
                                nn.Dropout(0.25)
                                )
        self.fc_quadrant = nn.Sequential(
                                nn.ReLU(inplace=True),
                                nn.Linear(8, 4),
                                nn.Dropout(0.25)
                                )

    def forward(self, b_input_ids, token_type_ids=None, attention_mask=None, labels=None):
        outputs = self.transformer(b_input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels)
        output = self.fc_middle(outputs[1])
        fc_quadrant_out = self.fc_quadrant(output)
        return fc_quadrant_out
