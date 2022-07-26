import sys
import torch
from transformers import BertTokenizer
import time
import torch.nn.functional as F
from src.features.preprocess_feature_creation import nrc_feats, vad_feats, create_dataloaders_BERT
from src.helpers import format_time
from src.helpers import probs_to_labels_multilabel
from src.config import BATCH_SIZE, MAX_LEN, project_root_path
sys.path.append(project_root_path)


class MulticlassEmotionIdentification:
    def __init__(self):
        # self.model = torch.load("C:/Users/georgiapant/PycharmProjects/GitHub/rebecca/models/multiclass_emotion.pt")
        self.model = torch.jit.load(project_root_path+'/models/model_scripted_multiclass_emotion_VAD_NRC.pt')
        self.tokenizer = BertTokenizer.from_pretrained(project_root_path+"/models/tokenizer/")
        self.MAX_LEN = MAX_LEN
        self.BATCH_SIZE = BATCH_SIZE
        self.device = torch.device('cuda')

    def inference(self, content, labels=None):
        """
            Perform a forward pass on the trained BERT model to predict probabilities
            on the test set.
            """
        # Put the model into the evaluation mode. The dropout layers are disabled during
        # the test time.
        t0 = time.time()
        model = self.model.to(self.device)
        model.eval()

        test_dataloader = create_dataloaders_BERT(content, labels, self.tokenizer, self.MAX_LEN, self.BATCH_SIZE
                                             , sampler='sequential')

        all_logits = []

        # For each batch in our test set...
        for batch in test_dataloader:
            # Load batch to GPU
            b_input_ids, b_attn_mask = tuple(t.to(self.device) for t in batch)[:2]
            lex_feats = nrc_feats(b_input_ids, self.tokenizer).to(self.device)
            vad = vad_feats(b_input_ids, self.tokenizer, self.MAX_LEN).to(self.device)

            # Compute logits
            with torch.no_grad():
                logits = self.model(b_input_ids, b_attn_mask, lex_feats, vad)
            all_logits.append(logits)

        # Concatenate logits from each batch
        all_logits = torch.cat(all_logits, dim=0)

        # Apply softmax to calculate probabilities
        probs = F.softmax(all_logits, dim=1).cpu().numpy()

        print(f"Total prediction time: {format_time(time.time() - t0)}", flush=True)

        predicted_labels = probs_to_labels_multilabel(probs)

        return predicted_labels
