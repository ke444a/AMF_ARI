import json
import logging
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from xaif_eval import xaif

logger = logging.getLogger(__name__)

MODEL_ID = "raruidol/ArgumentMining-EN-ARI-AIF-RoBERTa_L"
BATCH_SIZE = 4096
LABEL_THRESHOLDS = {
    "Inference": (1, 0.9),
    "Conflict": (2, 0.75),
    "Rephrase": (3, 0.9),
}


def _load_config():
    config_path = Path(__file__).parent.parent / "config" / "config.json"
    if config_path.exists():
        with config_path.open() as f:
            return json.load(f)
    return {"model_path": MODEL_ID}


def _load_model():
    config = _load_config()
    model_path = config.get("model_path", MODEL_ID)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for AMF_ARI GPU inference")

    logger.info("Loading CUDA model from: %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
    ).to("cuda")
    model.eval()

    return tokenizer, model


TOKENIZER, MODEL = _load_model()


def preprocess_data(filexaif, wnd_size):
    idents = []
    idents_comb = []
    propositions = {}
    data = {"text": [], "text2": []}

    for node in filexaif["nodes"]:
        if node["type"] == "I":
            propositions[node["nodeID"]] = node["text"]
            idents.append(node["nodeID"])

    if wnd_size == -1:
        window_size = len(idents)
    elif wnd_size < 2:
        return data, idents_comb, propositions
    else:
        window_size = min(wnd_size, len(idents))

    # Pair each proposition with later propositions that still fall within the
    # requested window so every eligible pair is emitted exactly once.
    for left in range(len(idents) - 1):
        right_end = min(len(idents), left + window_size)
        for right in range(left + 1, right_end):
            pair = (idents[left], idents[right])
            idents_comb.append(pair)
            data["text"].append(propositions[pair[0]])
            data["text2"].append(propositions[pair[1]])

    return data, idents_comb, propositions


def model_predictions(data):
    labels = []

    id2label = MODEL.config.id2label
    for start in range(0, len(data["text"]), BATCH_SIZE):
        text_batch = data["text"][start : start + BATCH_SIZE]
        text_pair_batch = data["text2"][start : start + BATCH_SIZE]
        inputs = TOKENIZER(
            text_batch,
            text_pair_batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {
            key: value.to("cuda", non_blocking=True)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            logits = MODEL(**inputs).logits
            probabilities = torch.softmax(logits, dim=-1)
            scores, label_ids = torch.max(probabilities, dim=-1)

        for score, label_id in zip(scores.tolist(), label_ids.tolist()):
            label = id2label[label_id]
            mapped = LABEL_THRESHOLDS.get(label)
            if mapped is not None and score >= mapped[1]:
                labels.append(mapped[0])
            else:
                labels.append(0)

    return labels


def output_xaif(idents, labels, fileaif):
    original_aif = xaif.AIF(fileaif)

    for i in range(len(labels)):
        lb = labels[i]

        if lb == 0:
            continue

        elif lb == 1:
            # Add the RA node
            original_aif.add_component(
                "argument_relation", "RA", idents[i][1], idents[i][0]
            )

        elif lb == 2:
            # Add the CA node
            original_aif.add_component(
                "argument_relation", "CA", idents[i][1], idents[i][0]
            )

        elif lb == 3:
            # Add the MA node
            original_aif.add_component(
                "argument_relation", "MA", idents[i][1], idents[i][0]
            )

    return original_aif.xaif


def relation_identification(xaif, window_size):
    # Generate proposition pairs and matching "I" node ids for the output xAIF.
    data, ids, _ = preprocess_data(xaif["AIF"], window_size)

    if len(data["text"]) == 0:
        logger.info("Fewer than 2 I-nodes; skipping ARI classification")
        return xaif

    # Predict the list of labels for all the pairs of "I" nodes.
    labels = model_predictions(data)

    # Prepare the xAIF output file.
    out_xaif = output_xaif(ids, labels, xaif)

    return out_xaif


# DEBUGGING:
if __name__ == "__main__":
    ff = open("../data.json", "r")
    content = json.load(ff)
    # print(content)
    out = relation_identification(content, -1)
    with open("../data_out3.json", "w") as outfile:
        json.dump(out, outfile, indent=4)
