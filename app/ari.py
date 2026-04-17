import json
import logging
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer, pipeline
import numpy as np
from optimum.intel import OVModelForSequenceClassification, OVWeightQuantizationConfig
from xaif_eval import xaif

logger = logging.getLogger(__name__)

MODEL_ID = "raruidol/ArgumentMining-EN-ARI-AIF-RoBERTa_L"


def _load_config():
    config_path = Path(__file__).parent.parent / "config" / "config.json"
    if config_path.exists():
        with config_path.open() as f:
            return json.load(f)
    return {"model_path": MODEL_ID, "ov_model_path": None}


def _load_model():
    config = _load_config()
    model_path = config.get("model_path", MODEL_ID)
    ov_model_path = config.get("ov_model_path")

    if ov_model_path:
        logger.info("Loading pre-exported OpenVINO model from: %s", ov_model_path)
        tokenizer = AutoTokenizer.from_pretrained(ov_model_path)
        ov_model = OVModelForSequenceClassification.from_pretrained(
            ov_model_path, export=False, compile=True
        )
    else:
        logger.warning(
            "'ov_model_path' is not set in config/config.json. "
            "Falling back to exporting from PyTorch at runtime (slow). "
            "Run the export script and set 'ov_model_path' to avoid this."
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        quantization_config = OVWeightQuantizationConfig(bits=8, ratio=1.0)
        ov_model = OVModelForSequenceClassification.from_pretrained(
            model_path,
            export=True,
            compile=True,
            quantization_config=quantization_config,
        )

    return tokenizer, ov_model


TOKENIZER, PRUNED_MODEL = _load_model()


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
        return Dataset.from_dict(data), idents_comb, propositions
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

    final_data = Dataset.from_dict(data)

    return final_data, idents_comb, propositions


def tokenize_sequence(samples):
    return TOKENIZER(
        samples["text"], samples["text2"], padding="max_length", truncation=True
    )


def pipeline_predictions(pipeline, data):
    labels = []
    pipeline_input = []
    for i in range(len(data["text"])):
        sample = data["text"][i] + ". " + data["text2"][i]
        pipeline_input.append(sample)

    outputs = pipeline(pipeline_input, batch_size=32, truncation=True)
    for out in outputs:
        out = out[0] if isinstance(out, list) else out
        if out["label"] == "Inference" and out["score"] >= 0.9:
            labels.append(1)
        elif out["label"] == "Conflict" and out["score"] >= 0.75:
            labels.append(2)
        elif out["label"] == "Rephrase" and out["score"] >= 0.9:
            labels.append(3)
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
    # Generate a HF Dataset from all the "I" node pairs to make predictions from the xAIF file
    # and a list of tuples with the corresponding "I" node ids to generate the final xaif file.
    dataset, ids, _ = preprocess_data(xaif["AIF"], window_size)

    if len(dataset) == 0:
        logger.info("Fewer than 2 I-nodes; skipping ARI classification")
        return xaif

    # Inference Pipeline
    pl = pipeline("text-classification", model=PRUNED_MODEL, tokenizer=TOKENIZER)

    # Predict the list of labels for all the pairs of "I" nodes.
    labels = pipeline_predictions(pl, dataset)

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
