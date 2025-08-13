import random
import json
from pathlib import Path
from keyneuron import KeyNeuron
import torch

random.seed(42)

DATA_PATH = Path("data/domain_sample_multi_choice_qa.json")
DATA_SAMPLE = json.load(open(DATA_PATH))
print("[ sample number = {a}]".format(a=len(DATA_SAMPLE)))

KN = KeyNeuron(
    model_name = "/data/shenth/models/llama/2-7b-hf",
    data_samples = DATA_SAMPLE,
    result_dir = 'data/',
    common_threshold=0.3,
    top_v=20, 
    attr_threshold=0.2,
    option_letters = ["A", "B", "C", "D"],
    batch_size = 1,
    steps = 16,
    torch_dtype=torch.float16
)
# extract key neurons and store them in the result_dir
KN._extract_key_neuron()