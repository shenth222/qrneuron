# keyneuron
This is the torch implementation of [Analyzing Key Neurons in Large Language Models](https://arxiv.org/pdf/2406.10868). <br>
The code of computing neuron attribution is adapted from [EleutherAI](https://github.com/EleutherAI/knowledge-neurons)， and we thank them for their effort and contribution.

Functions
* extract key neurons ✅
* edit knowledge ❌
* code examples ❌
* test cases ❌

# install
```
pip install -i https://test.pypi.org/simple/ keyneuron==0.0.2
```

# example

```python
import random
import json
from pathlib import Path
from keyneuron import KeyNeuron

random.seed(42)

DATA_PATH = Path("data/domain_sample_multi_choice_qa.json")
DATA_SAMPLE = json.load(open(DATA_PATH))
print("[ sample number = {a}]".format(a=len(DATA_SAMPLE)))

KN = KeyNeuron(
    model_name = "meta-llama/Llama-2-7b-chat-hf",
    data_samples = DATA_SAMPLE,
    result_dir = 'data/',
    common_threshold=0.7,
    top_v=5, 
    attr_threshold=0.3,
    option_letters = ["A", "B", "C", "D"],
    batch_size = 4,
    steps = 20,
)
# extract key neurons and store them in the result_dir
KN._extract_key_neuron()



```
After, you can find the key neuron file in `data/`

## Citation
If you find our paper and code useful, please give us a citation :blush:
```bibtex
@article{chen2024analyzing,
  title={Analyzing Key Neurons in Large Language Models},
  author={Chen, Lihu and Dejl, Adam and Toni, Francesca},
  journal={arXiv preprint arXiv:2406.10868},
  year={2024}
}
```

