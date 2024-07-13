# keyneuron
This is the torch implementation of [Analyzing Key Neurons in Large Language Models](https://arxiv.org/pdf/2406.10868).

Functions
* extract key neurons ✅
* edit knowledge ❌
* show code examples ❌

# install
```
pip install -i https://test.pypi.org/simple/ keyneurons==0.0.2
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
    model_name = "EleutherAI/gpt-neo-125M",
    data_samples = DATA_SAMPLE,
    result_dir = 'data/',
    common_threshold=0.7,
    top_v=5, 
    attr_threshold=0.3,
    option_letters = ["A", "B", "C", "D"],
)
# extract key neurons and store them in the result_dir
KN._extract_key_neuron()



```
After, you can find the key neuron file in `data/`
