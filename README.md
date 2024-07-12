# keyneurons


# example

```python
random.seed(42)

DATA_PATH = Path("data/domain_sample_multi_choice_qa.json")
DATA_SAMPLE = json.load(open(DATA_PATH))
print("[ sample number = {a}]".format(a=len(DATA_SAMPLE)))

KeyNeuron = NaicaKeyNeuron(
    model_name = "EleutherAI/gpt-neo-125M",
    data_samples = DATA_SAMPLE,
    result_dir = 'data/',
    common_threshold=0.7,
    top_v=5, 
    attr_threshold=0.3
)
KeyNeuron._extract_key_neuron()
```
