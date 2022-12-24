# coding: utf-8

from datasets import load_dataset
import jsonlines

from instruction_files.act_classification import instruction_dict

ds = load_dataset("diwank/silicone-merged", "balanced")
labels = [
    'acknowledge',
    'answer',
    'backchannel',
    'reply_yes',
    'exclaim',
    'say',
    'reply_no',
    'hold',
    'ask',
    'intent',
    'ask_yes_no',
]
all_options = "||||".join(labels)
instruction = ". ".join(instruction_dict["Definitions"]).strip()

def transform(record, split, index):

    label_idx = record["labels"]
    label_idx = label_idx[0] if isinstance(label_idx, list) else label_idx
    label = labels[label_idx]
    
    dialog = f"Person 1: {record['text_a']} [ENDOFTURN] Person 2: {record['text_b']}"
    context = f"[CONTEXT] {dialog} [ENDOFDIALOGUE]"
    options = f"[OPTIONS] {all_options}"
    question = f"[QUESTION] The simplified dialog act is"
    input = f"{context} {options} {question} "
    prompt = f"Instruction: {instruction}\nInput: {input}"
    
    return dict(
        output=label,
        all_outputs=[label],
        candidates=labels,
        classes_in_options=labels,
        dataset="silicone-merged",
        index=index,
        input=input,
        metadata={},
        prompt=prompt,
        split=split,
        task="act_classification",
        text="")
        
transformed = {split: [transform(record, split, i) for i, record in enumerate(ds[split])] for split in ds.keys()}

for split, data in transformed.items():
    with jsonlines.open(f"{split}.jsonl", 'w') as f:
        for item in data:
            f.write(item)
