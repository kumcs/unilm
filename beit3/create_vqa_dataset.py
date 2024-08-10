from datasets import VQAv2Dataset
from transformers import XLMRobertaTokenizer

tokenizer = XLMRobertaTokenizer("beit3.spm")

VQAv2Dataset.make_dataset_index(
    data_path="/mnt/d/ai-research/vizwiz",
    tokenizer=tokenizer,
    annotation_data_path="/home/adcenter/projects/scripts",
)