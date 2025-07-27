import torch
from transformers import AutoProcessor, Pix2StructForConditionalGeneration, T5Tokenizer, T5ForConditionalGeneration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = AutoProcessor.from_pretrained("brainventures/deplot_kr")
deplot_model = Pix2StructForConditionalGeneration.from_pretrained("brainventures/deplot_kr")
deplot_model.to(device).eval()

tokenizer = T5Tokenizer.from_pretrained("KETI-AIR/ke-t5-base")
t5_model = T5ForConditionalGeneration.from_pretrained("KETI-AIR/ke-t5-base")

t5_model.load_state_dict(torch.load("model/ke_t5.pt"))
t5_model.to(device).eval()
