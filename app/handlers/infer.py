from PIL import Image
from app.handlers.load_model import (
    device, processor, deplot_model,
    tokenizer, t5_model
)

def run_inference(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    deplot_ids = deplot_model.generate(
        flattened_patches=inputs.flattened_patches,
        attention_mask=inputs.attention_mask,
        max_length=1024
    )
    table_text = processor.batch_decode(deplot_ids, skip_special_tokens=True)[0].strip()

    tokenized = tokenizer.encode(table_text, return_tensors="pt").to(device)
    max_length = 512

    t5_generated_ids = t5_model.generate(
        tokenized,
        max_length=max_length,
        num_beams=4,
        repetition_penalty=5.0,
        length_penalty=1.0,
        early_stopping=True,
        temperature=0.6
    )
    description = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in t5_generated_ids]

    return {
        "generated_table": table_text,
        "generated_text": description
    }
