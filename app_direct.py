from transformers import (
    AutoModelForSequenceClassification,
    AutoProcessor,
)
from PIL import Image
from urllib.request import urlopen


model = AutoModelForSequenceClassification.from_pretrained("fedihch/InvoiceReceiptClassifier")
processor = AutoProcessor.from_pretrained("fedihch/InvoiceReceiptClassifier")

input_img_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0b/ReceiptSwiss.jpg/1024px-ReceiptSwiss.jpg"
with urlopen(input_img_url) as testImage:
    input_img = Image.open(testImage).convert("RGB")
    
encoded_inputs = processor(input_img, padding="max_length", return_tensors="pt")
outputs = model(**encoded_inputs)
logits = outputs.logits
predicted_class_idx = logits.argmax(-1).item()
id2label = {0: "invoice", 1: "receipt"}
print(id2label[predicted_class_idx])