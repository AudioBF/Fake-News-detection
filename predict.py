import torch
from transformers import BertTokenizer, BertForSequenceClassification
import argparse

def load_model(model_path):
    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=2
    )
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    return model, tokenizer

def predict(text, model, tokenizer, max_length=512):
    # Prepare input
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Make prediction
    with torch.no_grad():
        outputs = model(
            input_ids=encoding['input_ids'],
            attention_mask=encoding['attention_mask']
        )
        _, predicted = torch.max(outputs.logits, 1)
        
    return "REAL" if predicted.item() == 1 else "FAKE"

def main():
    parser = argparse.ArgumentParser(description='Predict if a news article is real or fake')
    parser.add_argument('--text', type=str, help='Text of the news article to classify')
    parser.add_argument('--model_path', type=str, default='best_model.pt', help='Path to the trained model')
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model(args.model_path)
    
    # Make prediction
    prediction = predict(args.text, model, tokenizer)
    print(f"\nPrediction: {prediction}")

if __name__ == '__main__':
    main() 