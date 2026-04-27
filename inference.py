import torch
from transformers import BertTokenizer, BertForSequenceClassification
import argparse

class FakeNewsClassifier:
    def __init__(self, model_path='fake_news_model'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Carrega o tokenizer e o modelo do diretório salvo
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text, max_length=512):
        # Prepara o input
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move para o dispositivo correto
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Faz a predição
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            probabilities = torch.softmax(outputs.logits, dim=1)
            _, predicted = torch.max(outputs.logits, 1)
            confidence = probabilities[0][predicted].item()
            
        return {
            "prediction": "REAL" if predicted.item() == 1 else "FAKE",
            "confidence": f"{confidence:.2%}"
        }

def main():
    parser = argparse.ArgumentParser(description='Classificador de Fake News')
    parser.add_argument('--text', type=str, help='Texto da notícia para classificar')
    parser.add_argument('--model_path', type=str, default='fake_news_model', help='Caminho para o diretório do modelo treinado')
    args = parser.parse_args()

    # Inicializa o classificador
    classifier = FakeNewsClassifier(args.model_path)
    
    # Faz a predição
    result = classifier.predict(args.text)
    
    print("\nResultado da Análise:")
    print(f"Classificação: {result['prediction']}")
    print(f"Confiança: {result['confidence']}")

if __name__ == '__main__':
    main() 