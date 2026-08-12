import os
import sys
import joblib
from train import clean_text, train_and_evaluate

def load_pipeline():
    model_path = os.path.join(os.path.dirname(__file__), "sms_spam_pipeline.pkl")
    if not os.path.exists(model_path):
        print("[!] Model file not found. Running training pipeline to build model...")
        train_and_evaluate()
    return joblib.load(model_path)

def predict_sms(text, pipeline=None):
    if pipeline is None:
        pipeline = load_pipeline()
        
    cleaned = clean_text(text)
    prediction = pipeline.predict([cleaned])[0]
    
    label = "SPAM" if prediction == 1 else "HAM"
    
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba([cleaned])[0]
        confidence = probs[prediction] * 100
        return label, confidence
    return label, None

def main():
    pipeline = load_pipeline()
    
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        label, conf = predict_sms(text, pipeline)
        print("=" * 50)
        print(f"SMS Input : \"{text}\"")
        print(f"Prediction: {label}")
        if conf is not None:
            print(f"Confidence: {conf:.2f}%")
        print("=" * 50)
    else:
        print("\n--- SMS Spam Predictor (CLI) ---")
        print("Type an SMS message and press Enter (or 'quit' / 'exit' to stop):\n")
        
        sample_messages = [
            "Congratulations! You have won a $1,000 Walmart gift card. Click here to claim.",
            "Hey buddy, are we still meeting for lunch tomorrow at 1 PM?"
        ]
        
        print("[*] Running Sample Inference:")
        for sample in sample_messages:
            lbl, conf = predict_sms(sample, pipeline)
            conf_str = f" ({conf:.2f}%)" if conf else ""
            print(f"  Input : \"{sample}\"")
            print(f"  Result: {lbl}{conf_str}\n")
            
        while True:
            try:
                user_input = input("Enter SMS message: ").strip()
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                if not user_input:
                    continue
                lbl, conf = predict_sms(user_input, pipeline)
                conf_str = f" (Confidence: {conf:.2f}%)" if conf else ""
                print(f"--> Prediction: [{lbl}]{conf_str}\n")
            except (KeyboardInterrupt, EOFError):
                print("\nExiting.")
                break

if __name__ == "__main__":
    main()
