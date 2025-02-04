from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch  

app = Flask(__name__)

# GPT-2 মডেল লোড করা  
model_name = "gpt2"  
model = AutoModelForCausalLM.from_pretrained(model_name)  
tokenizer = AutoTokenizer.from_pretrained(model_name)  

# চ্যাটবট ফাংশন  
def chatbot(prompt):  
    inputs = tokenizer(prompt, return_tensors="pt")  
    outputs = model.generate(**inputs, max_length=100, pad_token_id=tokenizer.eos_token_id)  
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)  
    return response  

# API রুট  
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message", "")
    bot_response = chatbot(user_input)
    return jsonify({"response": bot_response})

if __name__ == "__main__":
    app.run(debug=True)
