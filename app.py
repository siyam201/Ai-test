from flask import Flask, request, jsonify, render_template
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

# **HTML ফাইল রেন্ডার করা (ফ্রন্টএন্ড)**
@app.route("/")
def home():
    return """<!DOCTYPE html>
<html lang="bn">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Chatbot</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; }
        #chat-box { width: 60%; margin: auto; border: 1px solid black; padding: 10px; height: 400px; overflow-y: scroll; }
        input { width: 60%; padding: 10px; }
        button { padding: 10px; }
    </style>
</head>
<body>
    <h1>AI Chatbot</h1>
    <div id="chat-box"></div>
    <input type="text" id="user-input" placeholder="একটি প্রশ্ন লিখুন...">
    <button onclick="sendMessage()">পাঠান</button>

    <script>
        async function sendMessage() {
            let userInput = document.getElementById("user-input").value;
            document.getElementById("chat-box").innerHTML += `<p><b>আপনি:</b> ${userInput}</p>`;

            let response = await fetch("/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: userInput })
            });

            let data = await response.json();
            document.getElementById("chat-box").innerHTML += `<p><b>চ্যাটবট:</b> ${data.response}</p>`;
        }
    </script>
</body>
</html>"""  

# **API রুট (চ্যাটবট)**
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message", "")
    bot_response = chatbot(user_input)
    return jsonify({"response": bot_response})

if __name__ == "__main__":
    app.run(debug=True)
