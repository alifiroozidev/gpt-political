 
from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

model = GPT2LMHeadModel.from_pretrained("../model")
tokenizer = GPT2Tokenizer.from_pretrained("../model")

@app.route('/generate', methods=['POST'])
def generate_text():
    prompt = request.json['prompt']
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(input_ids)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return jsonify({'answer': generated_text})

if __name__ == '__main__':
    app.run(port=5000)
