from flask import Flask, render_template, request
from transformers import pipeline, set_seed
import textwrap

app = Flask(__name__)

# Set up the text generation pipeline with GPT-2 model
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
set_seed(42)  # Optional: Set a seed for reproducibility


# text wrapping function
def wrap(x):
    return textwrap.fill(x, replace_whitespace=False, fix_sentence_endings=True)


@app.route("/")
def index():
    return render_template("traits.html")


@app.route("/summarize", methods=["POST"])
def summarize():
    if request.method == "POST":
        input_text = str(request.form["prompt"])

        doc = wrap(input_text)

        # Summarize text using Hugging Face summarization transformer
        results = summarizer(doc.split("\n", 1)[1])

        summary = results[0]["summary_text"]

        print(summary)

        return render_template("summary.html", summarized_text=summary)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
