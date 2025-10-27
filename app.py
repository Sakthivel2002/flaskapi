from flask import Flask, request, jsonify
import fitz  # PyMuPDF
import nltk
import random
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Download NLTK tokenizer (for sentence splitting)
nltk.download("punkt", quiet=True)

app = Flask(__name__)

# ✅ Detect GPU automatically
device = 0 if torch.cuda.is_available() else -1
print(f"🚀 Using device: {'GPU' if device == 0 else 'CPU'}")

# ✅ Load Question Generation Model
MODEL_NAME = "valhalla/t5-base-qg-hl"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
qg_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device)

# ✅ Load Answer Generation Model
ans_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", device=device)


def extract_text_from_pdf(file_storage):
    """Extract readable text safely from uploaded PDF."""
    import io

    try:
        # Move pointer to start
        file_storage.seek(0)

        # Read into memory
        pdf_bytes = file_storage.read()
        if not pdf_bytes or len(pdf_bytes) < 500:
            raise ValueError("Invalid or empty PDF file")

        # Ensure file is opened from a byte stream
        pdf_stream = io.BytesIO(pdf_bytes)

        # Open PDF using fitz
        with fitz.open(stream=pdf_stream, filetype="pdf") as pdf:
            text = ""
            for page in pdf:
                text += page.get_text("text")

        if not text.strip():
            raise ValueError("No extractable text found in PDF (might be scanned).")

        print(f"✅ Extracted {len(text)} characters from PDF.")
        return text.strip()

    except Exception as e:
        print(f"❌ PDF extraction failed: {e}")
        return ""


def split_into_sentences(text):
    """Split text into meaningful sentences."""
    sentences = nltk.sent_tokenize(text)
    return [s.strip() for s in sentences if len(s.split()) > 5]


def polish_question(q):
    """Ensure better phrasing and grammar for generated questions."""
    q = q.strip()
    if not q.endswith("?"):
        q += "?"
    starts = ["What", "How", "Why", "Which", "When", "Where"]
    if not any(q.startswith(w) for w in starts):
        q = "What " + q[0].lower() + q[1:]
    return q


def assign_difficulty(question):
    """Assign difficulty label based on question length."""
    length = len(question.split())
    if length < 8:
        return "Easy"
    elif length < 14:
        return "Medium"
    else:
        return "Hard"


def generate_answer(question, context):
    """Generate a short answer using FLAN-T5."""
    try:
        prompt = f"Answer briefly: {question}\nContext: {context}"
        result = ans_pipeline(prompt, max_length=60, num_return_sequences=1)
        return result[0]['generated_text'].strip()
    except Exception:
        return "Answer unavailable."


def generate_questions(text, num_questions=5):
    """Generate intelligent questions + difficulty + answers."""
    sentences = split_into_sentences(text)
    if len(sentences) < num_questions:
        while len(sentences) < num_questions:
            sentences.append(random.choice(sentences))

    results = []
    for i, sent in enumerate(sentences[:num_questions]):
        prompt = f"generate question: {sent.strip()}"
        try:
            output = qg_pipeline(prompt, max_length=128, num_return_sequences=1)
            q = output[0]['generated_text'].strip()
            q = polish_question(q)
            difficulty = assign_difficulty(q)
            answer = generate_answer(q, sent)
            results.append({
                "question": q,
                "difficulty": difficulty,
                "answer": answer
            })
        except Exception as e:
            print(f"⚠️ Error generating question {i}: {e}")
            continue

    return results[:num_questions]


@app.route("/generate", methods=["POST"])
def generate():
    """Main route: PDF or text input -> JSON questions."""
    text = ""

    # ✅ PDF Upload Case
    if "file" in request.files and request.files["file"].filename:
        file = request.files["file"]
        text = extract_text_from_pdf(file)

    # ✅ Text Input Case
    elif request.form.get("text"):
        text = request.form.get("text", "").strip()

    if not text:
        print("❌ No valid text extracted from input.")
        return jsonify({"error": "No input text provided."}), 400

    num_qs = int(request.form.get("num_questions", 5))

    try:
        data = generate_questions(text, num_qs)
        return jsonify({"questions": data})
    except Exception as e:
        print(f"❌ Generation error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
