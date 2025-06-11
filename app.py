from flask import Flask, render_template, request, redirect, url_for, send_from_directory, make_response, jsonify
from models import db, Machine, Screenshot
import os
import fitz  # PyMuPDF
# import cv2
from uuid import uuid4
from openai import OpenAI
# from PIL import Image
# import torch
# from transformers import BlipProcessor, BlipForConditionalGeneration
import pdfkit
from docx import Document
from bs4 import BeautifulSoup

# --- BLIP Model Init ---
"""device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)"""

# --- Flask Setup ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SOP_FOLDER'] = 'sops'
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get("DATABASE_URL")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
with app.app_context():
    db.create_all()

openai_api_key = os.environ.get('OPENAI_API_KEY')
client = OpenAI(api_key=openai_api_key)

#PDFKIT_CONFIG = pdfkit.configuration(wkhtmltopdf='C:/Program Files/wkhtmltopdf/bin/wkhtmltopdf.exe')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['SOP_FOLDER'], exist_ok=True)

# --- Utility Functions ---
def extract_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join([page.get_text() for page in doc])

def extract_video_screenshots(video_path, output_dir, num_frames=10):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = total_frames // (num_frames + 1)
    screenshots = []
    for i in range(1, num_frames + 1):
        frame_no = i * interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        if ret:
            filename = f"screenshot_{i}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            screenshots.append(filename)
    cap.release()
    return screenshots

def generate_clip_captions(screenshot_paths):
    captions = []
    for path in screenshot_paths:
        image = Image.open(path).convert('RGB')
        inputs = processor(image, return_tensors="pt").to(device)
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        captions.append((os.path.basename(path), caption))
    return captions

# --- Routes ---
@app.route('/')
def home():
    machines = Machine.query.order_by(Machine.created_at.desc()).all()
    return render_template("index.html", machines=machines)

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/add', methods=["GET", "POST"])
def add_machine():
    if request.method == "POST":
        name = request.form['machine_name']
        pdf = request.files.get('pdf_file')
        video = request.files.get('video_file')

        if not pdf and not video:
            return "At least one of PDF or video is required.", 400

        machine = Machine(name=name)

        if pdf:
            pdf_filename = f"{uuid4()}_pdf.pdf"
            pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_filename)
            pdf.save(pdf_path)
            machine.pdf_filename = pdf_filename

        if video:
            video_filename = f"{uuid4()}_video.mp4"
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
            video.save(video_path)
            machine.video_filename = video_filename

        db.session.add(machine)
        db.session.commit()
        return redirect(url_for('generate_sop', machine_id=machine.id))

    return render_template("add_machine.html")

@app.route('/generate_sop/<machine_id>', methods=["GET", "POST"])
def generate_sop(machine_id):
    machine = Machine.query.get_or_404(machine_id)

    if request.method == "POST":
        sop_html = request.form["sop"]
        filename = f"{machine_id}.html"
        filepath = os.path.join(app.config['SOP_FOLDER'], filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(sop_html)
        machine.sop_filename = filename
        db.session.commit()
        return redirect(url_for("home"))

    pdf_text = ""
    if machine.pdf_filename:
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], machine.pdf_filename)
        pdf_text = extract_pdf_text(pdf_path)

    captions = []
    screenshot_relpaths = []
    if machine.video_filename:
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], machine.video_filename)
        screenshot_dir = os.path.join(app.config['UPLOAD_FOLDER'], f"{machine_id}_screens")
        screenshot_files = extract_video_screenshots(video_path, screenshot_dir)
        screenshot_paths = [os.path.join(screenshot_dir, f) for f in screenshot_files]
        captions = generate_clip_captions(screenshot_paths)
        screenshot_relpaths = [f"{machine_id}_screens/{fname}" for fname, _ in captions]

        for fname, cap in captions:
            screenshot = Screenshot(machine_id=machine.id, filename=fname, caption=cap)
            db.session.add(screenshot)
        db.session.commit()

    captions_text = "\n".join([f"{fname}: {desc}" for fname, desc in captions]) or "No video captions available."

    prompt = f"""You are an expert at writing clear and concise Standard Operating Procedures (SOPs).
Given the following datasheet or machine manual and if applicable video captions, generate an SOP in a structured format using numbered steps, clear headings, and bullet points where needed.
Make the SOP easy to read and follow. Include sections like 'Preparation', 'Operation Steps', 'Safety Precautions', and 'Shutdown Procedure' if applicable.
Create a Standard Operating Procedure in HTML format, using <h2>, <h3>, <ul>, <ol>, and <li> as appropriate.

Datasheet:
{pdf_text or "No datasheet provided."}

Visual Observations from Video Frames:
{captions_text}
"""

    response = client.chat.completions.create(
        model="gpt-4o", #gpt-3.5-turbo gpt-4o
        messages=[
            {"role": "system", "content": "You generate detailed machine SOPs."},
            {"role": "user", "content": prompt}
        ]
    )

    sop_text = response.choices[0].message.content

    return render_template("generate_sop.html",
                           machine_id=machine.id,
                           sop_text=sop_text,
                           screenshots=screenshot_relpaths)

@app.route('/machine/<machine_id>')
def machine_detail(machine_id):
    machine = Machine.query.get_or_404(machine_id)
    sop_html = ""
    if machine.sop_filename:
        sop_path = os.path.join(app.config['SOP_FOLDER'], machine.sop_filename)
        if os.path.exists(sop_path):
            with open(sop_path, "r", encoding="utf-8") as f:
                sop_html = f.read()
    return render_template("machine_detail.html", machine=machine, sop_html=sop_html)

@app.route('/export/pdf/<machine_id>')
def export_pdf(machine_id):
    machine = Machine.query.get_or_404(machine_id)
    if not machine.sop_filename:
        return "SOP not found", 404
    sop_path = os.path.join(app.config['SOP_FOLDER'], machine.sop_filename)
    with open(sop_path, "r", encoding="utf-8") as f:
        sop_html = f.read()
    pdf = pdfkit.from_string(sop_html, False, configuration=PDFKIT_CONFIG)
    response = make_response(pdf)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = f'attachment; filename={machine_id}.pdf'
    return response

@app.route('/export/docx/<machine_id>')
def export_docx(machine_id):
    machine = Machine.query.get_or_404(machine_id)
    sop_path = os.path.join(app.config['SOP_FOLDER'], machine.sop_filename or "")
    if not os.path.exists(sop_path):
        return "SOP not found", 404
    with open(sop_path, "r", encoding="utf-8") as f:
        sop_html = f.read()
    soup = BeautifulSoup(sop_html, "html.parser")
    text = soup.get_text()
    doc = Document()
    for line in text.splitlines():
        doc.add_paragraph(line)
    output_path = os.path.join(app.config['SOP_FOLDER'], f"{machine_id}.docx")
    doc.save(output_path)
    return send_from_directory(app.config['SOP_FOLDER'], f"{machine_id}.docx", as_attachment=True)

@app.route("/copilot/<machine_id>", methods=["POST"])
def copilot(machine_id):
    data = request.get_json()
    user_prompt = data.get("prompt", "")
    machine = Machine.query.get_or_404(machine_id)

    sop_text = ""
    if machine.sop_filename:
        sop_path = os.path.join(app.config['SOP_FOLDER'], machine.sop_filename)
        if os.path.exists(sop_path):
            with open(sop_path, "r", encoding="utf-8") as f:
                sop_text = f.read()

    manual_text = ""
    if machine.pdf_filename:
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], machine.pdf_filename)
        manual_text = extract_pdf_text(pdf_path)

    context = f"""
Machine Name: {machine.name}

Manual:
{manual_text[:2000] if manual_text else "Not provided"}

SOP:
{sop_text[:2000] if sop_text else "Not generated yet"}
"""

    full_prompt = f"""You are a helpful assistant for troubleshooting and operation support.

Context:
{context}

User Question:
{user_prompt}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4", #gpt-3.5-turbo gpt-4
            messages=[
                {"role": "system", "content": "You assist users by analyzing SOPs and manuals."},
                {"role": "user", "content": full_prompt}
            ]
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        return jsonify({"response": f"Error contacting GPT: {str(e)}"}), 500

    return jsonify({"response": answer})

@app.route('/delete/<machine_id>', methods=["POST"])
def delete_machine(machine_id):
    machine = Machine.query.get_or_404(machine_id)

    try:
        if machine.pdf_filename:
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], machine.pdf_filename))
        if machine.video_filename:
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], machine.video_filename))

        shot_dir = os.path.join(app.config['UPLOAD_FOLDER'], f"{machine_id}_screens")
        if os.path.exists(shot_dir):
            for f in os.listdir(shot_dir):
                os.remove(os.path.join(shot_dir, f))
            os.rmdir(shot_dir)

        if machine.sop_filename:
            os.remove(os.path.join(app.config['SOP_FOLDER'], machine.sop_filename))
            for ext in ['.docx', '.pdf']:
                path = os.path.join(app.config['SOP_FOLDER'], f"{machine_id}{ext}")
                if os.path.exists(path):
                    os.remove(path)

        for shot in machine.screenshots:
            db.session.delete(shot)

        db.session.delete(machine)
        db.session.commit()

    except Exception as e:
        return f"Error deleting files: {str(e)}", 500

    return redirect(url_for("home"))

@app.route('/init-db')
def init_db_route():
    from models import db  # Replace with your actual import if different
    db.create_all()
    return "Database initialized!"

if __name__ == "__main__":
    import webbrowser
    webbrowser.open("http://127.0.0.1:5000")
    app.run(debug=True)