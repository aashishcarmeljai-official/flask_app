from flask import Flask, render_template, request, redirect, url_for, send_from_directory, make_response, jsonify
import os
import fitz
import cv2
from uuid import uuid4
from openai import OpenAI
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import pdfkit

# --- BLIP Model Init ---
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# --- Flask Setup ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SOP_FOLDER'] = 'sops'
openai_api_key = os.environ.get('OPENAI_API_KEY')
client = OpenAI(api_key=openai_api_key)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['SOP_FOLDER'], exist_ok=True)

machines = {}

# --- Utility Functions ---
def extract_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join([page.get_text() for page in doc])

def extract_video_screenshots(video_path, output_dir, num_frames=3):
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
    return render_template("index.html", machines=machines)

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/add', methods=["GET", "POST"])
def add_machine():
    if request.method == "POST":
        machine_id = str(uuid4())
        name = request.form['machine_name']
        pdf = request.files['pdf_file']
        video = request.files['video_file']

        pdf_filename = f"{machine_id}_pdf.pdf"
        video_filename = f"{machine_id}_video.mp4"
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)

        pdf.save(pdf_path)
        video.save(video_path)

        machines[machine_id] = {
            "name": name,
            "pdf": pdf_filename,
            "video": video_filename
        }

        return redirect(url_for('generate_sop', machine_id=machine_id))
    return render_template("add_machine.html")

@app.route('/generate_sop/<machine_id>', methods=["GET", "POST"])
def generate_sop(machine_id):
    if request.method == "POST":
        sop_html = request.form["sop"]
        filepath = os.path.join(app.config['SOP_FOLDER'], f"{machine_id}.html")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(sop_html)
        machines[machine_id]["sop"] = f"{machine_id}.html"
        return redirect(url_for("home"))

    machine = machines[machine_id]
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], machine["pdf"])
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], machine["video"])

    pdf_text = extract_pdf_text(pdf_path)

    screenshot_dir = os.path.join(app.config['UPLOAD_FOLDER'], f"{machine_id}_screens")
    screenshot_paths = [
        os.path.join(screenshot_dir, f)
        for f in extract_video_screenshots(video_path, screenshot_dir)
    ]

    captions = generate_clip_captions(screenshot_paths)
    captions_text = "\n".join([f"{fname}: {desc}" for fname, desc in captions])

    prompt = f"""
You are an expert technical writer. Create a detailed, professional Standard Operating Procedure (SOP)
for a machine using the following datasheet and captions from the SOP video.

Datasheet:
{pdf_text}

Visual Observations from Video Frames:
{captions_text}
"""
    print(prompt)
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You generate detailed machine SOPs."},
            {"role": "user", "content": prompt}
        ]
    )
    sop_text = response.choices[0].message.content

    return render_template("generate_sop.html",
                           machine_id=machine_id,
                           sop_text=sop_text,
                           screenshots=[f for f, _ in captions],
                           screenshot_dir=f"{machine_id}_screens")

@app.route('/machine/<machine_id>')
def machine_detail(machine_id):
    machine = machines.get(machine_id)
    if not machine:
        return "Machine not found", 404

    sop_path = os.path.join(app.config['SOP_FOLDER'], machine.get('sop', ''))
    sop_html = ""
    if os.path.exists(sop_path):
        with open(sop_path, "r", encoding="utf-8") as f:
            sop_html = f.read()

    return render_template("machine_detail.html", machine=machine, sop_html=sop_html)

@app.route('/export/pdf/<machine_id>')
def export_pdf(machine_id):
    sop_path = os.path.join(app.config['SOP_FOLDER'], f"{machine_id}.html")
    if not os.path.exists(sop_path):
        return "SOP not found", 404
    with open(sop_path, "r", encoding="utf-8") as f:
        sop_html = f.read()

    pdf = pdfkit.from_string(sop_html, False, configuration=PDFKIT_CONFIG)
    response = make_response(pdf)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = f'attachment; filename={machine_id}.pdf'
    return response

@app.route('/export/docx/<machine_id>')
def export_docx(machine_id):
    sop_path = os.path.join(app.config['SOP_FOLDER'], f"{machine_id}.html")
    if not os.path.exists(sop_path):
        return "SOP not found", 404
    with open(sop_path, "r", encoding="utf-8") as f:
        sop_html = f.read()

    # Strip HTML and save to docx
    from bs4 import BeautifulSoup
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

    machine = machines.get(machine_id)
    if not machine:
        return jsonify({"response": "Machine not found."}), 404

    # Read the SOP HTML
    sop_path = os.path.join(app.config['SOP_FOLDER'], machine.get('sop', ''))
    sop_text = ""
    if os.path.exists(sop_path):
        with open(sop_path, "r", encoding="utf-8") as f:
            sop_text = f.read()

    # Read the PDF manual text
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], machine["pdf"])
    manual_text = extract_pdf_text(pdf_path)

    # Prepare the GPT prompt
    context = f"""
Machine Name: {machine['name']}

Manual:
{manual_text[:2000]}

SOP:
{sop_text[:2000]}
"""

    full_prompt = f"""You are a helpful assistant for troubleshooting and operation support.

Context:
{context}

User Question:
{user_prompt}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You assist users by analyzing SOPs and manuals."},
                {"role": "user", "content": full_prompt}
            ]
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        return jsonify({"response": f"Error contacting GPT: {str(e)}"}), 500

    return jsonify({"response": answer})

if __name__ == "__main__":
    import webbrowser
    webbrowser.open("http://127.0.0.1:5000")
    app.run(debug=True)