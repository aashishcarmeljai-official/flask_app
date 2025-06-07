from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from uuid import uuid4

db = SQLAlchemy()

def generate_uuid():
    return str(uuid4())

class Machine(db.Model):
    id = db.Column(db.String, primary_key=True, default=generate_uuid)
    name = db.Column(db.String, nullable=False)
    pdf_filename = db.Column(db.String)
    video_filename = db.Column(db.String)
    sop_filename = db.Column(db.String)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    screenshots = db.relationship('Screenshot', backref='machine', lazy=True)

class Screenshot(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    machine_id = db.Column(db.String, db.ForeignKey('machine.id'), nullable=False)
    filename = db.Column(db.String)
    caption = db.Column(db.String)