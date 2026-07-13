import torch
import torch.nn as nn
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image
from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import time

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# =========================
# LOAD TRAINED MODEL
# =========================

model = EfficientNet.from_name('efficientnet-b0')
model._fc = nn.Linear(model._fc.in_features, 2)

model.load_state_dict(
    torch.load("coral_efficientnet.pth", map_location="cpu")
)

model.eval()

classes = ["Bleached Coral", "Healthy Coral"]

# =========================
# IMAGE TRANSFORM
# =========================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# =========================
# HELPER FUNCTIONS
# =========================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_health_message(result, confidence):
    """Return contextual message based on prediction"""
    if result == "Healthy Coral":
        if confidence > 90:
            return "Excellent! This coral reef shows vibrant signs of life and biodiversity."
        else:
            return "This coral appears healthy, though continued monitoring is recommended."
    else:
        if confidence > 90:
            return "Critical: Significant coral bleaching detected. Immediate conservation action needed."
        else:
            return "Warning: Possible bleaching event. Further investigation recommended."

# =========================
# ROUTES
# =========================

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    filename = None
    health_message = None
    error = None

    if request.method == "POST":
        if 'image' not in request.files:
            error = "No file uploaded"
        else:
            file = request.files["image"]
            
            if file.filename == '':
                error = "No file selected"
            elif not allowed_file(file.filename):
                error = "Invalid file type. Please upload PNG, JPG, or JPEG images."
            else:
                try:
                    # Secure filename and save
                    timestamp = str(int(time.time()))
                    filename = secure_filename(f"{timestamp}_{file.filename}")
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)

                    # Process image
                    img = Image.open(filepath).convert("RGB")
                    img_tensor = transform(img).unsqueeze(0)

                    # Make prediction
                    with torch.no_grad():
                        outputs = model(img_tensor)
                        probs = torch.softmax(outputs, dim=1)
                        conf, pred = torch.max(probs, 1)

                    result = classes[pred.item()]
                    confidence = round(conf.item() * 100, 2)
                    health_message = get_health_message(result, confidence)
                    filename = f"uploads/{filename}"
                    
                except Exception as e:
                    error = f"Error processing image: {str(e)}"
                    if filename:
                        # Clean up file if processing failed
                        try:
                            os.remove(filepath)
                        except:
                            pass

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        filename=filename,
        health_message=health_message,
        error=error
    )

@app.route("/about")
def about():
    return render_template("about.html")

# =========================

if __name__ == "__main__":
    app.run(debug=True)
