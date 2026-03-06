import cv2
import time
import requests
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from ultralytics import YOLO

# ========================
# Telegram Configuration
# ========================
TOKEN = "xxxxxxxxxx"
CHAT_ID = "xxxx"

def send_telegram_report(pdf_path):
    url = f"https://api.telegram.org/bot{TOKEN}/sendDocument"
    with open(pdf_path, "rb") as file:
        requests.post(url,
                      data={"chat_id": CHAT_ID},
                      files={"document": file})
    print("Report sent to Telegram")

# ========================
# PDF Report Creation
# ========================
def create_pdf(image_path, helmet_count, vest_count, head_count, detections_list, missing_helmet, missing_vest):
    pdf_path = r"C:\Users\elbostan\PycharmProjects\pythonProject2\PPE_Safety_Report.pdf"
    c = canvas.Canvas(pdf_path, pagesize=letter)

    # Title
    c.setFont("Helvetica-Bold", 20)
    c.drawString(150, 750, "PPE Safety Detection Report")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.setFont("Helvetica", 12)
    c.drawString(400, 730, f"Time: {timestamp}")  # top right corner

    # Detection summary
    c.setFont("Helvetica", 12)
    y = 700
    c.drawString(50, y, f"Helmets detected: {helmet_count}")
    y -= 20
    c.drawString(50, y, f"Vests detected: {vest_count}")
    y -= 20
    c.drawString(50, y, f"Heads detected: {head_count}")
    y -= 40

    # Detailed detections
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Detected Objects")
    y -= 25
    c.setFont("Helvetica", 12)
    for label, conf in detections_list:
        c.drawString(60, y, f"{label}   (confidence: {conf:.2f})")
        y -= 18
    y -= 20

    # Warnings
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Safety Warnings")
    y -= 25
    c.setFont("Helvetica", 12)
    if missing_helmet > 0:
        c.drawString(60, y, f"WARNING: {missing_helmet} person(s) without helmet")
        y -= 20
    if missing_vest > 0:
        c.drawString(60, y, f"WARNING: {missing_vest} person(s) without safety vest")
        y -= 20
    if missing_helmet == 0 and missing_vest == 0:
        c.drawString(60, y, "All workers are wearing proper PPE")

    # Insert detection image
    c.drawImage(image_path, 100, 200, width=400, height=250)

    c.save()
    print("PDF report generated successfully!")
    return pdf_path

# ========================
# Load YOLO Model
# ========================
model = YOLO(r"C:\Users\elbostan\PycharmProjects\pythonProject2\best.pt")

# ========================
# Camera & Detection
# ========================
cap = cv2.VideoCapture(0)
last_alert_time = 0
alert_interval = 30

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame)

    classes = []
    detections_list = []

    for r in results:
        for box, conf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
            cls = int(cls)
            classes.append(cls)
            label = model.names[cls]
            detections_list.append((label, conf.item()))

    # PPE counting
    helmet_count = sum(1 for cls in classes if model.names[int(cls)] == "helmet")
    vest_count = sum(1 for cls in classes if model.names[int(cls)] == "vest")
    head_count = sum(1 for cls in classes if model.names[int(cls)] == "head")

    # Calculate missing PPE
    missing_helmet = max(0, head_count )
    missing_vest = max(0, (head_count+helmet_count) - vest_count)

    # Draw warnings on live camera
    if missing_helmet > 0:
        cv2.putText(frame, f"⚠ HELMET MISSING: {missing_helmet}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    if missing_vest > 0:
        cv2.putText(frame, f"⚠ VEST MISSING: {missing_vest}", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    if missing_helmet == 0 and missing_vest == 0:
        cv2.putText(frame, "✅ All PPE Worn", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    # Overlay YOLO annotations
    annotated_frame = results[0].plot()
    annotated_frame = cv2.addWeighted(frame, 0.7, annotated_frame, 0.3, 0)

    # Alert system with cooldown
    current_time = time.time()
    if (missing_helmet > 0 or missing_vest > 0) and (current_time - last_alert_time > alert_interval):
        print("🚨 PPE VIOLATION DETECTED")

        # Save alert image for PDF
        image_path = r"C:\Users\elbostan\PycharmProjects\pythonProject2\alarm2.jpg"
        cv2.imwrite(image_path, frame)

        # Create PDF report
        pdf_path = create_pdf(image_path, helmet_count, vest_count, head_count, detections_list, missing_helmet, missing_vest)

        # Send to Telegram
        send_telegram_report(pdf_path)

        last_alert_time = current_time

    # Show live camera with warnings
    cv2.imshow("PPE Detection", annotated_frame)


    if cv2.waitKey(1) & 0xFF == 'q':
        break

cap.release()

cv2.destroyAllWindows()
