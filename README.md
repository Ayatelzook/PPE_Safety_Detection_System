# PPE_Safety_Detection_System
Automated Helmet &amp; Vest Detection using YOLOv8n with Real-Time PDF Reports &amp;  Telegram Alerts
# PPE Safety Detection System 🦺

## Overview
The **PPE Safety Detection System** is an AI-based project designed to improve workplace safety in environments such as construction sites and industrial facilities.  

The system uses a **YOLOv8 deep learning model** to detect whether workers are wearing required **Personal Protective Equipment (PPE)** such as **helmets and safety vests**.

If a safety violation is detected, the system can:
- Generate a **PDF safety report**
- Capture the **detected image**
- Send a **Telegram alert notification**

---

## Features
- Real-time **PPE detection**
- Detection of **helmets and safety vests**
- **Automatic PDF report generation**
- **Telegram alert system**
- **Camera-based monitoring**
- Fast detection using **YOLOv8**

---

## Project Structure
PPE-Detection-System
│
├── train_ppe_model.py # Train the YOLOv8 model
├── ppe_report_generator.py # Detect PPE and generate PDF report
├── ppe_live_monitor.py # Camera detection + Telegram alert
│
├── PPE_Safety_Report_1/ # Generated PDF reports
│── PPE_Safety_Report_2/ # Generated PDF reports

---

## Technologies Used

- Python
- YOLOv8 (Ultralytics)
- OpenCV
- ReportLab (PDF generation)
- Telegram Bot API

---
