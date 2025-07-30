# Invoice Information Extraction System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellowgreen)

A scalable machine learning system for extracting structured data from invoices using **LayoutLMv3**. Designed for real-world document processing with 90%+ accuracy on key fields.

## Key Features

- Field Extraction: Invoice numbers, dates, line items (description, quantity, price)
- Scalable Architecture: Easily train new models for additional fields
- Multi-Format Support: Processes both PDFs and images (JPG/PNG)
- Production-Ready: Includes training and inference pipelines

##  Technical Approach

### 1. Model Architecture
- Base Model: Fine-tuned `LayoutLMv3` (pretrained on FUNSD dataset)
- Token Classification: Custom head for:
- ["O", "B-INVOICE_NUMBER", "I-INVOICE_NUMBER", "B-INVOICE_DATE", ...]
- - **OCR Integration**: Tesseract for text/box detection

### 2. Data Pipeline
```mermaid
graph TD
  A[PDF/Image] --> B(OCR Processing)
  B --> C[Text + Bounding Boxes]
  C --> D[LayoutLMv3 Encoding]
  D --> E[Field Prediction]
  E --> F{JSON Output}

Key Technical Decisions
Dynamic Padding: Handles variable-length invoices

Nested Field Handling: For line items (B-I tagging)

Error Resilience: Skip corrupt files during training


git clone https://github.com/Anubhavy14/invoice_extraction.git
cd invoice_extraction

# Install dependencies
pip install -r requirements.txt

# Install Tesseract OCR (Windows)
choco install tesseract


Why This Approach?
Scalability: Add new fields by simply annotating samples and retraining

Accuracy: LayoutLMv3 outperforms traditional OCR + regex

Maintainability: Modular codebase with clear interfaces
