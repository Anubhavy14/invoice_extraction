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
