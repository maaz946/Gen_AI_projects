# AI Projects Showcase

Welcome to my collection of four AI-powered projects! These projects leverage state-of-the-art language and vision models to solve real-world problems, from creative writing to synthetic data generation. Built with Python and tools like Hugging Face Transformers, each project is a standalone solution with practical applications.

## Table of Contents
- [Projects](#projects)
  - [1. Text Generator for Creative Writing]
  - [2. Chatbot for FAQ Automation]
  - [3. Image Caption Generator]
  - [4. Synthetic Data Generator]
- [Setup](#setup)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Projects

### 1. Text Generator for Creative Writing
- **Problem Solved:** Overcoming writer’s block by generating quick story prompts or poetry.
- **What It Does:** Takes a starting sentence and outputs 50-100 words of creative text.
- **Model Used:** GPT-2 (Hugging Face Transformers)
- **Why GPT-2?** Lightweight, pre-trained, and balances creativity with coherence—ideal for rapid prototyping.
- **Time to Build:** ~3-5 days
- **Tools:** Python, Transformers, Google Colab

### 2. Chatbot for FAQ Automation
- **Problem Solved:** Automating responses to common queries (e.g., customer FAQs).
- **What It Does:** Answers predefined questions based on a fine-tuned dataset.
- **Model Used:** DistilBERT or DialoGPT
- **Why These Models?** DistilBERT is fast and efficient; DialoGPT excels in conversation. Fine-tuned on a small dataset for quick deployment.
- **Time to Build:** ~5-7 days
- **Tools:** Python, Transformers, PyTorch/TensorFlow, Colab

### 3. Image Caption Generator
- **Problem Solved:** Automatically generating captions for images (e.g., for accessibility or tagging).
- **What It Does:** Converts uploaded images into descriptive text.
- **Model Used:** CLIP-ViT (with captioning head)
- **Why CLIP-ViT?** Exceptional vision-language alignment makes it perfect for image-to-text tasks.
- **Time to Build:** ~7 days
- **Tools:** Python, Transformers, PIL/OpenCV, Colab or local GPU

### 4. Synthetic Data Generator
- **Problem Solved:** Creating synthetic datasets for rare ML scenarios.
- **What It Does:** Generates structured text (e.g., customer reviews) parsed into CSV format.
- **Model Used:** GPT-Neo or LLaMA
- **Why These Models?** Large-scale and flexible, they produce realistic, domain-specific data.
- **Time to Build:** ~7-10 days
- **Tools:** Python, Pandas, Transformers, Colab or local machine

## Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/maaz946/Gen_AI_projects.git
   cd Gen_AI_projects
