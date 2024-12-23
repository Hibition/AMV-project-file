# AMV-project-file

Joke Generation Model

This repository hosts a Joke Generation Model built using PyTorch, designed to generate humorous text with multiple modeling strategies. The project explores different approaches to optimize performance and generate jokes with varying styles and complexity.

Project Overview

The project contains multiple submodules, each representing a different modeling strategy:

1. Optimized LSTM

Implements a customized and optimized LSTM model for sequential joke generation.

Focuses on capturing context and generating coherent punchlines through effective sequence modeling.

2. T5-Based Text-to-Text Strategy

Utilizes the T5 model (Text-to-Text Transfer Transformer) to frame joke generation as a text-to-text task.

Leverages pre-trained T5 capabilities and fine-tunes it on joke datasets for enhanced humor generation.

3. LLaMA Fine-Tuned LLM

Adopts the LLaMA large language model (LLM) for fine-tuned joke generation.

Incorporates advanced fine-tuning techniques to produce context-aware and humor-rich text.

4. GPT-2 Integration

Includes experiments and implementation of GPT-2 for generating jokes with pre-trained language modeling capabilities.

5. Joke Classifier

Contains a jokeClassifier.ipynb notebook to classify or evaluate jokes based on their humor levels.

Repository Structure

├── code/
│   ├── GPT2/             # Code for GPT-2 experiments
│   ├── LSTM/             # Code for Optimized LSTM
│   ├── Llama/            # Code for LLaMA fine-tuning
│   ├── T5/               # Code for T5-based strategy
│   └── jokeClassifier.ipynb  # Joke evaluation and classification
├── README.md             # Project documentation

Getting Started

Prerequisites

Python 3.8+

PyTorch 1.12+

Transformers Library

CUDA-enabled GPU (optional, for faster training)

Installation

Clone the repository:

git clone https://github.com/Hibition/AMV-project-file.git
cd AMV-project-file

Install dependencies:

pip install -r requirements.txt

Training

For Optimized LSTM:

python code/LSTM/train.py

For T5-Based Strategy:

python code/T5/train.py

For LLaMA Fine-Tuning:

python code/Llama/train.py

For GPT-2 Experiments:

python code/GPT2/train.py

Inference

Run the inference script for any branch to generate jokes:

python generate_joke.py --model <model_branch> --input "Your joke prompt here."

Example:

python generate_joke.py --model T5 --input "Why did the chicken cross the road?"

Dataset

The model was trained on the Short Jokes Dataset available on Kaggle. You can access the dataset at:
Short Jokes Dataset

Use the provided scripts in the data/ directory for preprocessing.

Pre-Trained Models

Pre-trained model weights for each branch are available at:Pre-Trained Models Download

Contributing

Contributions are welcome! Feel free to submit issues or pull requests to improve the project.

License

This project is licensed under the MIT License. See the LICENSE file for details.

Contact

For any questions or feedback, please reach out via GitHub Issues:Open an Issue

Enjoy generating jokes with our model!
