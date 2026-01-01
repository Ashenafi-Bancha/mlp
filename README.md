# Multilayer Perceptron (MLP) → Feedforward network with backpropagation 

This project implements a Multilayer Perceptron using Python and NumPy only.
The network is trained using feedforward propagation and backpropagation
to solve the XOR problem.

---

## 🎯 Objectives
The main objectives of this project are:
- To understand how a Multilayer Perceptron works internally
- To implement feedforward propagation using matrix operations
- To implement backpropagation using gradient descent and the chain rule
- To demonstrate learning on a non-linearly separable problem (XOR)
- To provide a clear and explainable implementation suitable for an AI course assignment

---

## Features
- NumPy-only implementation
- Sigmoid activation
- Manual backpropagation
- XOR dataset


---

## 📂 Project Structure
```bash
mlp/
│
├── mlp_.py # MLP implementation using NumPy
├── README.md # Project documentation
├── requirements.txt # Project dependencies
└── .gitignore # Files ignored by Git
```

## Group Members

| Name            | ID          |
|-----------------|-------------|
| Ashenafi Bancha | UGR/1796/15 |
| Elham Jemal     | UGR/1757/14 |
| Feruza Hassen   | UGR/6423/15 |
| Ihsan Jemal     | UGR/9433/15 |

---

##  Installation and Setup

### 1️. Clone the Repository
```bash
git clone https://github.com/Ashenafi-Bancha/mlp.git
cd mlp
```
### 2️. Create a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
```
### 3️. Activate the Virtual Environment
- On Windows:
```bash
venv\Scripts\activate
```

- On macOS/Linux:
```bash
source venv/bin/activate
```

```bash
## How to Run
```bash
pip install -r requirements.txt
python mlp.py

```
