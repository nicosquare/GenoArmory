<div align="center">
  <img src="asserts/logo.jpg" alt="Image" />
</div>


<p align="center">
    <p align="center">A comprehensive toolkit for DNA sequence Adversarial Attack and Defense Benchmark.
    <br>
</p>



<h4 align="center">
    <a href="" target="_blank">Paper</a> |
    <a href="https://huggingface.co/datasets/magicslabnu/GFM-Attack" target="_blank">Datasets</a> |
    <a href="https://huggingface.co/collections/magicslabnu/gfm-67f4d4a9327ee4acdcb3806b" target="_blank">Pretrained Checkpoints</a>
</h4>




## Installation

You can install GenoArmory using pip:

```bash
pip install genoarmory
```

## Quick Start

```python
# Initialize model
from genoarmory import DNAModel

model = DNAModel(model_name="dnabert-v1", api_key="your-api-key")

# Perform attack
sequences = ["ATCGGTCA", "GCTATAGC"]
responses = model.query(
    sequences=sequences,
    attack_method="bertattack"
)

# Apply defense
defense = model.get_defense(defense_method="adfar", epsilon=0.1)
response = defense.query(sequence="ATCGGTCA")
```

## Command Line Usage

GenoArmory can also be used from the command line:

```bash
# Attack
genoarmory attack --sequence "ATCGGTCA" --method bertattack

# Defense
genoarmory defend --sequence "ATCGGTCA" --method adfar

# Visualization
genoarmory visualize --sequences "ATCGGTCA" "GCTATAGC" --save-path "attention.png"
```

## Features

- Multiple attack methods:

  - BERT-Attack
  - TextFooler
  - PGD
  - FIMBA

- Defense methods:

  - ADFAR
  - FreeLB
  - Traditional Adversarial Training

- Visualization tools
- Artifact management
- Batch processing
- Command-line interface

## Documentation

For detailed documentation, visit [docs](We will release soon).

## License

This project is licensed under the MIT License.

## Citation

If you have any question regarding our paper or codes, please feel free to start an issue.

If you use GenoArmory in your work, please kindly cite our paper:

```

```
