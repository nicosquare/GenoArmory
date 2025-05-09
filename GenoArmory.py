import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Union, NamedTuple
from transformers import AutoModel, AutoTokenizer
import argparse
import sys
import json
from dataclasses import dataclass, field
from pathlib import Path
import datetime
import pandas as pd
import subprocess
import os


@dataclass
class AttackMetadata:
    """Metadata for attack methods"""

    method: str
    model_name: str
    description: str
    date_created: str = field(
        default_factory=lambda: datetime.datetime.now().isoformat()
    )
    total_sequences: int = 0
    success_rate: float = 0.0
    average_queries: float = 0.0
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DefenseMetadata:
    """Metadata for defense methods"""

    method: str
    model_name: str
    description: str
    date_created: str = field(
        default_factory=lambda: datetime.datetime.now().isoformat()
    )
    total_sequences: int = 0
    average_protection: float = 0.0
    average_robustness: float = 0.0
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AttackInfo:
    """Information about an attack attempt"""

    index: int
    method: str
    target_sequence: str
    modified_sequence: str
    target_label: Optional[int]
    success: bool
    number_of_queries: int
    queries_to_success: int
    confidence_original: float
    confidence_modified: float


@dataclass
class DefenseInfo:
    """Information about a defense attempt"""

    index: int
    method: str
    original_sequence: str
    protected_sequence: str
    protection_score: float
    computational_cost: float
    robustness_score: float


class AttackArtifact:
    """Container for attack artifacts"""

    def __init__(self, method: str, model_name: str, description: str = ""):
        self.metadata = AttackMetadata(
            method=method, model_name=model_name, description=description
        )
        self.attacks: List[AttackInfo] = []

    def add_attack(self, attack: AttackInfo):
        self.attacks.append(attack)
        self._update_metadata()

    def _update_metadata(self):
        """Update metadata statistics"""
        self.metadata.total_sequences = len(self.attacks)
        if self.attacks:
            self.metadata.success_rate = sum(a.success for a in self.attacks) / len(
                self.attacks
            )
            self.metadata.average_queries = sum(
                a.number_of_queries for a in self.attacks
            ) / len(self.attacks)

    def __getitem__(self, idx: int) -> AttackInfo:
        return self.attacks[idx]

    def __len__(self) -> int:
        return len(self.attacks)


class DefenseArtifact:
    """Container for defense artifacts"""

    def __init__(self, method: str, model_name: str, description: str = ""):
        self.metadata = DefenseMetadata(
            method=method, model_name=model_name, description=description
        )
        self.defenses: List[DefenseInfo] = []

    def add_defense(self, defense: DefenseInfo):
        self.defenses.append(defense)
        self._update_metadata()

    def _update_metadata(self):
        """Update metadata statistics"""
        self.metadata.total_sequences = len(self.defenses)
        if self.defenses:
            self.metadata.average_protection = sum(
                d.protection_score for d in self.defenses
            ) / len(self.defenses)
            self.metadata.average_robustness = sum(
                d.robustness_score for d in self.defenses
            ) / len(self.defenses)

    def __getitem__(self, idx: int) -> DefenseInfo:
        return self.defenses[idx]

    def __len__(self) -> int:
        return len(self.defenses)


class DNAModelConfig:
    """Configuration for DNA sequence model"""

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 32
        self.max_length = 512


class DNADefense:
    """Base class for DNA sequence defenses"""

    def __init__(self, target_model: "DNAModel", **kwargs):
        self.target_model = target_model
        self.config = kwargs

    def query(self, sequence: str, **kwargs) -> DefenseInfo:
        """
        Apply defense to a sequence and return defense info

        Args:
            sequence: Input DNA sequence
            **kwargs: Additional arguments for the defense method

        Returns:
            DefenseInfo object containing defense results
        """
        raise NotImplementedError("Subclasses must implement query method")


class ADFARDefense(DNADefense):
    """ADFAR (Adversarial Training with Fast Adaptation and Regularization) defense"""

    def __init__(self, target_model: "DNAModel", epsilon: float = 0.1, **kwargs):
        super().__init__(target_model, epsilon=epsilon, **kwargs)

    def query(self, sequence: str, **kwargs) -> DefenseInfo:
        """Apply ADFAR defense to a sequence"""
        armory = GenoArmory(
            self.target_model.model,
            self.target_model.tokenizer,
            device=self.target_model.config.device,
        )
        return armory.defend(
            sequence=sequence,
            defense_method="adfar",
            epsilon=self.config["epsilon"],
            **kwargs,
        )


class FreeLBDefense(DNADefense):
    """FreeLB (Free Large-Batch) defense"""

    def __init__(self, target_model: "DNAModel", batch_size: int = 32, **kwargs):
        super().__init__(target_model, batch_size=batch_size, **kwargs)

    def query(self, sequence: str, **kwargs) -> DefenseInfo:
        """Apply FreeLB defense to a sequence"""
        armory = GenoArmory(
            self.target_model.model,
            self.target_model.tokenizer,
            device=self.target_model.config.device,
        )
        return armory.defend(
            sequence=sequence,
            defense_method="freelb",
            batch_size=self.config["batch_size"],
            **kwargs,
        )


class DNAModel:
    """Interface for DNA sequence models"""

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.config = DNAModelConfig(
            model_name=model_name,
            api_key=api_key,
            device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
        )
        self.model = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model.to(self.config.device)

    def get_defense(self, defense_method: str, **kwargs) -> DNADefense:
        """
        Get a defense wrapper for this model

        Args:
            defense_method: One of ['adfar', 'freelb']
            **kwargs: Additional arguments for the defense method

        Returns:
            DNADefense object
        """
        defense_classes = {"adfar": ADFARDefense, "freelb": FreeLBDefense}

        if defense_method not in defense_classes:
            raise ValueError(
                f"Defense method must be one of {list(defense_classes.keys())}"
            )

        return defense_classes[defense_method](self, **kwargs)

    def query(
        self,
        sequences: List[str],
        attack_method: str,
        target_labels: Optional[List[int]] = None,
        **kwargs,
    ) -> List[AttackInfo]:
        """
        Query the model with multiple sequences for attack

        Args:
            sequences: List of DNA sequences to attack
            attack_method: Attack method to use
            target_labels: Optional list of target labels for each sequence
            **kwargs: Additional arguments for the attack method

        Returns:
            List of AttackInfo objects containing attack results
        """
        if target_labels is None:
            target_labels = [None] * len(sequences)
        elif len(target_labels) != len(sequences):
            raise ValueError("Number of target labels must match number of sequences")

        armory = GenoArmory(self.model, self.tokenizer, device=self.config.device)
        results = []

        for sequence, target in zip(sequences, target_labels):
            result = armory.attack(
                sequence=sequence,
                attack_method=attack_method,
                target_label=target,
                **kwargs,
            )
            results.append(result)

        return results


class GenoArmory:
    def __init__(
        self, model, tokenizer, device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize GenoArmory with a model and tokenizer

        Args:
            model: The DNABERT model to be used
            tokenizer: The tokenizer for the model
            device: Device to run the model on (cuda/cpu)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)

        # Store artifacts
        self._attack_artifacts: Dict[str, AttackArtifact] = {}
        self._defense_artifacts: Dict[str, DefenseArtifact] = {}

    @classmethod
    def from_pretrained(cls, model_path: str, device: Optional[str] = None):
        """Load GenoArmory from a pretrained model"""
        model = AutoModel.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return cls(
            model,
            tokenizer,
            device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
        )

    def read_attack_artifact(
        self, method: str, model_name: str, description: str = ""
    ) -> AttackArtifact:
        """Read attack artifacts for a specific method and model"""
        key = f"{method}_{model_name}"
        if key not in self._attack_artifacts:
            # Load artifact from storage or create new
            self._attack_artifacts[key] = AttackArtifact(
                method, model_name, description
            )
        return self._attack_artifacts[key]

    def read_defense_artifact(
        self, method: str, model_name: str, description: str = ""
    ) -> DefenseArtifact:
        """Read defense artifacts for a specific method and model"""
        key = f"{method}_{model_name}"
        if key not in self._defense_artifacts:
            # Load artifact from storage or create new
            self._defense_artifacts[key] = DefenseArtifact(
                method, model_name, description
            )
        return self._defense_artifacts[key]

    def get_attack_metadata(
        self, method: str, model_name: str
    ) -> Optional[AttackMetadata]:
        """Get metadata for a specific attack method and model"""
        key = f"{method}_{model_name}"
        if key in self._attack_artifacts:
            return self._attack_artifacts[key].metadata
        return None

    def get_defense_metadata(
        self, method: str, model_name: str
    ) -> Optional[DefenseMetadata]:
        """Get metadata for a specific defense method and model"""
        key = f"{method}_{model_name}"
        if key in self._defense_artifacts:
            return self._defense_artifacts[key].metadata
        return None

    def visualization(
        self,
        sequences: List[str],
        attention_layer: int = -1,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Visualize attention patterns for DNA sequences

        Args:
            sequences: List of DNA sequences to visualize
            attention_layer: Which attention layer to visualize
            save_path: Path to save the visualization
        """
        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenizer(sequences, return_tensors="pt", padding=True).to(
                self.device
            )
            outputs = self.model(**inputs, output_attentions=True)
            attention = outputs.attentions[attention_layer]

            # Create attention heatmap
            plt.figure(figsize=(10, 8))
            plt.imshow(attention[0].mean(dim=0).cpu())
            plt.colorbar()
            if save_path:
                plt.savefig(save_path)
            plt.close()

    def attack(
        self,
        sequence: str,
        attack_method: str,
        target_label: Optional[int] = None,
        **kwargs,
    ) -> AttackInfo:
        """
        Perform attacks on DNA sequences

        Args:
            sequence: Input DNA sequence
            attack_method: One of ['bertattack', 'textfooler', 'pgd', 'fimba']
            target_label: Target label for attack
            **kwargs: Additional arguments for specific attack methods

        Returns:
            AttackInfo object containing attack results
        """
        attack_methods = {
            "bertattack": self._bert_attack,
            "textfooler": self._textfooler_attack,
            "pgd": self._pgd_attack,
            "fimba": self._fimba_attack,
        }

        if attack_method not in attack_methods:
            raise ValueError(
                f"Attack method must be one of {list(attack_methods.keys())}"
            )

        result = attack_methods[attack_method](sequence, target_label, **kwargs)

        # Create attack info
        attack_info = AttackInfo(
            index=len(
                self._attack_artifacts.get(
                    f"{attack_method}_{self.model.name_or_path}", []
                )
            ),
            method=attack_method,
            target_sequence=sequence,
            modified_sequence=result["modified_sequence"],
            target_label=target_label,
            success=result["success"],
            number_of_queries=result["num_queries"],
            queries_to_success=result["queries_to_success"],
            confidence_original=result["confidence_original"],
            confidence_modified=result["confidence_modified"],
        )

        # Add to artifacts
        artifact = self.read_attack_artifact(attack_method, self.model.name_or_path)
        artifact.add_attack(attack_info)

        return attack_info

    def defend(self, sequence: str, defense_method: str, **kwargs) -> DefenseInfo:
        """
        Apply defense methods to DNA sequences

        Args:
            sequence: Input DNA sequence
            defense_method: One of ['adfar', 'freelb']
            **kwargs: Additional arguments for specific defense methods

        Returns:
            DefenseInfo object containing defense results
        """
        defense_methods = {"adfar": self._adfar_defense, "freelb": self._freelb_defense}

        if defense_method not in defense_methods:
            raise ValueError(
                f"Defense method must be one of {list(defense_methods.keys())}"
            )

        result = defense_methods[defense_method](sequence, **kwargs)

        # Create defense info
        defense_info = DefenseInfo(
            index=len(
                self._defense_artifacts.get(
                    f"{defense_method}_{self.model.name_or_path}", []
                )
            ),
            method=defense_method,
            original_sequence=sequence,
            protected_sequence=result["protected_sequence"],
            protection_score=result["protection_score"],
            computational_cost=result["computational_cost"],
            robustness_score=result["robustness_score"],
        )

        # Add to artifacts
        artifact = self.read_defense_artifact(defense_method, self.model.name_or_path)
        artifact.add_defense(defense_info)

        return defense_info

    # Attack method implementations
    def _bert_attack(
        self, sequence: str, target_label: Optional[int], **kwargs
    ) -> Dict[str, Any]:
        """BERT-Attack implementation"""
        # Implementation will be added
        pass

    def _textfooler_attack(
        self, **kwargs
    ) -> Dict[str, Any]:
        """TextFooler attack implementation"""
        # Call the subprocess runner for TextFooler
        run_textfooler_attack_script(
            base_dir=kwargs["base_dir"],
            dataset_dirs=kwargs["dataset_dirs"],
            model=kwargs["model"],
            target_model_path_template=kwargs["target_model_path_template"],
            output_dir_base=kwargs["output_dir_base"],
            attack_script_path=kwargs.get("attack_script_path", "TextFooler/attack_classification_general.py")
        )
        # Return a dummy result (could be extended to parse output)
        return {
            "modified_sequence": None,
            "success": None,
            "num_queries": None,
            "queries_to_success": None,
            "confidence_original": None,
            "confidence_modified": None,
        }

    def _pgd_attack(self, **kwargs) -> Dict[str, Any]:
        """PGD attack implementation"""
        run_pgd_attack_script(
            tasks=kwargs["tasks"],
            model=kwargs["model"],
            data_base_dir=kwargs["data_base_dir"],
            model_base_dir=kwargs["model_base_dir"],
            output_base_dir=kwargs["output_base_dir"],
            tokenizer_base_dir=kwargs["tokenizer_base_dir"],
            test_script_path=kwargs.get("test_script_path", "PGD/test.py"),
            task_name=kwargs.get("task_name", "0"),
            num_label=kwargs.get("num_label", "2"),
            n_gpu=kwargs.get("n_gpu", "1"),
            max_seq_length=kwargs.get("max_seq_length", "256"),
            batch_size=kwargs.get("batch_size", "16"),
            model_type=kwargs.get("model_type", "nt")
        )
        return {
            "modified_sequence": None,
            "success": None,
            "num_queries": None,
            "queries_to_success": None,
            "confidence_original": None,
            "confidence_modified": None,
        }

    def _fimba_attack(
        self, sequence: str, target_label: Optional[int], **kwargs
    ) -> Dict[str, Any]:
        """FIMBA attack implementation"""
        # Implementation will be added
        pass

    # Defense method implementations
    def _adfar_defense(self, sequence: str, **kwargs) -> Dict[str, Any]:
        """ADFAR defense implementation"""
        # Implementation will be added
        pass

    def _freelb_defense(self, sequence: str, **kwargs) -> Dict[str, Any]:
        """FreeLB defense implementation"""
        # Implementation will be added
        pass


def run_textfooler_attack_script(
    base_dir,
    dataset_dirs,
    model,
    target_model_path_template,
    output_dir_base,
    targert_model='bert',
    max_seq_length=256,
    batch_size=128,
    num_label=2,
    attack_script_path="TextFooler/attack_classification_general.py",
    
):
    command_template = (
        f'python {attack_script_path} --dataset_path {{dataset_path}} '
        '--target_model {{target_model}} '
        '--target_model_path {{target_model_path}} '
        '--output_dir {{output_dir}} '
        '--max_seq_length {{max_seq_length}} --batch_size {{batch_size}} '
        '--counter_fitting_embeddings_path {{counter_fitting_embeddings_path}} '
        '--counter_fitting_cos_sim_path {{counter_fitting_cos_sim_path}} '
        '--USE_cache_path {{USE_cache_path}} '
        '--nclasses {{num_label}} --tokenizer_path /scratch/hlv8980/Attack_Benchmark/models/{model}/{{dataset_dir}}/origin'
    )

    for dataset_dir in dataset_dirs:
        dataset_path = os.path.join(base_dir, dataset_dir, 'five_percent/cat.csv')
        target_model_path = target_model_path_template.format(model=model, dataset_dir=dataset_dir)
        output_dir = os.path.join(output_dir_base, model, dataset_dir)
        counter_fitting_embeddings_path="/projects/p32013/DNABERT-meta/TextFooler/embeddings/subword_{model}_embeddings.txt"
        counter_fitting_cos_sim_path="/projects/p32013/DNABERT-meta/TextFooler/cos_sim_counter_fitting/cos_sim_counter_fitting_{model}.npy"
        USE_cache_path="/projects/p32013/DNABERT-meta/TextFooler/tf_cache"
        tokenizer_path="/scratch/hlv8980/Attack_Benchmark/models/{model}/{{dataset_dir}}/origin"
        if os.path.exists(dataset_path):
            print(f"Dataset file found: {dataset_path}")
            command = command_template.format(
                dataset_path=dataset_path,
                dataset_dir=dataset_dir,
                target_model_path=target_model_path,
                model=model,
                output_dir=output_dir,
                counter_fitting_embeddings_path=counter_fitting_embeddings_path,
                counter_fitting_cos_sim_path=counter_fitting_cos_sim_path,
                USE_cache_path=USE_cache_path,
                tokenizer_path=tokenizer_path,
                max_seq_length=max_seq_length,
                batch_size=batch_size,
                num_label=num_label,
            )
            subprocess.run(command, shell=True, check=True)
        else:
            print(f"Dataset file not found: {dataset_path}")


def run_pgd_attack_script(
    tasks,
    model,
    data_base_dir,
    model_base_dir,
    output_base_dir,
    tokenizer_base_dir,
    test_script_path="PGD/test.py",
    task_name="0",
    num_label=2,
    n_gpu=1,
    max_seq_length=256,
    batch_size=16,
    model_type="nt"
):
    for task in tasks:
        data_dir = os.path.join(data_base_dir, task)
        model_name_or_path = os.path.join(model_base_dir, model, task)
        output_dir = os.path.join(output_base_dir, model, task)
        tokenizer_name = os.path.join(tokenizer_base_dir, model, task)
        command = [
            "python", test_script_path,
            "--data_dir", data_dir,
            "--model_name_or_path", model_name_or_path,
            "--task_name", str(task_name),
            "--num_label", num_label,
            "--n_gpu", n_gpu,
            "--max_seq_length", max_seq_length,
            "--batch_size", batch_size,
            "--output_dir", output_dir,
            "--model_type", str(model_type),
            "--overwrite_cache",
            "--tokenizer_name", tokenizer_name
        ]
        print("Running:", " ".join(command))
        subprocess.run(command, check=True)
        print(f"{task} finished")


def main():
    parser = argparse.ArgumentParser(
        description="GenoArmory: DNA Sequence Attack and Defense Tool"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Visualization command
    vis_parser = subparsers.add_parser("visualize", help="Visualize attention patterns")
    vis_parser.add_argument(
        "--sequences", nargs="+", required=True, help="List of DNA sequences"
    )
    vis_parser.add_argument(
        "--layer", type=int, default=-1, help="Attention layer to visualize"
    )
    vis_parser.add_argument("--save-path", type=str, help="Path to save visualization")

    # Read artifacts command
    read_parser = subparsers.add_parser("read", help="Read attack or defense artifacts")
    read_parser.add_argument("--type", choices=["attack", "defense"], required=True)
    read_parser.add_argument("--method", required=True, help="Method name")
    read_parser.add_argument("--model", required=True, help="Model name")
    read_parser.add_argument("--index", type=int, help="Artifact index (optional)")
    read_parser.add_argument(
        "--metadata", action="store_true", help="Show metadata only"
    )

    # Query command for batch attacks
    query_parser = subparsers.add_parser(
        "query", help="Query model with multiple sequences"
    )
    query_parser.add_argument(
        "--sequences", nargs="+", required=True, help="List of DNA sequences"
    )
    query_parser.add_argument(
        "--method", required=True, choices=["bertattack", "textfooler", "pgd", "fimba"]
    )
    query_parser.add_argument(
        "--target-labels", nargs="+", type=int, help="Target labels (optional)"
    )
    query_parser.add_argument(
        "--model-name", required=True, help="Name or path of the model"
    )
    query_parser.add_argument("--api-key", help="API key for the model (if needed)")
    query_parser.add_argument(
        "--params", type=str, help="Additional parameters in JSON format"
    )

    # Attack command
    attack_parser = subparsers.add_parser("attack", help="Perform an attack")
    attack_parser.add_argument("--input-csv", required=True, help="Path to CSV file with DNA sequences (column: 'sequence', optional: 'target_label')")
    attack_parser.add_argument(
        "--method", required=True, choices=["bertattack", "textfooler", "pgd", "fimba"]
    )
    attack_parser.add_argument(
        "--params", type=str, help="Additional parameters in JSON format"
    )

    # Defense command
    defense_parser = subparsers.add_parser("defend", help="Apply a defense")
    defense_parser.add_argument("--input-csv", required=True, help="Path to CSV file with DNA sequences (column: 'sequence')")
    defense_parser.add_argument("--method", required=True, choices=["adfar", "freelb"])
    defense_parser.add_argument(
        "--params", type=str, help="Additional parameters in JSON format"
    )

    # Defense query command
    defense_query_parser = subparsers.add_parser(
        "defense-query", help="Query model with defense"
    )
    defense_query_parser.add_argument(
        "--sequence", required=True, help="Input DNA sequence"
    )
    defense_query_parser.add_argument(
        "--model-name", required=True, help="Name or path of the model"
    )
    defense_query_parser.add_argument(
        "--defense-method", required=True, choices=["adfar", "freelb"]
    )
    defense_query_parser.add_argument(
        "--api-key", help="API key for the model (if needed)"
    )
    defense_query_parser.add_argument(
        "--params", type=str, help="Additional parameters in JSON format"
    )

    # Model loading arguments
    parser.add_argument("--model-path", required=True, help="Path to the DNABERT model")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the model on (cuda/cpu)",
    )

    args = parser.parse_args()

    # Initialize GenoArmory
    armory = GenoArmory.from_pretrained(args.model_path, device=args.device)

    # Execute commands
    if args.command == "visualize":
        armory.visualization(
            sequences=args.sequences,
            attention_layer=args.layer,
            save_path=args.save_path,
        )

    elif args.command == "read":
        if args.type == "attack":
            artifact = armory.read_attack_artifact(args.method, args.model)
            if args.metadata:
                print(artifact.metadata)
            elif args.index is not None:
                print(artifact[args.index])
            else:
                print(f"Total attacks: {len(artifact)}")
                print(f"Metadata: {artifact.metadata}")
        else:
            artifact = armory.read_defense_artifact(args.method, args.model)
            if args.metadata:
                print(artifact.metadata)
            elif args.index is not None:
                print(artifact[args.index])
            else:
                print(f"Total defenses: {len(artifact)}")
                print(f"Metadata: {artifact.metadata}")

    elif args.command == "query":
        model = DNAModel(args.model_name, api_key=args.api_key)
        kwargs = json.loads(args.params) if args.params else {}
        results = model.query(
            sequences=args.sequences,
            attack_method=args.method,
            target_labels=args.target_labels,
            **kwargs,
        )
        for result in results:
            print(result)

    elif args.command == "attack":
        kwargs = json.loads(args.params) if args.params else {}
        df = pd.read_csv(args.input_csv)
        sequences = df['sequence'].tolist()
        target_labels = df['target_label'].tolist() if 'target_label' in df.columns else [None] * len(sequences)
        results = []
        for seq, label in zip(sequences, target_labels):
            result = armory.attack(
                sequence=seq,
                attack_method=args.method,
                target_label=label,
                **kwargs,
            )
            results.append(result)
            print(result)

    elif args.command == "defend":
        kwargs = json.loads(args.params) if args.params else {}
        df = pd.read_csv(args.input_csv)
        sequences = df['sequence'].tolist()
        results = []
        for seq in sequences:
            result = armory.defend(
                sequence=seq, defense_method=args.method, **kwargs
            )
            results.append(result)
            print(result)

    elif args.command == "defense-query":
        model = DNAModel(args.model_name, api_key=args.api_key)
        kwargs = json.loads(args.params) if args.params else {}

        # Get defense wrapper
        defense = model.get_defense(args.defense_method, **kwargs)

        # Apply defense
        result = defense.query(args.sequence)
        print(result)


if __name__ == "__main__":
    main()
