cd /projects/p32013/DNABERT-meta

## Visualization
# python GenoArmory.py --model_path magicslabnu/GERM visualize --folder_path /projects/p32013/DNABERT-meta/BERT-Attack/results/meta/test --save_path /projects/p32013/DNABERT-meta/BERT-Attack/results/meta/test/frequency.pdf

## Attack
# python GenoArmory.py --model_path magicslabnu/GERM attack --method pgd --params_file /projects/p32013/DNABERT-meta/scripts/PGD/pgd_dnabert.json
# python GenoArmory.py --model_path magicslabnu/GERM attack --method fimba --params_file /projects/p32013/DNABERT-meta/scripts/FIMBA/fimba_dnabert.json
# python GenoArmory.py --model_path magicslabnu/GERM attack --method textfooler --params_file /projects/p32013/DNABERT-meta/scripts/TextFooler/textfooler_dnabert.json
# python GenoArmory.py --model_path magicslabnu/GERM attack --method bertattack --params_file /projects/p32013/DNABERT-meta/scripts/BertAttack/bertattack_dnabert.json


## Defense
python GenoArmory.py --model_path magicslabnu/GERM defense --method freelb --params_file /projects/p32013/DNABERT-meta/scripts/FreeLB/freelb_pgd_dnabert.json
# python GenoArmory.py --model_path magicslabnu/GERM defense --method adfar --params_file /projects/p32013/DNABERT-meta/scripts/ADFAR/adfar_pgd_dnabert.json
# python GenoArmory.py --model_path magicslabnu/GERM defense --method at --params_file /projects/p32013/DNABERT-meta/scripts/AT/at_pgd_dnabert.json

## Read Metadata
# python GenoArmory.py --model_path magicslabnu/GERM read --type attack --method TextFooler --model_name dnabert
# python GenoArmory.py --model_path magicslabnu/GERM read --type defense --method ADFAR --model_name dnabert --attack_method textfooler

