cd /projects/p32013/DNABERT-meta
# python GenoArmory.py --model_path magicslabnu/GERM visualize --folder_path /projects/p32013/DNABERT-meta/BERT-Attack/results/meta/test --save_path /projects/p32013/DNABERT-meta/BERT-Attack/results/meta/test/frequency.pdf
# python GenoArmory.py --model_path magicslabnu/GERM attack --method pgd --params_file /projects/p32013/DNABERT-meta/scripts/PGD/pgd_dnabert.json
# python GenoArmory.py --model_path magicslabnu/GERM attack --method fimba --params_file /projects/p32013/DNABERT-meta/scripts/FIMBA/fimba_dnabert.json
# python GenoArmory.py --model_path magicslabnu/GERM attack --method textfooler --params_file /projects/p32013/DNABERT-meta/scripts/TextFooler/textfooler_dnabert.json
# python GenoArmory.py --model_path magicslabnu/GERM attack --method bertattack --params_file /projects/p32013/DNABERT-meta/scripts/BertAttack/bertattack_dnabert.json


## Defense
python GenoArmory.py --model_path magicslabnu/GERM defense --method freelb --params_file /projects/p32013/DNABERT-meta/scripts/Freelb/freelb_dnabert.json
#python GenoArmory.py --model_path magicslabnu/GERM defense --method adfar --params_file 


