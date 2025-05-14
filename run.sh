cd /projects/p32013/DNABERT-meta
# python GenoArmory.py --model_path magicslabnu/GERM visualize --folder_path /projects/p32013/DNABERT-meta/BERT-Attack/results/meta/test --save_path /projects/p32013/DNABERT-meta/BERT-Attack/results/meta/test/frequency.pdf
# python GenoArmory.py --model_path magicslabnu/GERM attack --method pgd --params_file /projects/p32013/DNABERT-meta/scripts/PGD/pgd_dnabert.json
#python GenoArmory.py --model_path magicslabnu/GERM attack --method fimba --params_file /projects/p32013/DNABERT-meta/scripts/FIMBA/fimba_dnabert.json
python GenoArmory.py --model_path magicslabnu/GERM attack --method textfooler --params_file /projects/p32013/DNABERT-meta/scripts/TextFooler/textfooler_dnabert.json
