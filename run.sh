cd /projects/p32013/DNABERT-meta
# python GenoArmory.py --model-path magicslabnu/GERM visualize --folder_path /projects/p32013/DNABERT-meta/BERT-Attack/results/meta/test --save_path /projects/p32013/DNABERT-meta/BERT-Attack/results/meta/test/frequency.pdf
python GenoArmory.py --model-path magicslabnu/GERM attack --method pgd --params_file /projects/p32013/DNABERT-meta/scripts/pgd.json