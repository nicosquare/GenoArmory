cd /projects/p32013/DNABERT-meta/BERT-Attack
cat /projects/p32013/DNABERT-meta/BERT-Attack/scripts/all_nt2/emp.txt | python batch_run.py --gpus 0,0,0,0,0,0,0,0,0,0
cat /projects/p32013/DNABERT-meta/BERT-Attack/scripts/all_nt2/mouse.txt | python batch_run.py --gpus 0,0,0,0,0
cat /projects/p32013/DNABERT-meta/BERT-Attack/scripts/all_nt2/prom.txt | python batch_run.py --gpus 0,0,0,0,0,0
cat /projects/p32013/DNABERT-meta/BERT-Attack/scripts/all_nt2/tf.txt | python batch_run.py --gpus 0,0,0,0,0
