cd /projects/p32013/DNABERT-meta/BERT-Attack
cat /projects/p32013/DNABERT-meta/BERT-Attack/scripts/all_hyena/emp.txt | python batch_run.py --gpus 0,0,0,0,0,0,0,0,0,0
cat /projects/p32013/DNABERT-meta/BERT-Attack/scripts/all_hyena/mouse.txt | python batch_run.py --gpus 0,0,0,0,0
cat /projects/p32013/DNABERT-meta/BERT-Attack/scripts/all_hyena/prom.txt | python batch_run.py --gpus 0,0,0,0,0,0
cat /projects/p32013/DNABERT-meta/BERT-Attack/scripts/all_hyena/tf.txt | python batch_run.py --gpus 0,0,0,0,0