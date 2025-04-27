cd /projects/p32013/DNABERT-meta/BERT-Attack
cat /projects/p32013/DNABERT-meta/BERT-Attack/scripts/nt1_quant/emp3.txt | python batch_run.py --gpus 0,0,0,0,0
cat /projects/p32013/DNABERT-meta/BERT-Attack/scripts/nt1_quant/prom.txt | python batch_run.py --gpus 0,0,0
cat /projects/p32013/DNABERT-meta/BERT-Attack/scripts/nt1_quant/prom1.txt | python batch_run.py --gpus 0,0,0
cat /projects/p32013/DNABERT-meta/BERT-Attack/scripts/nt1_quant/tf.txt | python batch_run.py --gpus 0,0,0,0,0
cat /projects/p32013/DNABERT-meta/BERT-Attack/scripts/nt1_quant/emp.txt | python batch_run.py --gpus 0,0
cat /projects/p32013/DNABERT-meta/BERT-Attack/scripts/nt1_quant/mouse.txt | python batch_run.py --gpus 0
