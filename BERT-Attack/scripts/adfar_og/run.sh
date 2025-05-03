cd /projects/p32013/DNABERT-meta/BERT-Attack
cat /projects/p32013/DNABERT-meta/BERT-Attack/scripts/adfar_og/emp.txt | python batch_run.py --gpus 0,0,0,0,0,0,0,0,0,0
cat /projects/p32013/DNABERT-meta/BERT-Attack/scripts/adfar_og/mouse.txt | python batch_run.py --gpus 0,0,0,0,0
cat /projects/p32013/DNABERT-meta/BERT-Attack/scripts/adfar_og/prom.txt | python batch_run.py --gpus 0,0,0,0,0,0
cat /projects/p32013/DNABERT-meta/BERT-Attack/scripts/adfar_og/tf.txt | python batch_run.py --gpus 0,0,0,0,0
cat /projects/p32013/DNABERT-meta/BERT-Attack/scripts/adfar_og/emp1.txt | python batch_run.py --gpus 0
# cat /projects/p32013/DNABERT-meta/BERT-Attack/scripts/adfar_og/test.txt | python batch_run.py --gpus 0
