from GenoArmory import GenoArmory

# You need to initialize GenoArmory with a model and tokenizer.
# For visualization, you don't need a real model/tokenizer, so you can use None if the method doesn't use them.
gen = GenoArmory(model=None, tokenizer=None)

# gen.visualization(
#     folder_path='/projects/p32013/DNABERT-meta/BERT-Attack/results/meta/test',
#     output_pdf_path='/projects/p32013/DNABERT-meta/BERT-Attack/results/meta/test'
# )

gen.attack(
    
)