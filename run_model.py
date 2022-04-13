from models.bert_function import run_bert_model

model = "bert-base-cased"
output_dir = "output/"

run_bert_model(
    model, 
    output_dir, 
    epochs = 40,
    max_grad_norm= 1.0, 
    batch_size= 14
    )