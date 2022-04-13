# Import packages

import numpy as np
from sklearn.metrics import classification_report
from scripts.utils import *
from tqdm import trange
from preprocessing.extra_preprocessing import *

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences

import transformers
from transformers import BertForTokenClassification, AdamW, get_linear_schedule_with_warmup

import json
import os

import matplotlib.pyplot as plt
import seaborn as sns

def run_bert_model(model_type, output_dir, epochs=40, max_grad_norm = 1.0, batch_size=32):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    torch.cuda.empty_cache()

    with open ("config.json") as json_config_file:
        config = json.load(json_config_file)
        df_answers_dir = config["input_data"]["answers"]
        df_testset_dir = config["input_data"]["testset"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer.from_pretrained(model_type, do_lower_case=False)

    sentences_trainset, labels_trainset, train_tagset = process_input_data(df_answers_dir)
    sentences_testset, labels_testset, test_tagset = process_input_data(df_testset_dir)

    # GET MAX LENGTH
    MAX_LEN = max(get_max_len(sentences_trainset), get_max_len(sentences_testset))

    tag_values = list(train_tagset.union(test_tagset))
    tag_values.append("PAD")
    tag2idx = {t: i for i, t in enumerate(tag_values)}

    # save tag2idx in prep for model load

    np.save(f'{output_dir}tag2idx.npy', tag2idx)

    # preparing training data

    tokenized_texts_and_labels_train = [
        tokenize_and_preserve_labels(sent, labs, tokenizer)
        for sent, labs in zip(sentences_trainset, labels_trainset)
    ]

    tokenized_texts_train = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels_train]
    labels_train = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels_train]

    tr_inputs = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts_train],
                            maxlen=MAX_LEN, dtype="long", value=0.0,
                            truncating="post", padding="post")

    tr_tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels_train],
                        maxlen=MAX_LEN, value=tag2idx["PAD"], padding="post",
                        dtype="long", truncating="post")


    tr_masks = [[float(i != 0.0) for i in ii] for ii in tr_inputs]

    # preparing test data

    tokenized_texts_and_labels_test = [
        tokenize_and_preserve_labels(sent, labs, tokenizer)
        for sent, labs in zip(sentences_testset, labels_testset)
    ]

    tokenized_texts_test = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels_test]
    labels_test = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels_test]

    val_inputs = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts_test],
                            maxlen=MAX_LEN, dtype="long", value=0.0,
                            truncating="post", padding="post")

    val_tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels_test],
                        maxlen=MAX_LEN, value=tag2idx["PAD"], padding="post",
                        dtype="long", truncating="post")


    val_masks = [[float(i != 0.0) for i in ii] for ii in val_inputs]

    # convert to torch tensors

    tr_inputs = torch.tensor(tr_inputs).to(torch.int64)
    val_inputs = torch.tensor(val_inputs).to(torch.int64)
    tr_tags = torch.tensor(tr_tags).to(torch.int64)
    val_tags = torch.tensor(val_tags).to(torch.int64)
    tr_masks = torch.tensor(tr_masks).to(torch.int64)
    val_masks = torch.tensor(val_masks).to(torch.int64)

    # define dataloaders

    train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    valid_data = TensorDataset(val_inputs, val_masks, val_tags)
    valid_sampler = SequentialSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=batch_size)

    # initialise model

    model = BertForTokenClassification.from_pretrained(
        model_type,
        num_labels=len(tag2idx),
        output_attentions = False,
        output_hidden_states = False
    )

    model.cuda();

    FULL_FINETUNING = True
    if FULL_FINETUNING:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {
                'params': [
                    p
                    for n, p in param_optimizer
                    if all(nd not in n for nd in no_decay)
                ],
                'weight_decay_rate': 0.01,
            },
            {
                'params': [
                    p
                    for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)
                ],
                'weight_decay_rate': 0.0,
            },
        ]

    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=3e-5,
        eps=1e-8
    )

    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    ## Store the average loss after each epoch so we can plot them.
    loss_values, validation_loss_values = [], []

    for _ in trange(epochs, desc="Epoch"):
        # ========================================
        #               Training
        # ========================================
        # Perform one full pass over the training set.

        # Put the model into training mode.
        model.train()
        # Reset the total loss for this epoch.
        total_loss = 0

        # Training loop
        for _, batch in enumerate(train_dataloader):
            # add batch to gpu
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            # Always clear any previously calculated gradients before performing a backward pass.
            model.zero_grad()
            # forward pass
            # This will return the loss (rather than the model output)
            # because we have provided the `labels`.
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask, labels=b_labels)
            # get the loss
            loss = outputs[0]
            # Perform a backward pass to calculate the gradients.
            loss.backward()
            # track train loss
            total_loss += loss.item()
            # Clip the norm of the gradient
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            # update parameters
            optimizer.step()
            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)
        print("Average train loss: {}".format(avg_train_loss))

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)


        # save the model as a checkpoint UNCOMMENT IF NECESSARY
        torch.save(model, f'{output_dir}model.pt')

        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        # Put the model into evaluation mode
        model.eval()
        # Reset the validation loss for this epoch.
        eval_loss = 0
        predictions , true_labels = [], []
        for batch in valid_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            # Telling the model not to compute or store gradients,
            # saving memory and speeding up validation
            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                # This will return the logits rather than the loss because we have not provided labels.
                outputs = model(b_input_ids, token_type_ids=None,
                                attention_mask=b_input_mask, labels=b_labels)
            # Move logits and labels to CPU
            logits = outputs[1].detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences.
            eval_loss += outputs[0].mean().item()
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            true_labels.extend(label_ids)

        eval_loss /= len(valid_dataloader)
        validation_loss_values.append(eval_loss)
        pred_tags = [tag_values[p_i] for p, l in zip(predictions, true_labels)
                                    for p_i, l_i in zip(p, l) if tag_values[l_i] != "PAD"]
        valid_tags = [tag_values[l_i] for l in true_labels
                                    for l_i in l if tag_values[l_i] != "PAD"]

    report = classification_report(
    y_pred = pred_tags,
    y_true = valid_tags,
    labels = list(set(tag2idx.keys()) - {'PAD', 'O'})
        )     

    with open(f'{output_dir}report.txt', "w+") as outfile:
        outfile.write('{}\n\n{}'.format(model_type, report))      

    # torch.save(model.state_dict(), 'weights/bert_state_dict.pt')
    torch.save(model, f'{output_dir}model.pt')

    # Use plot styling from seaborn.
    sns.set(style='darkgrid')

    # Increase the plot size and font size.
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12,6)

    # Plot the learning curve.
    plt.plot(loss_values, 'b-o', label="training loss")
    plt.plot(validation_loss_values, 'r-o', label="validation loss")

    # Label the plot.
    plt.title("Learning curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.savefig(output_dir+"learning_curve.png")

