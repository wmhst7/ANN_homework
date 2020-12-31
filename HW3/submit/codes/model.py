import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from rnn_cell import RNNCell, GRUCell, LSTMCell

class RNN(nn.Module):
    def __init__(self,
            num_embed_units,  # pretrained wordvec size
            num_units,        # RNN units size
            num_layers,       # number of RNN layers
            num_vocabs,       # vocabulary size
            wordvec,            # pretrained wordvec matrix [len(vocab_list), n_dims]
            dataloader,      # dataloader
            cell_kind
            ):

        super().__init__()

        # load pretrained wordvec
        self.wordvec = wordvec
        # the dataloader
        self.dataloader = dataloader

        # TODO START
        self.num_vocabs = num_vocabs
        self.cross_entropy = nn.CrossEntropyLoss()
        # fill the parameter for multi-layer RNN
        self.cell_kind = cell_kind
        if self.cell_kind == "RNN":
            self.cells = nn.Sequential(
                RNNCell(num_embed_units, num_units),
                *[RNNCell(num_units, num_units) for _ in range(num_layers - 1)]
            )
        elif self.cell_kind == "GRU":
            self.cells = nn.Sequential(
                GRUCell(num_embed_units, num_units),
                *[GRUCell(num_units, num_units) for _ in range(num_layers - 1)]
            )
        elif self.cell_kind == "LSTM":
            self.cells = nn.Sequential(
                LSTMCell(num_embed_units, num_units),
                *[LSTMCell(num_units, num_units) for _ in range(num_layers - 1)]
            )
        # TODO END

        # intialize other layers
        self.linear = nn.Linear(num_units, num_vocabs)

    def forward(self, batched_data, device):
        # Padded Sentences
        sent = torch.tensor(batched_data["sent"], dtype=torch.long, device=device) # shape: (batch_size, length)
        # An example:
        #   [
        #   [2, 4, 5, 6, 3, 0],   # first sentence: <go> how are you <eos> <pad>
        #   [2, 7, 3, 0, 0, 0],   # second sentence:  <go> hello <eos> <pad> <pad> <pad>
        #   [2, 7, 8, 1, 1, 3]    # third sentence: <go> hello i <unk> <unk> <eos>
        #   ]
        # You can use self.dataloader.convert_ids_to_sentence(sent[0]) to translate the first sentence to string in this batch.

        # Sentence Lengths
        length = torch.tensor(batched_data["sent_length"], dtype=torch.long, device=device) # shape: (batch)
        # An example (corresponding to the above 3 sentences):
        #   [5, 3, 6]

        batch_size, seqlen = sent.shape

        # TODO START
        # implement embedding layer
        # print(sent.shape)
        embedding = self.wordvec[sent]
        # print(embedding.shape)
        # shape: (batch_size, length, num_embed_units)
        # one hot
        # x = []
        # for i in range(batch_size):
        #     x.append(torch.zeros((seqlen, self.num_vocabs), device=device).scatter_(1, sent[i].reshape((seqlen, 1)), 1))
        # embedding = torch.stack(x, dim=0)
        # print(embedding.shape)
        # TODO END

        now_state = []
        for cell in self.cells:
            now_state.append(cell.init(batch_size, device))

        loss = []
        logits_per_step = []
        for i in range(seqlen - 1):
            hidden = embedding[:, i]
            for j, cell in enumerate(self.cells):
                hidden, now_state[j] = cell(hidden, now_state[j]) # shape: (batch_size, num_units)
            logits = self.linear(hidden) # shape: (batch_size, num_vocabs)
            logits_per_step.append(logits) # shape: [sentence_length-1, tensor:(batch_size, num_vocabs)]

        # TODO START
        # calculate loss
        logits_per_step = torch.stack(logits_per_step, dim=1)
        # print(logits_per_step.shape, logits_per_step[0,:,:2])
        # print(length.shape, length[0])
        # print(sent.shape, sent[0,:])
        for i, len in enumerate(length): # i->batch_size, len->sentence_length
            if len > 1:
                # cross_entropy((token_len, num_vocabs), (token_len))
                # token_len is used as batch_size in cross entropy
                loss.append(self.cross_entropy(logits_per_step[i, :len-1, :], sent[i, 1:len]))

        # print(loss)
        loss = torch.mean(torch.stack(loss, dim=0))
        # TODO END

        return loss, logits_per_step

    def inference(self, batch_size, device, decode_strategy, temperature, max_probability):
        # First Tokens is <go>
        now_token = torch.tensor([self.dataloader.go_id] * batch_size, dtype=torch.long, device=device)
        flag = torch.tensor([1] * batch_size, dtype=torch.float, device=device)

        now_state = []
        for cell in self.cells:
            now_state.append(cell.init(batch_size, device))

        generated_tokens = []
        for _ in range(50): # max sentecne length

            # TODO START
            # translate now_token to embedding
            embedding = self.wordvec[now_token]
            # embedding = torch.zeros((now_token.shape[0], 3689), device=device).scatter_(1, now_token.reshape((now_token.shape[0], 1)), 1)
            # shape: (batch_size, num_embed_units)
            # TODO END

            hidden = embedding
            for j, cell in enumerate(self.cells):
                hidden, now_state[j] = cell(hidden, now_state[j])
            logits = self.linear(hidden) # shape: (batch_size, num_vocabs)

            if decode_strategy == "random":
                prob = (logits / temperature).softmax(dim=-1) # shape: (batch_size, num_vocabs)
                now_token = torch.multinomial(prob, 1)[:, 0] # shape: (batch_size)
            elif decode_strategy == "top-p":
                # TODO START
                # Reference: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
                # 参考了top-p的写法
                logits_new = []
                for one_logits in logits:
                    # print("logits", one_logits)
                    # implement top-p samplings
                    sorted_logits, sorted_indices = torch.sort(one_logits, descending=True)
                    # print(sorted_logits, sorted_indices)
                    cumulative_prob = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    # print("cumulative_prob", cumulative_prob)
                    sorted_indices_to_remove = cumulative_prob > max_probability
                    # print(sorted_indices_to_remove)
                    # to avoid no token left
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    # print("sorted_indices_to_remove", sorted_indices_to_remove)
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    # print(indices_to_remove)
                    one_logits[indices_to_remove] = -float('Inf')
                    logits_new.append(one_logits)

                logits_new = torch.stack(logits_new, dim=0)
                prob = (logits_new / temperature).softmax(dim=-1)  # shape: (batch_size, num_vocabs)
                now_token = torch.multinomial(prob, 1)[:, 0] # shape: (batch_size)
                # TODO END
            else:
                raise NotImplementedError("unknown decode strategy")

            generated_tokens.append(now_token)
            flag = flag * (now_token != self.dataloader.eos_id)

            if flag.sum().tolist() == 0: # all sequences has generated the <eos> token
                break

        return torch.stack(generated_tokens, dim=1).detach().cpu().numpy()
