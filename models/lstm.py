import torch.nn as nn
import torchtext


class SentimentLSTM(nn.Module):
    def __init__(self,
                 num_layers,
                 embedding_size,
                 hidden_size,
                 #input_size,
                 vocab_size=12950,
                 output_size=3):
        super().__init__()
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_size
        self.hidden_size = hidden_size
        #self.input_size = input_size
        self.output_size = output_size

        #Embedding and LSTM layers
        # Load pre-trained GloVe embeddings
        #glove = torchtext.vocab.GloVe(name='6B', dim=self.embedding_dim)
        #self.embedding = nn.Embedding.from_pretrained(glove.vectors)
        self.embedding=nn.Embedding(self.vocab_size, self.embedding_dim)
        self.lstm=nn.LSTM(self.embedding_dim,
                            self.hidden_size,
                            self.num_layers,
                            batch_first=True)

        #Linear and sigmoid layer
        self.fc1=nn.Linear(self.hidden_size, 3)
        #self.fc2=nn.Linear(64, 16)
        #self.fc3=nn.Linear(16, self.output_size)
        self.act = nn.ReLU()
        self.final_act = nn.Softmax(dim=1)


    def forward(self, x):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        #batch_size=x.size()
        #hidden_states = torch.zeros(self.num_layers,
        #                     x.shape[0], self.hidden_size)
        #cell_states = torch.zeros(self.num_layers,
        #                    x.shape[0], self.hidden_size)

        #Embadding and LSTM output
        #print(x)
        embeddings = self.embedding(x)
        #out, h = self.lstm(embeddings, (hidden_states, cell_states))
        out, (h, _) = self.lstm(embeddings)
        #out = out.flatten(start_dim=1)
        #out = out[:, -1, :]
        #print(out.shape, h.shape)
        #h = torch.squeeze(h, 0)
        h = h[-1, :, :]
        #print(h.shape)
        out = self.final_act(self.fc1(h))
        #out = self.act(self.fc2(out))
        #out = self.final_act(self.fc3(out))




        return out
