import torch
import torch.nn as nn


class LongShortTermMemoryModel(nn.Module):
    def __init__(self, encoding_size, label_size):
        super(LongShortTermMemoryModel, self).__init__()

        self.lstm = nn.LSTM(encoding_size, 128)  # 128 is the state size
        self.dense = nn.Linear(128, label_size)  # 128 is the state size

    def reset(self, batch_size):  # Reset states prior to new input sequence
        zero_state = torch.zeros(1, batch_size, 128)  # Shape: (number of layers, batch size, state size)
        self.hidden_state = zero_state
        self.cell_state = zero_state

    def logits(self, x):  # x shape: (sequence length, batch size, encoding size)
        out, (self.hidden_state, self.cell_state) = self.lstm(x, (self.hidden_state, self.cell_state))
        return self.dense(out[-1].reshape(-1, 128))

    def f(self, x):  # x shape: (sequence length, batch size, encoding size)
        return torch.softmax(self.logits(x), dim=1)

    def loss(self, x,
             y):  # x shape: (sequence length, batch size, encoding size), y shape: (sequence length, encoding size)
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))


char_encodings = [
    [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # ' ' 0
    [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'h' 1
    [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'a' 2
    [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 't' 3
    [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'c' 4
    [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],  # 'f' 5
    [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],  # 'p' 6
    [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],  # 's' 7
    [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],  # 'm' 8
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],  # 'r' 9
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],  # 'l' 10
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],  # 'o' 11
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]  # 'n' 12
]
encoding_size = len(char_encodings)

index_to_char = [' ', 'h', 'a', 't', 'c', 'f', 'p', 's', 'm', 'r', 'l', 'o', 'n']

# x_data = ["hatt", "ratt", "cat ", "flat", "matt", "cap ", "son "]


emoji_encodings = [
    [1., 0., 0., 0., 0., 0., 0.],  # "🐀", 'rat '
    [0., 1., 0., 0., 0., 0., 0.],  # "🎩", 'hat '
    [0., 0., 1., 0., 0., 0., 0.],  # "🐈", 'cat '
    [0., 0., 0., 1., 0., 0., 0.],  # "🏦️", 'flat'
    [0., 0., 0., 0., 1., 0., 0.],  # "🧔🏻", 'matt'
    [0., 0., 0., 0., 0., 1., 0.],  # "⛑", 'cap '
    [0., 0., 0., 0., 0., 0., 1.]  # "👶🏻", 'son '
]
y_data = ["🎩", "🐀", "🐈", "🏦", "🧔🏻", "⛑", "👶🏻"]

x_train = torch.tensor(
    [[char_encodings[1], char_encodings[2], char_encodings[3], char_encodings[3]],
     [char_encodings[9], char_encodings[2], char_encodings[3], char_encodings[3]],
     [char_encodings[4], char_encodings[2], char_encodings[3], char_encodings[0]],
     [char_encodings[5], char_encodings[10], char_encodings[2], char_encodings[3]],
     [char_encodings[8], char_encodings[2], char_encodings[3], char_encodings[3]],
     [char_encodings[4], char_encodings[2], char_encodings[6], char_encodings[0]],
     [char_encodings[7], char_encodings[11], char_encodings[12], char_encodings[0]]
     ]).transpose(0, 1)  #

y_train = torch.tensor(
    [emoji_encodings[0], emoji_encodings[1], emoji_encodings[2], emoji_encodings[3], emoji_encodings[4],
     emoji_encodings[5], emoji_encodings[6]])  #

label_size = len(y_data)
model = LongShortTermMemoryModel(encoding_size, label_size)

optimizer = torch.optim.RMSprop(model.parameters(), 0.01)
for epoch in range(5000):
    model.reset(x_train.size(1))
    model.loss(x_train, y_train).backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 10 == 9:
        #
        model.reset(1)
        text = 'natt'
        index_of_first_char = 0
        index_of_second_char = 0
        index_of_third_char = 0
        index_of_forth_char = 0
        for find in range(len(index_to_char)):
            if text[0] == index_to_char[find]:
                index_of_first_char = find
            if text[1] == index_to_char[find]:
                index_of_second_char = find
            if text[2] == index_to_char[find]:
                index_of_third_char = find
            if text[3] == index_to_char[find]:
                index_of_forth_char = find

        y = model.f(torch.tensor([[char_encodings[index_of_first_char]], [char_encodings[index_of_second_char]],
                                  [char_encodings[index_of_third_char]], [char_encodings[index_of_forth_char]]]))
        # y = model.f(torch.tensor([[char_encodings[index_of_second_char]]]))

        print(y_data[y.argmax(1)])
        print(y.argmax(1))
