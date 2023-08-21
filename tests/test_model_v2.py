import sys
sys.path.append("../model")
from LFAI_LSTM import LFAI_LSTM_V2
import unittest
import torch


class TestLFAI_LSTM(unittest.TestCase):

    def test_forward_pass(self):
        print('Running single test.')
        # Create an instance of the model
        model = LFAI_LSTM_V2(vocab_size=1, hidden_size=1, num_layers=1, block_size=1)
        hidden = model.init_hidden(1)
        # Create test data
        test_input = torch.randint(0, 1, size=(1,1))  # Shape (batch_size, input_size)

        # Perform the forward pass
        output, hidden = model(test_input, hidden)

        expected_output_shape = (1,1)  # Shape (batch_size, 1)

        # Check if the output has the expected shape
        self.assertEqual(output.view(-1, 1).shape, expected_output_shape)

    def test_forward_pass_batch(self):
        print('Running batch test.')
        # Create an instance of the model
        model = LFAI_LSTM(vocab_size=1, hidden_size=1, num_layers=1, block_size=1)
        hidden = model.init_hidden(6)
        # Create test data
        test_input = torch.randint(0, 1, size=(6,1))  # Shape (batch_size, input_size)

        # Perform the forward pass
        output, hidden = model(test_input, hidden)

        expected_output_shape = (6,1)  # Shape (batch_size, 1)

        # Check if the output has the expected shape
        self.assertEqual(output.view(-1, 1).shape, expected_output_shape)

if __name__ == '__main__':
    unittest.main()
