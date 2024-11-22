from rewrite_model.models.backbones.neuronnet import NeuronNet
import pytest

@pytest.mark.base
class TestNeuronNet:
    def test_demo1(self):
        nn = NeuronNet(2, 2, 2, hidden_layer_weights=[[0.15, 0.2], [0.25, 0.3]], hidden_layer_bias=[0.35,0.45],
                        output_layer_weights=[[0.4, 0.45], [0.5, 0.55]], output_layer_bias=[0.6,0.7])
        nn.inspect()
        print(nn.feed_forward([0, 1]))
        for i in range(100):
            nn.train([0.05, 0.1], [0.01, 0.99])
            print(i, round(nn.calculate_total_error([[[0.05, 0.1], [0.01, 0.99]]]), 9))
        nn.inspect()