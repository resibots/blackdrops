#ifndef MEDROPS_NN_NN_HPP
#define MEDROPS_NN_NN_HPP

namespace medrops {

    struct PfSum {
        double operator()(const std::vector<double>& in, const std::vector<double>& w) const
        {
            std::vector<double> res(in.size(), 0.0);
            std::transform(in.begin(), in.end(), w.begin(), res.begin(), std::multiplies<double>());

            return std::accumulate(res.begin(), res.end(), 0.0);
        }
    };

    struct AfRect {
        double operator()(double t) const
        {
            return std::log(1 + std::exp(t));
        }
    };

    struct AfSigmoid {
        double operator()(double t) const
        {
            return 1.0 / (std::exp(-t) + 1.0);
        }
    };

    struct AfFastSigmoid {
        double operator()(double t) const
        {
            return (t / (1.0 + std::abs(t)));
        }
    };

    struct AfErf {
        double operator()(double t) const
        {
            return std::erf(t);
        }
    };

    struct AfTanh {
        double operator()(double t) const
        {
            return std::tanh(t);
        }
    };

    struct AfSin {
        double operator()(double t) const
        {
            return std::sin(t);
        }
    };

    struct AfCos {
        double operator()(double t) const
        {
            return std::cos(t);
        }
    };

    struct AfGaussian {
        double operator()(double t) const
        {
            return std::exp(-t * t);
        }
    };

    struct AfDirect {
        double operator()(double t) const
        {
            return t;
        }
    };

    template <typename Af>
    struct Neuron {
        double compute(double input) const
        {
            return Af()(input);
        }
    };

    // Fully connected layer
    template <typename Neuron, typename Pf>
    struct NNLayer {
    public:
        NNLayer()
        {
            _num_weights = -1;
        }

        NNLayer(size_t N)
        {
            _neurons = std::vector<Neuron>(N);
            _num_weights = -1;
        }

        NNLayer(size_t N, size_t n_i)
        {
            _neurons = std::vector<Neuron>(N);
            _num_weights = n_i * N;
            _weights = std::vector<double>(_num_weights);
        }

        void set_weights(const std::vector<double>& weights)
        {
            if (_num_weights != -1)
                assert(weights.size() == (size_t)_num_weights);
            _weights = weights;
        }

        std::vector<double> weights() const
        {
            return _weights;
        }

        int n_weights() const
        {
            return _num_weights;
        }

        std::vector<double> step(const std::vector<double>& inputs) const
        {
            assert(_weights.size() == (inputs.size() * _neurons.size()));

            std::vector<double> res;

            for (size_t i = 0; i < _neurons.size(); i++) {
                std::vector<double> w(_weights.begin() + i * inputs.size(), _weights.begin() + (i + 1) * inputs.size());
                res.push_back(_neurons[i].compute(Pf()(inputs, w)));
            }

            return res;
        }

    protected:
        std::vector<Neuron> _neurons;
        std::vector<double> _weights;
        int _num_weights;
    };

    // Multi-Layer Perceptron (Fully connected feedforward NN)
    template <typename HiddenLayer, typename OutputLayer>
    struct MLP {
    public:
        MLP(size_t n_inputs, std::vector<size_t> n_hidden, size_t n_outputs)
        {
            assert(n_hidden.size() > 0);

            // add bias unit
            _hidden_layers.push_back(HiddenLayer(n_hidden[0], n_inputs + 1));

            for (size_t i = 1; i < n_hidden.size(); i++) {
                // add bias unit
                _hidden_layers.push_back(HiddenLayer(n_hidden[i], n_hidden[i - 1] + 1));
            }

            // add bias unit
            _output_layer = OutputLayer(n_outputs, n_hidden.back() + 1);
        }

        void set_weights(const std::vector<double>& weights)
        {
            size_t k = 0;
            for (size_t i = 0; i < _hidden_layers.size(); i++) {
                std::vector<double> w(weights.begin() + k, weights.begin() + k + _hidden_layers[i].n_weights());
                _hidden_layers[i].set_weights(w);
                k += _hidden_layers[i].n_weights();
            }

            std::vector<double> w(weights.begin() + k, weights.end());
            _output_layer.set_weights(w);
        }

        std::vector<double> weights() const
        {
            std::vector<double> w = _hidden_layers[0].weights();
            for (size_t i = 1; i < _hidden_layers.size(); i++) {
                auto h_w = _hidden_layers.weights();
                w.insert(w.end(), h_w.begin(), h_w.end());
            }

            auto o_w = _output_layer.weights();
            w.insert(w.end(), o_w.begin(), o_w.end());

            return w;
        }

        std::vector<double> compute(const std::vector<double>& inputs) const
        {
            std::vector<double> in = inputs;
            for (size_t i = 0; i < _hidden_layers.size(); i++) {
                // add bias to inputs
                in.push_back(1.0);
                in = _hidden_layers[i].step(in);
            }

            // add bias to inputs
            in.push_back(1.0);
            in = _output_layer.step(in);

            return in;
        }

        int n_weights() const
        {
            int n = 0;
            for (size_t i = 0; i < _hidden_layers.size(); i++) {
                n += _hidden_layers[i].n_weights();
            }
            n += _output_layer.n_weights();

            return n;
        }

    protected:
        std::vector<HiddenLayer> _hidden_layers;
        OutputLayer _output_layer;
    };
}

#endif
