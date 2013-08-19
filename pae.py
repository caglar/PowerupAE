import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import numpy

from ae import Autoencoder, CostType
from collections import OrderedDict
from ae_utils import as_floatX

#Contractive Autoencoder implementation.
class PowerupAutoencoder(Autoencoder):

    def __init__(self,
            input,
            nvis,
            nhid,
            num_pieces,
            rnd=None,
            rho=0.96,
            theano_rng=None,
            bhid=None,
            max_col_norm=2.9368,
            momentum=0.6,
            p_mean=2.0,
            p_std=0.005,
            orth_pen=-1,
            p_decay=-1,
            L2_reg=-1,
            L1_reg=-1,
            L1_act_reg=-1,
            sigma=0.06,
            tied_weights=False,
            sparse_initialize=False,
            cost_type=CostType.MeanSquared,
            bvis=None):

        self.sigma = sigma
        self.num_pieces = num_pieces
        self.max_col_norm = max_col_norm
        self.momentum = momentum
        self.rho = rho
        self.nhiddens = nhid
        detector_layer_dim = nhid * num_pieces
        self.p_decay = p_decay
        self.orth_pen = orth_pen
        self.learning_rate = None
        self.L1_act_reg = L1_act_reg

        # create a Theano random generator that gives symbolic random values
        super(PowerupAutoencoder, self).__init__(input,
                nvis,
                rnd=rnd,
                bhid=bhid,
                cost_type=cost_type,
                nhid=detector_layer_dim,
                num_pieces=num_pieces,
                nonlinearity=None,
                nvis_dec=nhid,
                nhid_dec=nvis,
                L2_reg=L2_reg,
                L1_reg=L1_reg,
                sparse_initialize=sparse_initialize,
                tied_weights=tied_weights,
                bvis=bvis)

        if not theano_rng :
            theano_rng = RandomStreams(rnd.randint(2 ** 30))

        self.theano_rng = theano_rng
        self.p = theano.shared(self.get_p_vals(mean=p_mean, std=p_std))
        self.p.name = "power"

    def get_p_vals(self, mean=None, std=None):
        p_vals = numpy.cast[theano.config.floatX](self.rnd.normal(loc=mean, scale=std,
            size=(self.nhiddens,)))
        return p_vals

    def get_linear_hidden_outs(self, x_in=None):
        if x_in is None:
            x_in = self.x
        return T.dot(x_in, self.hidden.W) + self.hidden.b

    def corrupt_input(self, in_data, corruption_level, noise_mode="gaussian"):
        if noise_mode == "gaussian":
            return in_data + self.theano_rng.normal(avg=0.0,
                    std=corruption_level,
                    size=(in_data.shape))
        else:
            return self.theano_rng.binomial(in_data.shape,
                    n=1,
                    p=1-corruption_level,
                    dtype=theano.config.floatX) * in_data

    def get_power_sgd_updates(self,
                            batch_size,
                            learning_rate=None,
                            corrupted_input=None,
                            epsilon=1e-6,
                            lr_scalers=None):

        if learning_rate is not None:
            self.learning_rate.set_value(as_floatX(learning_rate))
        param = self.p

        if corrupted_input is not None:
            X = corrupted_input
        else:
            X = self.x

        h = self.encode(X, batch_size)

        if self.tied_weights:
            x_rec = self.decode_tied(h)
        else:
            x_rec = self.decode(h)

        cost = self.get_rec_cost(x_rec)

        if self.p_decay != -1:
            p_decay_reg = T.sqr(2.-self.p.mean())
            cost += self.p_decay * p_decay_reg
        #self.learning_rate = theano.shared(numpy.asarray(learning_rate, dtype=theano.config.floatX))
        assert self.learning_rate is not None

        #Initialize parameters for rmsprop:
        accumulators = OrderedDict({})
        accumulators_mgrad = OrderedDict({})
        exp_sqr_grads = OrderedDict({})
        exp_sqr_ups = OrderedDict({})
        e0s = OrderedDict({})
        learn_rates = []
        momentum = self.momentum

        eps_p = numpy.zeros_like(param.get_value())
        accumulators[self.p] = theano.shared(value=as_floatX(eps_p), name="acc_%s" % self.p.name)
        accumulators_mgrad[self.p] = theano.shared(value=as_floatX(eps_p), name="acc_mgrad%s" % self.p.name)
        exp_sqr_grads[self.p] = theano.shared(value=as_floatX(eps_p), name="exp_grad_%s" % self.p.name)
        exp_sqr_ups[self.p] = theano.shared(value=as_floatX(eps_p), name="exp_grad_%s" % self.p.name)
        e0s[self.p] = self.learning_rate
        gparam  = T.grad(cost, self.p)
        gparam = self.clip_grads(gparam, max_norm=1.3)#T.clip(param, -1.4, 1.4)
        updates = OrderedDict({})
        i = 0
        acc = accumulators[param]
        if self.rho is not None:
            rms_grad = self.rho * acc + (1 - self.rho) * T.sqr(gparam)

        updates[acc] = rms_grad
        val = T.maximum(T.sqrt(T.sum(rms_grad, axis=0)), epsilon)
        lr_scaler = numpy.cast[theano.config.floatX](1.0)

        if lr_scalers is not None and param.name in lr_scalers:
            lr_scaler = numpy.cast[theano.config.floatX](lr_scalers[param.name])

        learn_rates.append(e0s[param]/val)

        if momentum:
            memory = theano.shared(param.get_value() * 0.)
            new_memo = momentum * memory - (lr_scaler * e0s[param] * gparam)
            updates[memory] = new_memo
            updates[param] = param + (momentum * new_memo - lr_scaler * e0s[param] * gparam) / val
        else:
            updates[param] = param - lr_scaler * learn_rates[i] * gparam
        i +=1

        return (cost, updates)


    def get_pae_sgd_updates(self,
                            batch_size,
                            learning_rate=None,
                            corrupted_input=None,
                            epsilon=1e-6,
                            lr_scalers=None):

        if learning_rate is not None:
            self.learning_rate.set_value(as_floatX(learning_rate))

        if corrupted_input is not None:
            X = corrupted_input
        else:
            X = self.input
        h = self.encode(X, batch_size)

        if self.tied_weights:
            x_rec = self.decode_tied(h)
        else:
            x_rec = self.decode(h)

        cost = self.get_rec_cost(x_rec)

        if self.L1_reg != -1 and self.L1_reg is not None:
            cost += self.L1_reg * self.L1

        if self.L2_reg != -1 and self.L2_reg is not None:
            cost += self.L2_reg * self.L2

        if self.L1_act_reg != -1:
            L1_act = abs(h).sum()
            cost += self.L1_act_reg * L1_act

        if self.orth_pen != -1:
            #W_pooled = self.hidden.W.reshape((self.nvis, self.nhiddens, self.num_pieces)).sum(2)
            W_pooled = h #self.encode(self.input, batch_size) #self.hidden.W
            W = T.dot(W_pooled.T, W_pooled)
            orth_pen_reg = T.sum(T.sqr(T.identity_like(W) - W), axis=0).mean()
            cost += self.orth_pen * orth_pen_reg
            #W_prime = T.dot(self.hidden.W_prime, self.hidden.W_prime.T)
            #orth_pen_reg_prime = T.sum(T.sqr(T.identity_like(W_prime) - W_prime), axis=0).mean()
            #cost += self.orth_pen * orth_pen_reg_prime

        #self.learning_rate = theano.shared(numpy.asarray(learning_rate, dtype=theano.config.floatX))
        assert self.learning_rate is not None

        #Initialize parameters for rmsprop:
        accumulators = OrderedDict({})
        accumulators_mgrad = OrderedDict({})
        exp_sqr_grads = OrderedDict({})
        exp_sqr_ups = OrderedDict({})
        e0s = OrderedDict({})
        learn_rates = []
        momentum = self.momentum
        gparams = []

        for param in self.params:
            eps_p = numpy.zeros_like(param.get_value())
            accumulators[param] = theano.shared(value=as_floatX(eps_p), name="acc_%s" % param.name)
            accumulators_mgrad[param] = theano.shared(value=as_floatX(eps_p), name="acc_mgrad%s" % param.name)
            exp_sqr_grads[param] = theano.shared(value=as_floatX(eps_p), name="exp_grad_%s" % param.name)
            exp_sqr_ups[param] = theano.shared(value=as_floatX(eps_p), name="exp_grad_%s" % param.name)
            e0s[param] = self.learning_rate
            gparam  = T.grad(cost, param)
            gparam = self.clip_grads(gparam, max_norm=2.5)#T.clip(param, -1.4, 1.4)
            gparams.append(gparam)

        updates = OrderedDict({})
        i = 0

        for param, gparam in zip(self.params, gparams):
            acc = accumulators[param]
            if self.rho is not None:
                rms_grad = self.rho * acc + (1 - self.rho) * T.sqr(gparam)

            updates[acc] = rms_grad
            val = T.maximum(T.sqrt(T.sum(rms_grad, axis=0)), epsilon)
            lr_scaler = numpy.cast[theano.config.floatX](1.0)

            if lr_scalers is not None and param.name in lr_scalers:
                lr_scaler = numpy.cast[theano.config.floatX](lr_scalers[param.name])

            learn_rates.append(e0s[param]/val)

            if momentum:
                memory = theano.shared(param.get_value() * 0.)
                new_memo = momentum * memory - lr_scaler * e0s[param] * gparam
                updates[memory] = new_memo
                if param.name != "power_":
                    updates[param] = param + (momentum * new_memo - lr_scaler * e0s[param] * gparam) / val
                else:
                    updates[param] = param + (momentum * new_memo - lr_scaler * e0s[param] * gparam)
            else:
                updates[param] = param - lr_scaler * learn_rates[i] * gparam

            i +=1

        if self.max_col_norm is not None:
            updates = self.constrain_weights(self.hidden.W, updates)
            updates = self.constrain_weights(self.hidden.W_prime, updates)

        return (cost, updates)

    def clip_grads(self, grads, max_norm=1., epsilon=1e-8):
        norm = T.sqrt(T.sum(T.sqr(grads)))
        desired_norm = T.clip(norm, 0., max_norm)
        rescaled_grads = grads * desired_norm / (norm + epsilon)
        return rescaled_grads

    def constrain_weights(self, weight, updates, epsilon=1e-8):
        updated_W = updates[weight]
        col_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=0))
        desired_norms = T.clip(col_norms, 0., self.max_col_norm)
        updates[weight] = updated_W * desired_norms / (epsilon + col_norms)
        return updates

    def stddev_bias(self, x, eps, axis=0):
        mu = T.mean(x + eps, axis=axis)
        mu.name = "std_mean"
        var = T.mean((x - mu)**2 + eps)
        var.name = "std_variance"
        stddev = T.sqrt(var)
        return stddev

    def encode(self, d_in, batch_size=100):
        z_in = T.dot(d_in, self.hidden.W) #+ self.hidden.b
        z_in.name = "z_in"
        h = self.powerup(z_in, batch_size=batch_size)
        h_std = (h - h.mean(0)) / self.stddev_bias(h, eps=1e-8)
        return h_std

    def decode(self, h):
        #pT = T.exp(T.log(self.p.dimshuffle('x', 0)))
        decoded = T.nnet.sigmoid(T.dot(h, self.hidden.W_prime.T) + self.hidden.b_prime)
        #decoded = T.tanh(T.dot(h, self.hidden.W_prime.T) + self.hidden.b_prime)
        #decoded = T.nnet.softplus(T.dot(h, self.hidden.W_prime.T) + self.hidden.b_prime)
        return decoded

    def decode_tied(self, h, epsilon=1e-6):
        #z_in = T.dot(h, self.hidden.W_prime) + self.hidden.b_prime
        pT = T.exp(T.log(self.p.dimshuffle(0, 'x', 'x')))
        w_pooled = abs(self.hidden.W.T.reshape((self.nhiddens, self.num_pieces, self.nvis)))**pT
        pT = pT.dimshuffle(0, 'x')
        w_norm = T.maximum(T.sum(w_pooled, axis=1) ** 1/pT, epsilon)
        recons = (abs(h.T) / w_norm) + self.hidden.b_prime
        return recons

    def maxup(self, z_in=None):
        assert z_in is not None, "input to powerup nonlinearity function should not be empty."
        z = z_in.dimshuffle(1, 0)
        z_summed_pools = T.sum(z_pools, axis=1)
        pT = pT.dimshuffle(0, 'x')
        z_summed_pools = z_summed_pools**(1./pT)
        z = z_summed_pools.dimshuffle(1, 0)
        z = z + self.hidden.b
        return z

    def powerup(self, z_in=None, batch_size=100):
        assert z_in is not None, "input to powerup nonlinearity function should not be empty."
        z = z_in.dimshuffle(1, 0)

        z_pools = z.reshape((self.num_pieces, self.nhiddens, batch_size))
        #pT = T.exp(T.log(self.p.dimshuffle('x', 0, 'x')))
        pT = T.exp(self.p.dimshuffle('x', 0, 'x'))
        pT = abs(self.p.dimshuffle('x', 0, 'x'))

        z_pools = abs(z_pools)**pT
        z_summed_pools = T.sum(z_pools, axis=0)

        #pT = T.exp(T.log(self.p.dimshuffle(0, 'x')))
        pT = T.exp(self.p.dimshuffle(0, 'x'))
        pT = abs(self.p.dimshuffle(0, 'x'))

        z_summed_pools = z_summed_pools**(1./pT)
        z = z_summed_pools.dimshuffle(1, 0)
        z = z + self.hidden.b
        return z

    def normalizefilters(self, center=True, SMALL=1e-6):
        def inplacemult(x, v):
            x[:, :] *= v
            return x
        def inplacesubtract(x, v):
            x[:, :] -= v
            return x
        nwxf = (self.hidden.W.get_value().std(0)+SMALL)[numpy.newaxis, :]
        nwyf = (self.hidden.W_prime.get_value().std(0)+SMALL)[numpy.newaxis, :]

        meannxf = nwxf.mean()
        meannyf = nwyf.mean()

        wxf = self.hidden.W.get_value(borrow=True)
        wyf = self.hidden.W_prime.get_value(borrow=True)

        # CENTER FILTERS
        if center:
            self.hidden.W.set_value(inplacesubtract(wxf, wxf.mean(0)[numpy.newaxis,:]), borrow=True)
            self.hidden.W_prime.set_value(inplacesubtract(wyf, wyf.mean(0)[numpy.newaxis,:]), borrow=True)

        # FIX STANDARD DEVIATION
        self.hidden.W.set_value(inplacemult(wxf, meannxf/nwxf),borrow=True)
        self.hidden.W_prime.set_value(inplacemult(wyf, meannyf/nwyf),borrow=True)

    def fit(self,
            data=None,
            learning_rate=0.2,
            batch_size=100,
            power_batch_size=-1,
            lr_scalers=None,
            n_epochs=32,
            corruption_level=0.1,
            shuffle_data=True,
            weights_file="out/pae_weights_mnist.npy"):

        if power_batch_size == -1:
            power_batch_size =  int(6 * batch_size)

        if data is None:
            raise Exception("Data can't be empty.")

        index = T.iscalar('index')
        data = numpy.asarray(data.tolist(), dtype="float32")
        data_shared = theano.shared(data)
        corrupted_input = None

        if corruption_level != -1:
            corrupted_input = self.corrupt_input(self.x, corruption_level)


        self.learning_rate = theano.shared(as_floatX(learning_rate), name="learning_rate")

        (cost, updates) = self.get_pae_sgd_updates(batch_size,
                corrupted_input=corrupted_input, lr_scalers=lr_scalers)

        (cost_p, updates_p) = self.get_power_sgd_updates(power_batch_size,
                corrupted_input=corrupted_input, lr_scalers=lr_scalers)


        train_ae = theano.function([index],
                                   cost,
                                   updates=updates,
                                   givens = {
                                       self.x: data_shared[index * batch_size: (index + 1) * batch_size]
                                    })

        train_power = theano.function([index],
                                   cost_p,
                                   updates=updates_p,
                                   givens = {
                                       self.x: data_shared[index * power_batch_size: (index + 1) * power_batch_size]
                                    })

        print "Started the training."

        best_ae_cost = float('inf')
        lr_border = 10
        decay_rate = 0.93
        final_lr = 0.000001
        n_p_epoch = 4
        gap_threshold = 1.4

        for epoch in xrange(n_epochs):

            ae_costs = []
            power_n_batches = data.shape[0] / power_batch_size
            n_batches = data.shape[0] / batch_size

            power_idxs = numpy.arange(power_n_batches)
            numpy.random.shuffle(power_idxs)

            idxs = numpy.arange(n_batches)
            numpy.random.shuffle(idxs)
            print "Training at epoch %d" % epoch
            idxs = idxs[50:]

            if epoch % n_p_epoch == 0:
                for i in xrange(n_p_epoch/2):
                    for batch_index in power_idxs:
                        ae_cost = train_power(batch_index)

                    print "Mean power is: ", self.p.get_value(borrow=True).mean()
                    print "Power adjusted, %d." % (i)
                #power_batch_size += 20

            for batch_index in idxs:
                ae_cost = train_ae(batch_index)
                ae_costs.append(ae_cost)

            ave_cost = numpy.mean(ae_costs)

            if ave_cost <= best_ae_cost:
                best_ae_cost = ave_cost

            gap_percent = (ave_cost - best_ae_cost) * 100 / (best_ae_cost + ave_cost)
            if epoch > lr_border:
                print "Gap percent ", gap_percent
                if gap_percent >= gap_threshold and self.learning_rate.get_value() > final_lr:
                    self.learning_rate.set_value(as_floatX(self.learning_rate.get_value() * decay_rate))

            print "Training at epoch %d, %f, learning_rate: %f" % (epoch, numpy.mean(ae_costs),  self.learning_rate.get_value())
            self.normalizefilters()
            print "Saving weights..."
            numpy.save(weights_file, self.params[0].get_value())
            epoch += 1

        print self.p.get_value()
        return ae_costs

