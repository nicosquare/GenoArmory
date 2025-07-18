"""
**For most use cases, this can just be considered an internal class and
ignored.**

This module houses the :class:`robustness.attacker.Attacker` and
:class:`robustness.attacker.AttackerModel` classes. 

:class:`~robustness.attacker.Attacker` is an internal class that should not be
imported/called from outside the library.
:class:`~robustness.attacker.AttackerModel` is a "wrapper" class which is fed a
model and adds to it adversarial attack functionalities as well as other useful
options. See :meth:`robustness.attacker.AttackerModel.forward` for documentation
on which arguments AttackerModel supports, and see
:meth:`robustness.attacker.Attacker.forward` for the arguments pertaining to
adversarial examples specifically.

For a demonstration of this module in action, see the walkthrough
":doc:`../example_usage/input_space_manipulation`"

**Note 1**: :samp:`.forward()` should never be called directly but instead the
AttackerModel object itself should be called, just like with any
:samp:`nn.Module` subclass.

**Note 2**: Even though the adversarial example arguments are documented in
:meth:`robustness.attacker.Attacker.forward`, this function should never be
called directly---instead, these arguments are passed along from
:meth:`robustness.attacker.AttackerModel.forward`.
"""

import torch as ch
import dill
import os
if int(os.environ.get("NOTEBOOK_MODE", 0)) == 1:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

import torch
from .tools import helpers
from . import attack_steps

STEPS = {
    'inf': attack_steps.LinfStep,
    '2': attack_steps.L2Step,
    'unconstrained': attack_steps.UnconstrainedStep,
    'fourier': attack_steps.FourierStep,
    'random_smooth': attack_steps.RandomStep
}

class Attacker(ch.nn.Module):
    """
    Attacker class, used to make adversarial examples.

    This is primarily an internal class, you probably want to be looking at
    :class:`robustness.attacker.AttackerModel`, which is how models are actually
    served (AttackerModel uses this Attacker class).

    However, the :meth:`robustness.Attacker.forward` function below
    documents the arguments supported for adversarial attacks specifically.
    """
    def __init__(self, args, model, dataset):
        """
        Initialize the Attacker

        Args:
            nn.Module model : the PyTorch model to attack
            Dataset dataset : dataset the model is trained on, only used to get mean and std for normalization
        """
        super(Attacker, self).__init__()
        #self.normalize = helpers.InputNormalize(dataset.mean, dataset.std)
        self.model = model
        self.args = args

    def forward(self, inp, target, *_, constraint, eps, step_size, iterations,
                random_start=False, random_restarts=False, do_tqdm=False,
                targeted=False, custom_loss=None, should_normalize=False,
                orig_input=None, use_best=True, return_image=True,
                est_grad=None, mixed_precision=False):
        """
        Implementation of forward (finds adversarial examples). Note that
        this does **not** perform inference and should not be called
        directly; refer to :meth:`robustness.attacker.AttackerModel.forward`
        for the function you should actually be calling.

        Args:
            x, target (ch.tensor) : see :meth:`robustness.attacker.AttackerModel.forward`
            constraint
                ("2"|"inf"|"unconstrained"|"fourier"|:class:`~robustness.attack_steps.AttackerStep`)
                : threat model for adversarial attacks (:math:`\ell_2` ball,
                :math:`\ell_\infty` ball, :math:`[0, 1]^n`, Fourier basis, or
                custom AttackerStep subclass).
            eps (float) : radius for threat model.
            step_size (float) : step size for adversarial attacks.
            iterations (int): number of steps for adversarial attacks.
            random_start (bool) : if True, start the attack with a random step.
            random_restarts (bool) : if True, do many random restarts and
                take the worst attack (in terms of loss) per input.
            do_tqdm (bool) : if True, show a tqdm progress bar for the attack.
            targeted (bool) : if True (False), minimize (maximize) the loss.
            custom_loss (function|None) : if provided, used instead of the
                criterion as the loss to maximize/minimize during
                adversarial attack. The function should take in
                :samp:`model, x, target` and return a tuple of the form
                :samp:`loss, None`, where loss is a tensor of size N
                (per-element loss).
            should_normalize (bool) : If False, don't normalize the input
                (not recommended unless normalization is done in the
                custom_loss instead).
            orig_input (ch.tensor|None) : If not None, use this as the
                center of the perturbation set, rather than :samp:`x`.
            use_best (bool) : If True, use the best (in terms of loss)
                iterate of the attack process instead of just the last one.
            return_image (bool) : If True (default), then return the adversarial
                example as an image, otherwise return it in its parameterization
                (for example, the Fourier coefficients if 'constraint' is
                'fourier')
            est_grad (tuple|None) : If not None (default), then these are
                :samp:`(query_radius [R], num_queries [N])` to use for estimating the
                gradient instead of autograd. We use the spherical gradient
                estimator, shown below, along with antithetic sampling [#f1]_
                to reduce variance:
                :math:`\\nabla_x f(x) \\approx \\sum_{i=0}^N f(x + R\\cdot
                \\vec{\\delta_i})\\cdot \\vec{\\delta_i}`, where
                :math:`\delta_i` are randomly sampled from the unit ball.
            mixed_precision (bool) : if True, use mixed-precision calculations
                to compute the adversarial examples / do the inference.
        Returns:
            An adversarial example for x (i.e. within a feasible set
            determined by `eps` and `constraint`, but classified as:

            * `target` (if `targeted == True`)
            *  not `target` (if `targeted == False`)

        .. [#f1] This means that we actually draw :math:`N/2` random vectors
            from the unit ball, and then use :math:`\delta_{N/2+i} =
            -\delta_{i}`.
        """
        # Can provide a different input to make the feasible set around
        # instead of the initial point
        input_ids = inp.get("input_ids")
        attention_mask = inp.get("attention_mask")
        labels = inp.get('labels')

        with torch.no_grad():  # Prevent initial gradients during embedding lookup
            inputs_embeds = self.model.get_input_embeddings()(inp["input_ids"].to(self.model.device)).detach()
        
        inputs_embeds.requires_grad_(True)
        inp['inputs_embeds'] = inputs_embeds


        if orig_input is None:
            orig_input = {"inputs_embeds": inputs_embeds,
                          "attention_mask": attention_mask,
                          "labels":labels}


        # Multiplier for gradient ascent [untargeted] or descent [targeted]
        m = -1 if targeted else 1
        self.model.train()
        # Initialize step class and attacker criterion
        criterion = ch.nn.CrossEntropyLoss(reduction='none')
        step_class = STEPS[constraint] if isinstance(constraint, str) else constraint
        step = step_class(eps=eps, orig_input=orig_input, step_size=step_size) 


        # Main function for making adversarial examples
        def get_adv_examples(x):
            # Random start (to escape certain types of gradient masking)
            if random_start:
                x["inputs_embeds"] = step.random_perturb(x["inputs_embeds"])

            if "attention_mask" not in x:
                x["attention_mask"] = ch.ones_like(x["input_ids"]).to(self.model.device)  # Added for robustness

            def calc_loss(inp, target):
                '''
                Calculates the loss of an input with respect to target labels
                Uses custom loss (if provided) otherwise the criterion
                '''
                
                if should_normalize:
                    inp["inputs_embeds"] = self.normalize(inp["inputs_embeds"].to(self.model.device))
                                
                output = self.model(**inp) 
                
                if custom_loss:
                    return custom_loss(self.model, inp, target)

                return criterion(output.logits, target), output.logits

            iterator = range(iterations)
            if do_tqdm:
                iterator = tqdm(iterator)

            # Track the "best" (worst-case) loss and its corresponding input
            best_loss = None
            best_x = None

            # A function that updates the best loss and best input
            def replace_best(loss, bloss, x, bx):
                if bloss is None:
                    bx = {"inputs_embeds": x["inputs_embeds"],
                        "attention_mask": x["attention_mask"],
                        "labels":x["labels"]}
                    bloss = loss.clone().detach()
                else:
                    replace = m * bloss < m * loss
                    
                    
                    bx["inputs_embeds"][replace] = x["inputs_embeds"][replace].detach().clone()
                    
                    bloss[replace] = loss[replace]
                    
                    device = x["attention_mask"].device  # Extract device from `bx`
                    replace = replace.to(device)         # Move `replace` to the same device
                    #print(replace.device)

                    bx["attention_mask"][replace] = x["attention_mask"][replace]
                    bx["labels"][replace] = x["labels"][replace]
                    
                    
                return bloss, bx

            # PGD iterates
            for _ in iterator:
                inputs_embeds = x['inputs_embeds']
                inputs_embeds.requires_grad_(True)

                losses, out = calc_loss({"inputs_embeds": step.to_image(inputs_embeds).to(self.model.device),
                                 "attention_mask": x["attention_mask"].to(self.model.device) if 'attention_mask' in x else None, "labels": x["labels"].to(self.model.device)}, target.to(self.model.device))
                
                loss = ch.mean(losses)

                if step.use_grad:
                    if (est_grad is None) and mixed_precision:
                        import torch.cuda.amp as amp
                        with amp.scale_loss(loss, []) as sl:
                            sl.backward()
                        grad = inputs_embeds.grad.detach()
                        inputs_embeds.grad.zero_()
                    elif (est_grad is None):
                        new_loss = m * loss
                        
                        grad, = ch.autograd.grad(new_loss, [inputs_embeds])
                    else:
                        f = lambda _x, _y: m * calc_loss(step.to_image(_x), _y)[0]
                        grad = helpers.calc_est_grad(f, inputs_embeds, target, *est_grad)
                else:
                    grad = None


                with ch.no_grad():
                    args = [losses, best_loss, x, best_x]
                    best_loss, best_x = replace_best(*args) if use_best else (losses, x)
                    inputs_embeds = step.step(inputs_embeds, grad)
                    inputs_embeds = step.project(inputs_embeds)
                    x["input_ids"] = inputs_embeds.round().long()

                    if do_tqdm:
                        iterator.set_description(f"Current loss: {loss}")

                    


            # Save computation (don't compute last loss) if not use_best
            if not use_best:
                ret = {"inputs_embeds": x["inputs_embeds"].clone().detach(),
                    "attention_mask": x["attention_mask"],
                    "labels": x["labels"]}
                return step.to_image(ret) if return_image else ret

            
            losses, _ = calc_loss({"inputs_embeds": step.to_image(x["inputs_embeds"]).to(self.model.device),
                           "attention_mask": x["attention_mask"].to(self.model.device), "labels":x["labels"].to(self.model.device)}, target.to(self.model.device))
            
            x['inputs_embeds'] = x['inputs_embeds'].detach().clone()
            best_x['inputs_embeds'] = best_x['inputs_embeds'].detach().clone()
            
            args = [losses.detach().clone(), best_loss, x, best_x]
            
            best_loss, best_x = replace_best(*args)
            
            return step.to_image(best_x) if return_image else best_x


        # Random restarts: repeat the attack and find the worst-case
        # example for each input in the batch
        if random_restarts:
            to_ret = None
            orig_cpy = {"input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "labels":labels}
            for _ in range(random_restarts):
                adv = get_adv_examples(orig_cpy)
                if to_ret is None:
                    to_ret = adv.detach()

                _, output = calc_loss(adv, target)
                corr, = helpers.accuracy(output, target, topk=(1,), exact=True)
                corr = corr.byte()
                misclass = ~corr
                to_ret["input_ids"][misclass] = adv["input_ids"][misclass]

            adv_ret = to_ret
        else:
            adv_ret = get_adv_examples(inp)

        return adv_ret

class AttackerModel(ch.nn.Module):
    """
    Wrapper class for adversarial attacks on models. Given any normal
    model (a ``ch.nn.Module`` instance), wrapping it in AttackerModel allows
    for convenient access to adversarial attacks and other applications.::

        model = ResNet50()
        model = AttackerModel(model)
        x = ch.rand(10, 3, 32, 32) # random images
        y = ch.zeros(10) # label 0
        out, new_im = model(x, y, make_adv=True) # adversarial attack
        out, new_im = model(x, y, make_adv=True, targeted=True) # targeted attack
        out = model(x) # normal inference (no label needed)

    More code examples available in the documentation for `forward`.
    For a more comprehensive overview of this class, see 
    :doc:`our detailed walkthrough <../example_usage/input_space_manipulation>`.
    """
    def __init__(self, args, model, dataset):
        super(AttackerModel, self).__init__()
        self.normalizer = self.normalize
        self.model = model
        self.attacker = Attacker(args, model, dataset)
        self.args = args



    def normalize(self, x):
        mean = x.mean(dim=0, keepdim=True)  # Compute mean along batch or feature axis
        std = x.std(dim=0, keepdim=True) + 1e-6  # Compute std, adding epsilon to avoid division by zero
        return (x - mean) / std  # Standard normalization


    def forward(self, inp, target=None, make_adv=False, with_latent=False,
                fake_relu=False, no_relu=False, with_image=True, **attacker_kwargs):
        """
        Main function for running inference and generating adversarial
        examples for a model.

        Parameters:
            inp (ch.tensor) : input to do inference on [N x input_shape] (e.g. NCHW)
            target (ch.tensor) : ignored if `make_adv == False`. Otherwise,
                labels for adversarial attack.
            make_adv (bool) : whether to make an adversarial example for
                the model. If true, returns a tuple of the form
                :samp:`(model_prediction, adv_input)` where
                :samp:`model_prediction` is a tensor with the *logits* from
                the network.
            with_latent (bool) : also return the second-last layer along
                with the logits. Output becomes of the form
                :samp:`((model_logits, model_layer), adv_input)` if
                :samp:`make_adv==True`, otherwise :samp:`(model_logits, model_layer)`.
            fake_relu (bool) : useful for activation maximization. If
                :samp:`True`, replace the ReLUs in the last layer with
                "fake ReLUs," which are ReLUs in the forwards pass but
                identity in the backwards pass (otherwise, maximizing a
                ReLU which is dead is impossible as there is no gradient).
            no_relu (bool) : If :samp:`True`, return the latent output with
                the (pre-ReLU) output of the second-last layer, instead of the
                post-ReLU output. Requires :samp:`fake_relu=False`, and has no
                visible effect without :samp:`with_latent=True`.
            with_image (bool) : if :samp:`False`, only return the model output
                (even if :samp:`make_adv == True`).

        """
        if make_adv:
            assert target is not None
            prev_training = bool(self.training)
            #self.eval()
            adv = self.attacker(inp, target, **attacker_kwargs)
            if prev_training:
                self.train()

            inp = adv

        
        normalized_inp = inp
        
        normalized_inp['inputs_embeds'] = self.normalizer(normalized_inp['inputs_embeds']).cuda()

        if no_relu and (not with_latent):
            print("WARNING: 'no_relu' has no visible effect if 'with_latent is False.")
        if no_relu and fake_relu:
            raise ValueError("Options 'no_relu' and 'fake_relu' are exclusive")

        normalized_inp['attention_mask'] = normalized_inp['attention_mask'].cuda()
        normalized_inp['labels'] = normalized_inp['labels'].cuda()

        output = self.model(**normalized_inp)
        if with_image:
            return (output, inp)
        return output
