import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import torch.nn.functional as F
from tqdm import tqdm
import wandb

import argparse
import os
import time

from models import Encoder, DecoderBOB, DecoderEVA
from utils import generate_key, generate_key_batch, prjPaths

def get_args():
    parser = argparse.ArgumentParser(description="PyTorch Implementation of cryptogan")

    parser.add_argument('--exp_name',
                        type=str,
                        help='Name of the experiment')
    
    parser.add_argument('-o',
                        '--overwrite',
                        action='store_true',
                        help='overwright experiment folder if exp with exp_name exists')

    parser.add_argument("--run_type",
                        type=str,
                        default="train",
                        choices=["train", "inference"],
                        help="train model or load trained model for interence")
    
    parser.add_argument("--n",
                        type=int,
                        default=196,
                        help="length of plaintext (message length)")
    
    parser.add_argument("--training_steps",
                        type=int,
                        default=10000,
                        help="number of training steps")
    
    parser.add_argument("--batch_size",
                        type=int,
                        default=10000,
                        help="number training examples per (mini)batch")
    
    parser.add_argument("--learning_rate",
                        type=float,
                        default=0.001,
                        help="learning rate")
    
    parser.add_argument("--show_every_n_steps",
                        type=int,
                        default=10,
                        help="during training print output to cli every n steps")
    
    parser.add_argument("--checkpoint_every_n_steps",
                        type=int,
                        default=1000,
                        help="checkpoint model files during training every n epochs")
    
    parser.add_argument("--verbose",
                        type=bool,
                        default=False,
                        help="during training print model outputs to cli")
    
    parser.add_argument("--clip_value",
                        type=float,
                        default=1,
                        help="maximum allowed value of the gradients in range(-clip_value, clip_value)")

    args = parser.parse_args()
    return args


def train(
        train_loader,
        gpu_available,
        prjPaths,
        training_steps,
        learning_rate,
        show_every_n_steps,
        checkpoint_every_n_steps,
        clip_value,
        key_size,
        aggregated_losses_every_n_steps=32):

    alice = Encoder()
    bob = DecoderBOB()
    eve = DecoderEVA()

    wandb.config.update({
        "alice_size": sum(p.numel() for p in alice.parameters()),
        "bob_size": sum(p.numel() for p in bob.parameters()),
        "eve_size": sum(p.numel() for p in eve.parameters())
    })

    alice.train()
    bob.train()
    eve.train()

    if gpu_available:
        alice.cuda()
        bob.cuda()
        eve.cuda()

    aggregated_losses = {
            "alice_bob_training_loss": [],
            "bob_reconstruction_training_errors": [],
            "eve_reconstruction_training_errors": [],
            "step": []
    }

    optimizer_alice = Adam(params=alice.parameters(), lr=learning_rate)
    optimizer_bob = Adam(params=bob.parameters(), lr=learning_rate)
    optimizer_eve = Adam(params=eve.parameters(), lr=learning_rate)

    # define losses 
    bob_reconstruction_error = nn.L1Loss()
    eve_reconstruction_error = nn.L1Loss()

    for epoch in tqdm(range(training_steps)):
        tic = time.time()
        for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
            data = data.to('cuda')
            # Training alternates between Alice/Bob and Eve
            for network, num_minibatches in {"alice_bob": 1, "eve": 2}.items():
                """ 
                Alice/Bob training for one minibatch, and then Eve training for two minibatches this ratio 
                in order to give a slight computational edge to the adversary Eve without training it so much
                that it becomes excessively specific to the exact current parameters of Alice and Bob
                """
                for _ in range(num_minibatches):

                    k = generate_key_batch(size=key_size, batchsize=data.shape[0], gpu_available=gpu_available)

                    # forward pass through alice and eve networks
                    
                    ciphertext = alice.forward(data, k)

                    eve_p = eve.forward(ciphertext)

                    if network == "alice_bob":

                        # forward pass through bob network
                        bob_p = bob.forward(ciphertext, k)

                        # calculate errors
                        error_bob = bob_reconstruction_error(input=bob_p, target=data)
                        error_eve = eve_reconstruction_error(input=eve_p, target=data)
                        # alice_bob_loss =  error_bob + F.relu(1 - error_eve)
                        alice_bob_loss = error_bob - error_eve

                            

                        # Zero gradients, perform a backward pass, clip gradients, and update the weights.
                        optimizer_alice.zero_grad()
                        optimizer_bob.zero_grad()
                        alice_bob_loss.backward()
                        nn.utils.clip_grad_value_(alice.parameters(), clip_value)
                        nn.utils.clip_grad_value_(bob.parameters(), clip_value)
                        optimizer_alice.step()
                        optimizer_bob.step()

                    elif network == "eve":

                        # calculate error
                        error_eve = eve_reconstruction_error(input=eve_p, target=data)

                        # Zero gradients, perform a backward pass, and update the weights
                        optimizer_eve.zero_grad()
                        error_eve.backward()
                        nn.utils.clip_grad_value_(eve.parameters(), clip_value)
                        optimizer_eve.step()

        # end time time for step
        time_elapsed = time.time() - tic

        if epoch % aggregated_losses_every_n_steps == 0:
            # aggregate min training errors for bob and eve networks
            aggregated_losses["alice_bob_training_loss"].append(alice_bob_loss.cpu().detach().numpy().tolist())
            aggregated_losses["bob_reconstruction_training_errors"].append(error_bob.cpu().detach().numpy().tolist())
            aggregated_losses["eve_reconstruction_training_errors"].append(error_eve.cpu().detach().numpy().tolist())
            aggregated_losses["step"].append(epoch)
        
        if epoch % show_every_n_steps == 0:
            print("Total_Steps: %i of %i || Time_Elapsed_Per_Step: (%.3f sec/step) || Bob_Alice_Loss: %.5f || Bob_Reconstruction_Error: %.5f || Eve_Reconstruction_Error: %.5f" % (epoch,
                                                                                                                                                                                training_steps,
                                                                                                                                                                                time_elapsed,
                                                                                                                                                                                aggregated_losses["alice_bob_training_loss"][-1],
                                                                                                                                                                                aggregated_losses["bob_reconstruction_training_errors"][-1],
                                                                                                                                                                                aggregated_losses["eve_reconstruction_training_errors"][-1]))
            wandb.log({
                "Epoch": epoch,
                "Time_Elapsed_Per_Step": time_elapsed, 
                "Bob_Alice_Loss": aggregated_losses["alice_bob_training_loss"][-1],
                "Bob_Reconstruction_Error": aggregated_losses["bob_reconstruction_training_errors"][-1],
                "Eve_Reconstruction_Error": aggregated_losses["eve_reconstruction_training_errors"][-1]
            })
            image_array = make_grid(nrow=3, ncols=3, pad_value=2, tensor=[*data[:3], *bob_p[:3], *eve_p[:3]])
            images = wandb.Image(
                image_array, 
                caption="Top: Ground truth, Middle: Bob reconstruction, Bottom: Eve reconstruction"
                )
            wandb.log({"examples": images})

        if epoch % checkpoint_every_n_steps == 0 and epoch != 0:
            print("checkpointing models...\n")
            torch.save(alice.state_dict(), os.path.join(prjPaths.CHECKPOINT_DIR, f"alice_epoch_{epoch}.pth"))
            torch.save(bob.state_dict(), os.path.join(prjPaths.CHECKPOINT_DIR, f"bob_epoch_{epoch}.pth"))
            torch.save(eve.state_dict(), os.path.join(prjPaths.CHECKPOINT_DIR, f"eve_epoch_{epoch}.pth"))


def main():
    args = get_args()
    prjPaths_ = prjPaths(exp_name = args.exp_name, overwrite = args.overwrite)
    run = wandb.init(
        dir=prjPaths,
        project="gun_crypto_system",
        name=args.exp_name,
        config={
            "n": args.n,
            "training_steps": args.training_steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "clip_value": args.clip_value,
    })

    if torch.cuda.device_count() > 0:
        gpu_available = True
    else:
        gpu_available = False

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset1 = datasets.MNIST('/homes/roma5okolow/gun_crypto_system/data',
                    train=True, download=True,
                    transform=transform)

    dataset2 = datasets.MNIST('/homes/roma5okolow/gun_crypto_system/data',
                    train=False, download=True,
                    transform=transform)
    
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.batch_size}

    train_loader = torch.utils.data.DataLoader(dataset1, drop_last=True, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, drop_last=True, **test_kwargs)

    if args.run_type == "train":
        train(
            train_loader=train_loader,
            gpu_available=gpu_available,
            prjPaths=prjPaths_,
            training_steps=args.training_steps,
            learning_rate=args.learning_rate,
            show_every_n_steps=args.show_every_n_steps,
            checkpoint_every_n_steps=args.checkpoint_every_n_steps,
            clip_value=args.clip_value,
            key_size=args.n)
    # elif args.run_type == "inference":
    #     inference(gpu_available, prjPaths=prjPaths_)

if __name__ == "__main__":
    main()


# def inference(gpu_available, prjPaths):
#     alice = Encoder()
#     bob = DecoderBOB()
#     eve = DecoderEVA()


#     # restoring persisted networks
#     print("restoring Alice, Bob, and Eve networks...\n")
#     alice.load_state_dict(torch.load(os.path.join(prjPaths.CHECKPOINT_DIR, "alice.pth")))
#     bob.load_state_dict(torch.load(os.path.join(prjPaths.CHECKPOINT_DIR, "bob.pth")))
#     eve.load_state_dict(torch.load(os.path.join(prjPaths.CHECKPOINT_DIR, "eve.pth")))

#     # specify that model is currently in training mode
#     alice.eval()
#     bob.eval()
#     eve.eval()

#     # if gpu available then run inference on gpu
#     if gpu_available:
#         alice.cuda()
#         bob.cuda()
#         eve.cuda()

#     convert_tensor_to_list_and_scale = lambda tensor: list(map(lambda x: int((round(x)+1)/2), tensor.cpu().detach().numpy().tolist()))

#     while True:

#         p_utf_8 = input("enter plaintext: ")

#         # ensure that p is correct length else pad with spaces
#         while not ((len(p_utf_8) * NUM_BITS_PER_BYTE) % img_dim == 0):
#             p_utf_8 = p_utf_8 + " "

#         # convert p UTF-8 -> Binary
#         p_bs = UTF_8_to_binary(p_utf_8)

#         print("plaintext ({}) in binary: {}".format(p_utf_8, p_bs))

#         # group Binary p into groups that are valid with input layer of network
#         p_bs = [np.asarray(list(p_bs[i-1]+p_bs[i]), dtype=np.float32) for i, p_b in enumerate(p_bs) if ((i-1) * NUM_BITS_PER_BYTE) % img_dim == 0]

#         eve_ps_b = []
#         bob_ps_b = []
#         for p_b in p_bs:

#             # generate k
#             _, k = generate_data(gpu_available=gpu_available, batch_size=1, n=img_dim)
#             p_b = torch.unsqueeze(torch.from_numpy(p_b)*2-1, 0)

#             if gpu_available:
#                 p_b = p_b.cuda()

#             # run forward pass through networks
#             alice_c = torch.unsqueeze(alice.forward(torch.cat((p_b, k), 1).float()), 0)
#             eve_p = convert_tensor_to_list_and_scale(eve.forward(alice_c))
#             bob_p = convert_tensor_to_list_and_scale(bob.forward(torch.cat((alice_c, k), 1).float()))

#             eve_ps_b.append("".join(list(map(str, eve_p))))
#             bob_ps_b.append("".join(list(map(str, bob_p))))
        
#         print("eve_ps_b:                     {}".format(list(itertools.chain.from_iterable([[i[:8], i[8:]]  for i in eve_ps_b]))))
#         print("bob_ps_b:                     {}\n".format(list(itertools.chain.from_iterable([[i[:8], i[8:]]  for i in bob_ps_b]))))
