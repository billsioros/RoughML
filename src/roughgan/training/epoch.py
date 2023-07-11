import logging
import time

import torch

logger = logging.getLogger(__name__)


def per_epoch(
    generator,
    discriminator,
    dataloader,
    optimizer_generator,
    optimizer_discriminator,
    criterion,
    content_loss_fn=None,
    vector_content_loss_fn=None,
    loss_weights=None,
    log_every_n=None,
):
    generator.train()

    if content_loss_fn is None:
        content_loss_weight, criterion_weight = 0, 1
    else:
        content_loss_weight, criterion_weight = loss_weights

    content_loss, vector_content_loss = 0, 0
    generator_loss, discriminator_loss = 0, 0
    discriminator_output_real, discriminator_output_fake = 0, 0

    start_time = time.time()
    for train_iteration, X_batch in enumerate(dataloader):
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ## Train with all-real batch
        discriminator.zero_grad()
        # Format batch
        label = torch.full((X_batch.size(0),), 1, dtype=X_batch.dtype, device=X_batch.device)
        # Forward pass real batch through D
        output = discriminator(X_batch).view(-1)
        # Calculate loss on all-real batch
        discriminator_error_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        discriminator_error_real.backward()
        discriminator_output_real_batch = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(
            X_batch.size(0),
            *generator.feature_dims,
            dtype=X_batch.dtype,
            device=X_batch.device,
        )
        # Generate fake image batch with G
        fake = generator(noise)
        label.fill_(0)
        # Classify all fake batch with D
        output = discriminator(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        discriminator_error_fake = criterion(output, label)
        # Calculate the gradients for this batch
        discriminator_error_fake.backward()
        # Add the gradients from the all-real and all-fake batches
        discriminator_error_total = discriminator_error_real + discriminator_error_fake
        # Update D
        optimizer_discriminator.step()

        # (2) Update G network: maximize log(D(G(z)))
        generator.zero_grad()
        label.fill_(1)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = discriminator(fake).view(-1)
        # Calculate G's loss based on this output
        if content_loss_weight <= 0:
            discriminator_error_fake = criterion(output, label)
        else:
            generator_content_loss = content_loss_fn(fake.cpu().detach().numpy().squeeze())
            generator_content_loss = torch.mean(generator_content_loss).to(fake.device)

            content_loss += generator_content_loss.item() / len(dataloader)

            generator_vector_content_loss = vector_content_loss_fn(
                fake.cpu().detach().numpy().squeeze(),
            )
            generator_vector_content_loss = torch.mean(generator_vector_content_loss).to(
                fake.device,
            )

            vector_content_loss += generator_vector_content_loss.item() / len(dataloader)

            discriminator_error_fake = criterion(output, label) / (0.5 + content_loss)
        # Calculate gradients for G, which propagate through the discriminator
        discriminator_error_fake.backward()
        discriminator_output_fake_batch = output.mean().item()
        # Update G
        optimizer_generator.step()

        generator_loss += discriminator_error_fake.item() / len(dataloader)
        discriminator_loss += discriminator_error_total.item() / len(dataloader)
        discriminator_output_real += discriminator_output_real_batch / len(dataloader)
        discriminator_output_fake += discriminator_output_fake_batch / len(dataloader)

        if log_every_n is not None and not train_iteration % log_every_n:
            logger.info(
                "Training Iteration #%04d ended after %7.3f seconds",
                train_iteration,
                time.time() - start_time,
            )

            start_time = time.time()

    return (
        generator_loss,
        discriminator_loss,
        discriminator_output_real,
        discriminator_output_fake,
        content_loss,
        vector_content_loss,
    )
