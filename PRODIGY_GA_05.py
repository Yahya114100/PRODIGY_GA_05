import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
from google.colab import files
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import io

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define the neural network architecture
class VGG(nn.Module):
    def _init_(self, features, content_layers, style_layers):
        super(VGG, self)._init_()
        self.features = features
        self.content_layers = content_layers
        self.style_layers = style_layers

    def forward(self, x):
        content_outputs = []
        style_outputs = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.content_layers:
                content_outputs.append(x)
            if i in self.style_layers:
                style_outputs.append(x)
        return content_outputs, style_outputs


# Load the VGG-19 model
content_layers = [21]  # relu4_2
# relu1_1, relu2_1, relu3_1, relu4_1, relu5_1
style_layers = [0, 5, 10, 19, 28]
vgg19 = VGG(models.vgg19(pretrained=True).features,
            content_layers, style_layers).to(device)


# Define the content and style loss functions
def gram_matrix(input):
    batch_size, channels, height, width = input.size()
    features = input.view(batch_size, channels, height * width)
    # Batch matrix multiplication
    G = torch.bmm(features, features.transpose(1, 2))
    return G.div(channels * height * width)


def optimize(content_img, style_img, target_img, vgg, optimizer, style_weight=1e4, content_weight=1, num_steps=50):
    """Optimize the target image with adjusted parameters for faster execution."""
    target_img.requires_grad_(True)

    for step in range(num_steps):
        def closure():
            optimizer.zero_grad()

            # Forward pass
            content_features, _ = vgg(content_img)
            target_content_features, target_style_features = vgg(target_img)
            _, style_features = vgg(style_img)

            # Compute content loss
            content_loss_value = sum(
                F.mse_loss(tc, cc) for tc, cc in zip(target_content_features, content_features)
            )

            # Compute style loss
            style_loss_value = sum(
                F.mse_loss(gram_matrix(ts), gram_matrix(ss))
                for ts, ss in zip(target_style_features, style_features)
            )

            # Total loss
            total_loss = content_weight * content_loss_value + style_weight * style_loss_value

            # Backward pass
            total_loss.backward()
            return total_loss

        # Perform one optimization step
        loss = optimizer.step(closure)

        # Print progress every 20 steps
        if step % 20 == 0 or step == num_steps - 1:
            print(f"Step {step}/{num_steps}, Loss: {loss.item():.4f}")

    return target_img


# Load and preprocess the images
uploaded = files.upload()
iterator = iter(uploaded)
content_img = Image.open(io.BytesIO(uploaded[next(iterator)])).convert("RGB")
style_img = Image.open(io.BytesIO(uploaded[next(iterator)])).convert("RGB")

preprocess = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize to consistent size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

content_img = preprocess(content_img).unsqueeze(0).to(device)
style_img = preprocess(style_img).unsqueeze(0).to(device)
target_img = content_img.clone().requires_grad_(True)

# Set up the optimizer
optimizer = optimizer = optim.Adam([target_img], lr=0.05)

# Run the optimization
output_img = optimize(content_img, style_img, target_img, vgg19, optimizer)

# Save and display the output image
output_img = output_img.squeeze(0).cpu().clamp(0, 1).detach().numpy()
output_img = output_img.transpose(1, 2, 0)
output_img = (output_img * 255).astype(np.uint8)
output_img = Image.fromarray(output_img)
output_img.save("output.jpg")

plt.imshow(output_img)
plt.title("Stylized Output Image")
plt.axis("off")
plt.show()
