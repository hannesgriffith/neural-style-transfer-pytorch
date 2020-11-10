import torch
from torch import nn
from torch import optim
from torchvision.models import vgg16

import numpy as np

torch.backends.cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def make_gram_matrix(features):
    B, C, H, W = features.size()
    features = features.view(B * C, H * W)
    return torch.matmul(features, torch.transpose(features, 0, 1))

class StyleTransferLoss:
    def __init__(self, target_content_features, target_style_features, content_style_ratio_weight):
        self.content_layer_of_interest = "conv4_2"
        self.style_layers_of_interest = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]

        self.target_features = target_content_features[self.content_layer_of_interest].contiguous()

        self.target_gram_matrices = {}
        for layer in self.style_layers_of_interest:
            features = target_style_features[layer]
            self.target_gram_matrices[layer] = make_gram_matrix(features)

        self.l2_loss = nn.MSELoss(reduction='sum')

        self.content_loss_weight = content_style_ratio_weight
        self.style_loss_weight = 1

    def calculate_content_loss(self, features_dict):
        features = features_dict[self.content_layer_of_interest]
        B, C, H, W = features.size()
        content_loss = self.l2_loss(features, self.target_features) / (B * C * H * W)
        return content_loss

    def calculate_style_loss(self, features_dict):
        total_style_loss = 0.

        for style_layer_of_interest in self.style_layers_of_interest:
            features = features_dict[style_layer_of_interest]
            gram_matrix = make_gram_matrix(features)
            target_gram_matrix = self.target_gram_matrices[style_layer_of_interest]

            B, C, H, W = features.size()
            style_loss = self.l2_loss(gram_matrix, target_gram_matrix) / (B * C * H * W) ** 2
            total_style_loss += style_loss

        return total_style_loss / len(self.style_layers_of_interest)

    def calculate_loss(self, features_dict):
        content_loss = self. calculate_content_loss(features_dict)
        style_loss = self.calculate_style_loss(features_dict)
        return content_loss * self.content_loss_weight + style_loss * self. style_loss_weight

class Vgg16(torch.nn.Module):
    def __init__(self):
        super().__init__()
        features = list(vgg16(pretrained=True).features.eval())
        self.features = nn.ModuleList(features)

        self.all_layers_of_interest = {
            0 : "conv1_1",
            5 : "conv2_1",
            10 : "conv3_1",
            17 : "conv4_1",
            19 : "conv4_2",
            24 : "conv5_1"
        }
        self.idxs_of_interest = list(self.all_layers_of_interest.keys())

    def forward(self, x):
        results = {}

        for i, model in enumerate(self.features):
            x = model(x)

            if i in self.idxs_of_interest:
                results[self.all_layers_of_interest[i]] = x
            
            if i >= max(self.idxs_of_interest):
                # Don't bother calculating outputs for remaining layers
                break

        return results

def post_process_output_image(output_image):
    output_image = output_image.astype(np.float32)
    output_image = np.squeeze(output_image)
    output_image = np.clip(output_image, -1, 1)
    output_image = 255 * (output_image + 1.) / 2.
    output_image = np.transpose(output_image, (1, 2, 0))
    return output_image.astype(np.uint8)

def prepare_input_image(image):
    image = image.astype(np.float32)
    image /= 255.

    # normalisation values from pytorch website for this pre-trained model
    normalisation_means = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, -1)
    normalisation_stds = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, -1)
    image = (image - normalisation_means) / normalisation_stds

    image = np.expand_dims(image, 0)
    image = np.transpose(image, (0, 3, 1, 2))
    return torch.tensor(image, dtype=torch.float, device=device)

def optimise_image(output_image, model, optimizer, loss_manager, num_iterations=100):

    for i in range(num_iterations):
        if i % 10 == 0:
            print(f"Iteration {i} / {num_iterations}")

        def closure():
            optimizer.zero_grad()
            output_image_features = model(output_image)
            loss = loss_manager.calculate_loss(output_image_features)
            loss.backward()
            return loss

        optimizer.step(closure)

    return output_image.detach().cpu().numpy()

def perform_style_transfer(content_image, style_image, hyper_parameters):
    model = Vgg16().to(device)

    content_image_torch = prepare_input_image(content_image)
    style_image_torch = prepare_input_image(style_image)

    with torch.no_grad():
        target_content_features = model(content_image_torch)
        target_style_features = model(style_image_torch)

    loss_manager = StyleTransferLoss(
        target_content_features,
        target_style_features,
        content_style_ratio_weight=hyper_parameters["content_style_ratio_weight"]
        )

    # output_image = torch.randn(
    #     content_image_torch.size(),
    #     dtype=torch.float,
    #     device=device,
    #     requires_grad=True
    #     )

    output_image = torch.clone(content_image_torch.detach()).contiguous()
    output_image.requires_grad = True

    optimizer = optim.LBFGS([output_image])

    output_image = optimise_image(
        output_image,
        model,
        optimizer,
        loss_manager,
        num_iterations=hyper_parameters["num_iterations"]
        )

    output_image_postprocessed = post_process_output_image(output_image)

    return output_image_postprocessed