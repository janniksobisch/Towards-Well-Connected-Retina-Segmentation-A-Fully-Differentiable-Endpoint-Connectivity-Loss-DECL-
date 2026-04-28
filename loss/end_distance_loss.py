import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftSkeletonize(torch.nn.Module):
    def __init__(self, num_iter=40):
        super(SoftSkeletonize, self).__init__()
        self.num_iter = num_iter

    def soft_erode(self, img):
        img = img.float()
        if len(img.shape)==4:
            p1 = -F.max_pool2d(-img, (3,1), (1,1), (1,0))
            p2 = -F.max_pool2d(-img, (1,3), (1,1), (0,1))
            return torch.min(p1,p2)
        elif len(img.shape)==5:
            p1 = -F.max_pool3d(-img,(3,1,1),(1,1,1),(1,0,0))
            p2 = -F.max_pool3d(-img,(1,3,1),(1,1,1),(0,1,0))
            p3 = -F.max_pool3d(-img,(1,1,3),(1,1,1),(0,0,1))
            return torch.min(torch.min(p1, p2), p3)

    def soft_dilate(self, img):
        if len(img.shape)==4:
            return F.max_pool2d(img, (3,3), (1,1), (1,1))
        elif len(img.shape)==5:
            return F.max_pool3d(img,(3,3,3),(1,1,1),(1,1,1))

    def soft_open(self, img):
        return self.soft_dilate(self.soft_erode(img))

    def soft_skel(self, img):
        img1 = self.soft_open(img)
        skel = F.relu(img-img1)

        for j in range(self.num_iter):
            img = self.soft_erode(img)
            img1 = self.soft_open(img)
            delta = F.relu(img-img1)
            skel = skel + F.relu(delta - skel * delta)

        return skel

    def forward(self, img):
        return self.soft_skel(img)

def soft_dice(y_true, y_pred):
    """[function to compute dice loss]

    Args:
        y_true ([float32]): [ground truth image]
        y_pred ([float32]): [predicted image]

    Returns:
        [float32]: [loss value]
    """
    smooth = 1
    intersection = torch.sum((y_true * y_pred))
    coeff = (2. *  intersection + smooth) / (torch.sum(y_true) + torch.sum(y_pred) + smooth)
    return (1. - coeff)

class EndpointDistanceLossAverage(nn.Module):
    """
    A dimension-agnostic loss function that promotes connectivity.
    Includes options for ablation studies.
    """
    def __init__(self, 
                 tau: float = 1.0, 
                 lambda_count: float = 1.0, 
                 alpha: float = 0.85,
                 use_mismatch_term: bool = True,
                 gamma: float = 1.0):
        super(EndpointDistanceLossAverage, self).__init__()
        self.tau = tau
        self.lambda_count = lambda_count
        self.alpha = alpha
        
        # Gamma controls the sharpness of the neighbor sum match.
        # Setting this to e.g., 5.0 suppresses non-endpoint line leakage (sum=12) into the count.
        # future directions might want to look into also accounting for bifurcations (sum>12) as these will be toned down in current iteration.
        #
        self.gamma = gamma
        self.soft_skeletonize = SoftSkeletonize()
        
        self.use_mismatch_term = use_mismatch_term

    def _get_soft_endpoints(self, skel_map: torch.Tensor) -> torch.Tensor:
        is_3d = len(skel_map.shape) == 5
        if is_3d:
            kernel_weights = torch.ones((3, 3, 3), device=skel_map.device, dtype=skel_map.dtype)
            kernel_weights[1, 1, 1] = 10
            conv_fn = F.conv3d
        else:
            kernel_weights = torch.ones((3, 3), device=skel_map.device, dtype=skel_map.dtype)
            kernel_weights[1, 1] = 10
            conv_fn = F.conv2d
        kernel = kernel_weights.unsqueeze(0).unsqueeze(0)
        neighbor_sum = conv_fn(skel_map, kernel, padding=1)
        target_val = torch.tensor(11.0, device=skel_map.device, dtype=skel_map.dtype)
        
        # Gamma scales the gaussian penalty. Higher gamma (e.g., 5.0) heavily penalizes line segments
        # (which sum to 12) from bleeding into the endpoint map.
        endpoint_map = torch.exp(-self.gamma * torch.pow(neighbor_sum - target_val, 2))
        return endpoint_map * skel_map

    def _get_weighted_coordinates(self, endpoint_map: torch.Tensor) -> torch.Tensor:
        batch_size = endpoint_map.shape[0]
        is_3d = len(endpoint_map.shape) == 5
        if is_3d:
            _, _, d, h, w = endpoint_map.shape
            z_coords, y_coords, x_coords = torch.meshgrid(torch.arange(d, device=endpoint_map.device), torch.arange(h, device=endpoint_map.device), torch.arange(w, device=endpoint_map.device), indexing='ij')
            coords = torch.stack([z_coords, y_coords, x_coords], dim=0)
        else:
            _, _, h, w = endpoint_map.shape
            y_coords, x_coords = torch.meshgrid(torch.arange(h, device=endpoint_map.device), torch.arange(w, device=endpoint_map.device), indexing='ij')
            coords = torch.stack([y_coords, x_coords], dim=0)
        coords = coords.to(endpoint_map.dtype)
        coords = coords.unsqueeze(0).expand(batch_size, -1, *coords.shape[1:])
        total_weight = torch.sum(endpoint_map, dim=list(range(1, endpoint_map.dim())), keepdim=True) + 1e-8
        weights = endpoint_map / total_weight
        product = weights * coords
        sum_dims = list(range(2, product.dim()))
        weighted_coords = torch.sum(product, dim=sum_dims)
        return weighted_coords

    def forward(self, network_output: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        is_3d = len(network_output.shape) == 5
        y_pred_prob = F.softmax(network_output, dim=1)
        pred_prob_fg = y_pred_prob[:, 1:2]
        y_true_float = y_true.float()

        skel_pred = self.soft_skeletonize(pred_prob_fg)
        skel_true = self.soft_skeletonize(y_true_float.to(pred_prob_fg.dtype))

        endpoint_map_pred = self._get_soft_endpoints(skel_pred)
        endpoint_map_true = self._get_soft_endpoints(skel_true)
        
        unnormalized_distance_loss = torch.tensor(0.0, device=network_output.device)
        if self.use_mismatch_term:
            pred_coords = self._get_weighted_coordinates(endpoint_map_pred)
            true_coords = self._get_weighted_coordinates(endpoint_map_true)
            # Calculate the Euclidean distance between the average coordinates
            distance_vector = torch.sqrt(torch.sum((pred_coords - true_coords)**2, dim=1))
            unnormalized_distance_loss = torch.mean(distance_vector)
        
        if is_3d:
            _, _, d, h, w = network_output.shape
            image_diagonal = math.sqrt(d**2 + h**2 + w**2)
        else:
            _, _, h, w = network_output.shape
            image_diagonal = math.sqrt(h**2 + w**2)
            
        distance_loss = unnormalized_distance_loss / (image_diagonal * self.tau + 1e-8)
        
        sum_dims = (1, 2, 3, 4) if is_3d else (1, 2, 3)
        num_pred_endpoints = torch.sum(endpoint_map_pred, dim=sum_dims)
        num_true_endpoints = torch.sum(endpoint_map_true, dim=sum_dims)
        
        abs_diff = torch.abs(num_pred_endpoints - num_true_endpoints)
        total_endpoints = num_pred_endpoints + num_true_endpoints
        count_penalty = torch.mean(abs_diff / (total_endpoints + 1e-8))
        
        final_endpoint_loss = distance_loss + self.lambda_count * count_penalty

        dice = soft_dice(y_true_float, pred_prob_fg)

        final_loss = self.alpha * dice + (1 - self.alpha) * final_endpoint_loss

        return final_loss
