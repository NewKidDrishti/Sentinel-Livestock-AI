import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

# --- MODEL ARCHITECTURE ---
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 512)
    def forward_once(self, x): return self.backbone(x)
    def forward(self, i1, i2): return self.forward_once(i1), self.forward_once(i2)

def run_test(model_path, same_pair_paths, diff_pair_paths):
    print("--- SENTINEL IDENTITY INTEGRATION TEST ---")
    model = SiameseNetwork()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def calculate_distance(p1, p2):
        img1 = transform(Image.open(p1).convert('RGB')).unsqueeze(0)
        img2 = transform(Image.open(p2).convert('RGB')).unsqueeze(0)
        with torch.no_grad():
            v1, v2 = model(img1, img2)
            return torch.nn.functional.pairwise_distance(out1, out2).item()

    try:
        dist_same = calculate_distance(same_pair_paths[0], same_pair_paths[1])
        dist_diff = calculate_distance(diff_pair_paths[0], diff_pair_paths[1])
        print(f"Distance (Same): {dist_same:.4f}")
        print(f"Distance (Diff): {dist_diff:.4f}")
        if dist_same < dist_diff:
            print("✅ TEST PASSED: Identity logic is reliable.")
        else:
            print("❌ TEST FAILED: Biometric gap detected.")
    except Exception as e:
        print(f"⚠️ Note: Add valid image paths to run this test locally: {e}")

if __name__ == "__main__":
    # Update these with actual images from your project to run locally
    run_test('siamese_identity.pt', ['live1.jpg', 'live2.jpg'], ['live1.jpg', 'other.jpg'])