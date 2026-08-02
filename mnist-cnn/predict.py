import torch
from PIL import Image, ImageOps, ImageFilter
from torchvision import transforms
from main import Net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image = Image.open("my_digit.png").convert("L")

# MNIST 通常是黑色背景、白色数字
image = ImageOps.invert(image)
image = image.filter(ImageFilter.MaxFilter(3))

bbox = image.getbbox()
if bbox:
    image = image.crop(bbox)

image.thumbnail((20, 20))

canvas = Image.new("L", (28, 28), 0)
x = (28 - image.width) // 2
y = (28 - image.height) // 2
canvas.paste(image, (x, y))

image = canvas

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

image = transform(image).unsqueeze(0).to(device)

model = Net().to(device)
model.load_state_dict(
    torch.load("mnist_cnn.pt", map_location=device)
)
model.eval()

with torch.no_grad():
    output = model(image)
    prediction = output.argmax(dim=1).item()

print(f"模型预测：{prediction}")
