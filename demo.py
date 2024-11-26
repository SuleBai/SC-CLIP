from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from scclip_segmentor import SCCLIPForSegmentation

img = Image.open('figs/demo.jpg')
name_list = ['skiiing man', 'tree with snow', 'sky', 'snow']

with open('my_name.txt', 'w') as writers:
    for i in range(len(name_list)):
        if i == len(name_list)-1:
            writers.write(name_list[i])
        else:
            writers.write(name_list[i] + '\n')
writers.close()

img_tensor = transforms.Compose([
    transforms.Lambda(lambda img: img.convert('RGB')),
    transforms.ToTensor(),
    transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
])(img)

img_tensor = img_tensor.unsqueeze(0).cuda()

model = SCCLIPForSegmentation(
    clip_path='ViT-B/16',
    name_path='my_name.txt',
    pamr_steps=0,
    pamr_stride=(8, 16),
    slide_crop=224,
    slide_stride=112
)

seg_pred = model.predict(img_tensor, data_samples=None)
seg_pred = seg_pred.data.cpu().numpy().squeeze(0)

fig, ax = plt.subplots(1, 3, figsize=(18, 6))
ax[0].imshow(img)
ax[0].axis('off')
ax[1].imshow(seg_pred, cmap='viridis')
ax[1].axis('off')
ax[2].imshow(img)
ax[2].axis('off')
ax[2].imshow(seg_pred, cmap='viridis', alpha=0.8)
plt.tight_layout()
plt.savefig('seg_ours.png', bbox_inches='tight')