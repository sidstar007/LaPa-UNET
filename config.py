image_h = 512
image_w = 512
num_classes = 5 # Change this field based on number of facial parts you want to finetune your model on

# Please add or remove colors based on the number of classes chosen, [0, 0, 0] is for background
rgb_codes = [
    [0, 0, 0],
    [0, 153, 255],
    [102, 255, 153],
]

# You may modify this list based on your choice of facial parts
classes = [
    "background", "left eyebrow", "right eyebrow", "left eye", "right eye"
]

model_dir = "/sample_model_dir"
dataset_path = "/sample_data_set_dir"

# Change as per your choice
batch_size = 8