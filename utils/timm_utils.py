import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# # import timm

# # # 列出所有可用的模型名称
# # model_names = timm.list_models(pretrained=True)  # 只显示有预训练权重的模型
# # print(model_names)


import timm

# 加载模型
model_name = 'vit_tiny_patch16_224'
model = timm.create_model(model_name, pretrained=True)

# 打印模型配置
print(f"Model '{model_name}' configuration: {model.default_cfg}")

'''
Model 'vit_tiny_patch16_224' configuration: {'url': 'https://storage.googleapis.com/vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz', 'hf_hub_id': 'timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k', 'architecture': 'vit_tiny_patch16_224', 'tag': 'augreg_in21k_ft_in1k', 'custom_load': True, 'input_size': (3, 224, 224), 'fixed_input_size': True, 'interpolation': 'bicubic', 'crop_pct': 0.9, 'crop_mode': 'center', 'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5), 'num_classes': 1000, 'pool_size': None, 'first_conv': 'patch_embed.proj', 'classifier': 'head'}
'''
