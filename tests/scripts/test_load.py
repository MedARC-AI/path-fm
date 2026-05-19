import torch
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')#, force_reload = True)

checkpoint = torch.load("./output/eval/training_12499/teacher_checkpoint.pth")


if "teacher" in checkpoint:
    print("this is a dino teacher model")
    checkpoint = checkpoint["teacher"]
            #Need to remove the word backbone from everything I think?
    checkpoint_new = {}
    for key in list(checkpoint.keys()):
        if "dino" in str(key) or "ibot" in str(key):
            checkpoint.pop(key, None)

    for key, keyb in zip(checkpoint.keys(), model.state_dict().keys()):
        print(key, keyb)
        checkpoint_new[keyb] = checkpoint[key]

    checkpoint = checkpoint_new
    #Manually change pos_embed shape.
    new_shape = checkpoint["pos_embed"]
    model.pos_embed = torch.nn.parameter.Parameter(new_shape)



model.load_state_dict(checkpoint, strict=True)


