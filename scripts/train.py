from rfdetr import RFDETRSmall

model = RFDETRSmall()

model.train(
    dataset_dir="dataset",
    epochs=200,
    batch_size=24,
    grad_accum_steps=4,
    lr=1e-4,
    output_dir="result",
    world_size=2,
    resume="result/checkpoint0119.pth"
)
