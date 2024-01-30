import torch
import final_model

def load_model_checkpoint(path):
    checkpoint = torch.load(path)

    model = final_model.Classifier(checkpoint["input"])

    model.load_state_dict(checkpoint['state_dict'])

    return model

model = load_model_checkpoint('checkpoint.pth')


example = torch.tensor([[0.0606, 0.5000, 0.3333, 0.4828, 0.4000, 0.4000, 0.4000, 0.4000, 0.4000,
        0.4000, 0.1651, 0.0869, 0.0980, 0.1825, 0.1054, 0.2807, 0.0016, 0.0000,
        0.0033, 0.0027, 0.0031, 0.0021]]).float()

# 7. Perform a prediction by inputting the following tensor into your model:
print(example.shape)

pred = model(example)
print(pred)
# tensor([[-2.5711, -0.0795]], grad_fn=<LogSoftmaxBackward>)
pred = torch.exp(pred)
print(pred)
# tensor([[0.0764, 0.9236]], grad_fn=<ExpBackward>)

top_p, top_class_test = pred.topk(1, dim=1)
print(top_class_test)
# tensor([[1]])

traced_script = torch.jit.trace(model, example, check_trace=False)

prediction = traced_script(example)
prediction = torch.exp(prediction)
top_p_2, top_class_test_2 = prediction.topk(1, dim=1)
print(top_class_test_2)



