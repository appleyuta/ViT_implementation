import torch

def evaluate(
        model,
        test_dataloader,
        criterion,
        device
    ):
    """
        pytorchで定義したモデルを評価する関数

        Args:
            model: 評価するモデル
            test_dataloader: データローダー（評価用）
            criterion: 損失関数
            device: 使用デバイス
        Returns:
            None
    """
    model.eval()

    loss_sum = 0
    correct = 0

    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            loss_sum += criterion(outputs, labels)

            pred = outputs.argmax(1)
            correct += pred.eq(labels.view_as(pred)).sum().item()
        
    print(f"Eval Loss: {loss_sum.item() / len(test_dataloader)}, Accuracy: {correct / len(test_dataloader.dataset)} ({correct} / {len(test_dataloader.dataset)})")
