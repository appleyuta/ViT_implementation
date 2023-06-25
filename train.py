import torch
from eval import evaluate

def train(
        model,
        num_epochs,
        train_dataloader,
        test_dataloader,
        criterion,
        device
    ):
    """
        pytorchで定義したモデルを学習させる関数

        Args:
            model: 学習させるモデル
            num_epochs: 学習回数
            train_dataloader: データローダー（学習用）
            test_dataloader: データローダー（評価用）
            criterion: 損失関数
            device: 使用デバイス
        Returns:
            None
    """
    optimizer = torch.optim.Adam(model.parameters())

    model.train()

    for epoch in range(num_epochs):
        loss_sum = 0

        for inputs, labels in train_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss_sum += loss

            loss.backward()

            optimizer.step()
        
        print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {loss_sum.item() / len(train_dataloader)}")

        evaluate(model, test_dataloader, criterion, device)
