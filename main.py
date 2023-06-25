import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from vit import Vit
from train import train


# 学習時間短縮のため、学習データを5000枚に限定する
# 十分な計算リソースが用意できる場合は不要
class TrainDataset(Dataset):
    def __init__(self):
        super(TrainDataset, self).__init__()
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.train_dataset = datasets.MNIST(
            "./data",
            train=True,
            download=True,
            transform=transform,
        )
    
    def __getitem__(self, index):
        return self.train_dataset[index]
    
    def __len__(self):
        # return len(self.train_dataset)
        return 5000


if __name__ == "__main__":
    # 学習ハイパーパラメータ
    batch_size = 32
    in_channels = 1
    image_size = 28
    num_classes = 10
    emb_dim = 384
    num_patch_row = 2
    num_blocks = 7
    head = 8
    hidden_dim = emb_dim * 4
    dropout = 0.0
    num_epochs = 5
    learning_rate = 0.001

    # デバイス設定（使用可能GPUある場合は学習に使用される）
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # データ準備（学習、評価用）
    train_dataset = TrainDataset()
    test_dataset = datasets.MNIST(
        "./data",
        train=False,
        transform=transform,
    )

    # フルサイズデータセット
    # 計算リソースに余裕ある場合は、下記コメントアウトを外して実行
    # train_dataset = datasets.MNIST(
    #     "./data",
    #     train=True,
    #     download=True,
    #     transform=transform,
    # )


    # データローダー設定（学習、評価用）
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    # ViTモデル
    model = Vit(
        in_channels=in_channels,
        num_classes=num_classes,
        emb_dim=emb_dim,
        num_patch_row=num_patch_row,
        image_size=image_size,
        num_blocks=num_blocks,
        head=head,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )

    # 損失関数
    criterion = nn.CrossEntropyLoss()

    # 最適化アルゴリズム
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    # 学習、評価
    train(model, num_epochs, train_dataloader, test_dataloader, criterion, device)

    # モデル保存
    torch.save(model.state_dict(), "vit_model.pth")
