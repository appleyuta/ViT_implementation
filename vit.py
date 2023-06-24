import torch
import torch.nn as nn
from vit_layers import VitInputLayer, VitEncoderBlock

class Vit(nn.Module):
    def __init__(
            self,
            in_channels:int=3,
            num_classes:int=10,
            emb_dim:int=384,
            num_patch_row:int=2,
            image_size:int=32,
            num_blocks:int=7,
            head:int=8,
            hidden_dim:int=384*4,
            dropout:float=0.
            ):
        """
            Args:
                in_channels: 入力画像のチャンネル数
                num_classes: 画像分類のクラス数
                emb_dim: 埋め込み後のベクトルの長さ
                num_patch_row: 1辺のパッチの数
                image_size: 入力画像の1辺の大きさ。入力画像の高さと幅は同じであると仮定
                num_blocks: Encoder Blockの数
                head: ヘッドの数
                hidden_dim: Encoder BlockのMLPにおける中間層のベクトルの長さ
                dropout: ドロップアウト率
        """
        super(Vit, self).__init__()

        # Input Layer
        self.input_layer = VitInputLayer(
            in_channels=in_channels,
            emb_dim=emb_dim,
            num_patch_row=num_patch_row,
            image_size=image_size)
        
        # Encoder (Encoder Blockをnum_blocks回重ねて定義)
        self.encoder = nn.Sequential(*[
            VitEncoderBlock(
                emb_dim=emb_dim,
                head=head,
                hidden_dim=hidden_dim,
                dropout=dropout
            )
            for _ in range(num_blocks)])
        
        # MLP Head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Args:
                x: ViTへの入力画像。(B, C, H, W)
                    B: バッチサイズ、C: チャンネル数、H: 高さ、W: 幅
            
            Returns:
                out: ViTの出力。(B, M)
                    B: バッチサイズ、M: クラス数
        """
        # Input Layer
        # (B, C, H, W) -> (B, N, D)
        # N: トークン数、 D: ベクトルの長さ
        out = self.input_layer(x)
        # Encoder
        # (B, N, D) -> (B, N, D)
        out = self.encoder(out)
        # クラストークンのみ抜き出す
        # (B, N, D) -> (B, D)
        cls_token = out[:,0]
        # MLP Head
        # (B, D) -> (B, M)
        pred = self.mlp_head(cls_token)
        return pred


if __name__ == "__main__":
    num_classes = 10
    batch_size, channel, height, width = 2, 3, 32, 32
    x = torch.randn(batch_size, channel, height, width)
    vit = Vit(in_channels=channel, num_classes=num_classes)
    pred = vit(x)

    # (2, 10)(=(B, M))になっていることを確認
    print(pred.shape)
