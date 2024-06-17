from isegm.utils.serialization import serialize
from .is_model import ISModel
from .modeling.swin_transformer import SwinTransformer_cross_encdecv4, SwinTransfomerSegHead

class SwinformerModel(ISModel):
    @serialize
    def __init__(
        self,
        backbone_params={}, 
        head_params={},
        **kwargs
        ):

        super().__init__(**kwargs)


        self.backbone = SwinTransformer_cross_encdecv4(**backbone_params)
        # self.backbone = SwinTransformer_cross_fusion(**backbone_params)
        # self.backbone = SwinTransformer(**backbone_params)
        self.head = SwinTransfomerSegHead(**head_params)

    def backbone_forward(self, image, coord_features=None):
        backbone_features = self.backbone.forward_backbone(image, coord_features)
        # backbone_features = self.backbone(image, coord_features)
        return {'instances': self.head(backbone_features), 'instances_aux': None}