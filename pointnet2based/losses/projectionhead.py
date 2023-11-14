class ProjectionHead(nn.Module):
    def __init__(self, dim_in, norm_cfg, proj_dim=256, proj='convmlp'):
        super(ProjectionHead, self).__init__()
        
        if proj == 'linear':
            self.proj = nn.Conv2d(dim_in, proj_dim, kernel_size=1)
        elif proj == 'convmlp':
            self.proj = nn.Sequential(
                nn.Conv2d(dim_in, dim_in, kernel_size=1),
                build_norm_layer(norm_cfg, dim_in)[1],
                nn.ReLU(inplace=True),
                nn.Conv2d(dim_in, proj_dim, kernel_size=1)
            )
            
    def forward(self, x):
        return torch.nn.functional.normalize(self.proj(x), p=2, dim=1)