import graphviz

dot = graphviz.Digraph('round-table', comment='The Round Table')  


print(dot)


(6): Sequential(


    (0): XResNetBlock(
      (convs): Sequential(
        (0): Sequential(
          (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): Mish_layer()
        )
        (1): Sequential(
          (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (id_conv): Sequential(
        (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (pool): AvgPool2d(kernel_size=2, stride=2, padding=0)
      (activation): Mish_layer()
    )

    
    (1): XResNetBlock(
      (convs): Sequential(
        (0): Sequential(
          (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): Mish_layer()
        )
        (1): Sequential(
          (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (id_conv): Identity()
      (pool): Identity()
      (activation): Mish_layer()
    )
  )