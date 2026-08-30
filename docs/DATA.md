# Data layout and policy

The official dataset contains 1,473 labeled training images and uses 1024×1024 JPG tiles. Labels follow normalized YOLO HBB format:

```text
0 x_center y_center width height
```

A convenient local layout is:

```text
data/official/
├── images/
│   ├── image_0001.jpg
│   └── ...
└── labels/
    ├── image_0001.txt
    └── ...
```

The data are not redistributed in this repository. Users must obtain them from the competition organizer. Initial- and final-round test images must only be used for final inference, never for training, validation, pseudo-labeling, or model selection unless the organizer explicitly changes the rules.
