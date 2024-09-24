
prelabeling_map = {
    "demo": {
        "Task_type": "os",
        "Weight_type": "onnx",    # 模型类型，支持onnx、caffemodel
        "Model_type": "yolov5",   # 模型类型
        "Num_classes": 3,         # 模型类别数
        "Score_thr": 0.3,         # 置信度阈值
        "Box_thr": 0.65,          # iou阈值
        "Fixed_scale": 1,         # 数据预处理方式
        "Color_space": "rgb",     # 输入颜色空间
        "Anchors": [[(9, 3), (6, 16), (26, 8)], [(15, 40), (32, 73), (63, 130)], [(91, 99), (190, 182),(339, 276)]], # 模型anchors，没有则填None
        "Used_classes": ["class1", "class3"], # 使用了这个模型中哪几个类别
        "Class_index": [0, 2],                # 使用的类别在classes中的索引
        "Parent": [None, None]                # 使用的类别的父级
    },
    "demo": {
        "Task_type": "os",
        "Weight_type": "onnx",
        "Num_classes": 81,             # 模型类别数
        "Fixed_scale": 1,              # 数据预处理方式
        "Color_space": "rgb",          # 输入颜色空间
        "Used_classes": ["person"],
        "Class_index": [1]
    }
}