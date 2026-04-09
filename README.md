Návod na spustenie v konzoli:

python {názov benchmarku} --models {názov modelu} --train-count {veľkosť trénovacej množiny} --val-count {veľkosť validačnej množiny} --test-count {veľkosť testovacej množiny}


models: sift, hog, vit_b_16, resnet50

V prípade ViT-B/16 a ResNet50 za --models pridať aj --epochs {počet epoch}
