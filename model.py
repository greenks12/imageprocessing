from ultralytics import YOLO

def main():
    yaml_path = r"F:\Work\Project\car-dataset\dataset.yaml"

    model = YOLO("yolo11s.pt")

    results = model.train(
        data=yaml_path,
        epochs=50,
        batch=8,
        imgsz=640,
        workers=0,     
        lr0=0.001,
        lrf=0.01,
        seed=0,
        dropout=0.2,
        iou=0.6,
        optimizer="SGD",
    )

    model.save('yolov11_trained.pt')
    print("Model trained and saved successfully!")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
