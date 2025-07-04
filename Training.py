from ultralytics import YOLO
from roboflow import Roboflow

def main():
    
    #print(torch.cuda.is_available())
    #print(torch.cuda.get_device_name(0))

    model = YOLO("runs/detect/train3/weights/last.pt")

    # Download dataset from Roboflow
    rf = Roboflow(api_key="API_KEY")
    project = rf.workspace("wlrisemanticlables").project("wlri-semantic_labels")
    version = project.version(8)
    dataset = version.download("yolov12")
                
    # Training
    assistive_model = model.train(resume=True)
    print("Done model:")
    print(assistive_model)


if __name__ == '__main__':
    main()


