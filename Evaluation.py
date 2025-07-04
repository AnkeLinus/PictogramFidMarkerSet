from ultralytics import YOLO
import os
import time
import glob

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


def measure_inference_time(model, images_path):
    """
    Code of this function originally from https://github.com/katrinmisel/assistive_training/blob/main/training.ipynb
    Measure inference time for a YOLO model on a given dataset.

    Parameters:
        model (YOLO): The YOLO model object.
        images_path (str): Path to the directory containing validation images.
        labels_path (str): Path to the directory containing labels (optional).
    
    Returns:
        dict: Dictionary containing total inference time, average inference time per image,
              average inference time per class, and average inference time over all classes.
    """
    # Define class names (you may pass this dynamically if needed)
    classes = ["Task-Brush", "Task-Drink", "Task-Eat", "Task-Fill-Cup", "Task-Pick", "Task-Place", "Task-Scratch", "Task-Switch"]

    # Collect all image files
    image_files = glob.glob(os.path.join(images_path, "*.jpg"))  # Adjust extension if needed (e.g., .png)

    # Initialize metrics
    inference_times = []
    class_inference_times = {cls: [] for cls in classes}

    # Run inference on all images
    for image_path in image_files:
        # Start timing
        start_time = time.time()
        
        # Perform inference
        results = model(image_path)
        
        # Stop timing
        end_time = time.time()
        inference_time = end_time - start_time
        inference_times.append(inference_time)
        
        # Process results and record per-class inference times
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls)  # Class ID
                cls_name = classes[cls_id]  # Class name
                class_inference_times[cls_name].append(inference_time)

    # Calculate total and average inference times
    total_inference_time = sum(inference_times)
    average_inference_time = total_inference_time / len(image_files)

    # Per-class average time
    average_class_times = {
        cls: sum(times) / len(times) if times else 0
        for cls, times in class_inference_times.items()
    }

    # Calculate average inference time over all classes
    total_class_times = sum([sum(times) for times in class_inference_times.values()])
    total_class_instances = sum([len(times) for times in class_inference_times.values()])
    average_time_over_classes = total_class_times / total_class_instances if total_class_instances > 0 else 0

    print(f"Total inference Time:{total_inference_time}")
    print(f"Average inference Time per image:{average_inference_time}")
    print(f"Average inference Time per Class:{average_class_times}")
    print(f"Average inference Time over classes:{average_time_over_classes}")

    # Return results
    return {
        "total_inference_time": total_inference_time,
        "average_inference_time_per_image": average_inference_time,
        "average_inference_time_per_class": average_class_times,
        "average_inference_time_over_classes": average_time_over_classes
    }

def plot(master_image_dir, model_weights):
    image_paths = [
    os.path.join(master_image_dir, "img1.jpeg"),
    os.path.join(master_image_dir, "img2.jpeg")
    ]

    all_results = {}
    for img_path in image_paths:
        results = model_weights.predict(source=img_path, save=False)  # Don't save automatically
        all_results[img_path] = results[0]  # Assuming one image at a time

    fig, axes = plt.subplots(1, 2, figsize=(30, 20))  # Adjust as needed
    axes = axes.flatten()

    # Plot each image with predictions
    for ax, img_path in zip(axes, image_paths):
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        result = all_results[img_path]

        # Iterate through each prediction box
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            label = int(box.cls[0])
            conf = float(box.conf[0])
            draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
            draw.text((x1, y1 - 10), f"Class {label}, {conf:.2f}", fill="red")

        # Save the annotated image
        #save_path = os.path.join(output_dir, os.path.basename(img_path))
        #img.save(save_path)

        ax.imshow(img)
        ax.axis("off")

    fig.suptitle("YOLO pretrained on Dataset.", fontsize=20)
    plt.tight_layout()
    plt.show()


def main():

    test_image_dir = "PATH_TO_GIT/pictodataset/WLRI-Semantic_labels-8/test/images" 
    data_path = "PATH_TO_GIT/pictodataset/WLRI-Semantic_labels-8/data.yaml"
    master_image_dir = "PATH_TO_GIT/pictodataset/Masterbilder"
    
    #Testing
    model_weights = YOLO("PATH_TO_GIT/pictodataset/runs/detect/train3/weights/best.pt")

    results = model_weights.val(data=data_path, imgsz=640)  
    #results = model_weights.val(data=data_path, imgsz=1280)  
    print(f"Resuts: {results}")

    # Inference times
    measure_inference_time(model_weights, images_path=test_image_dir)

    #mAP75 Score
    mAP75 = results.box.map75
    print(f"mAP75: {mAP75}")
    
    # Run predictions on master images and plot
    plot_true = False 
    if plot_true:
        plot(master_image_dir, model_weights)

if __name__ == '__main__':
    main()


