# **Superannotate** :-

- When a project's status is set to 'completed' in the Superannotate Tool, it triggers a webhook. This sends a request to the URL of the first cloud function, which is ***'Push images to GCP cloud function'.***

# Push images to GCP cloud function :-

- After the HTTPS trigger, the cloud function activates. It starts preparing an export from the respective SuperAnnotate project. It retrieves images with annotation statuses of 'Quality Check' and 'Completed'. Then, it pushes these images to GCP.
- Lastly, it will send a webhook to the subsequent cloud function.

# Firebase update cloud function :-

- This function retrieves the annotations of the respective project, converts them into COCO format, and starts calculating image attributes such as brightness, contrast, sharpness, average RGB value, and density, etc....
- These attributes can be used for data extrapolation and YOLOv8 training.

# Firebase :-

- Now, all necessary data for retraining is available. Each document refers to an image and its corresponding annotations and attribute details.

# Download Data from Firebase for Training :-

The above GitHub repo has the script to download and prepare the data for training Yolox and yolov8 format.
