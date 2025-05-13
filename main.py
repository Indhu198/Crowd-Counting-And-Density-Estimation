import cv2
# Load image
image_path = "sample.jpg"  # Replace with your image path
image = cv2.imread(image_path)
if image is None:
    print("Image not found or path is incorrect.")
    exit()
# Initialize HOG person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
# Detect people with weights
(rects, weights) = hog.detectMultiScale(image,winStride=(4, 4),  padding=(8, 8), scale=1.05)

# Draw bounding boxes and labels
for i, (x, y, w, h) in enumerate(rects):
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    label = f"Person {i+1}"
    cv2.putText(image, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                

# Display total count
count = len(rects)
cv2.putText(image, f"Total number of People : {count}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# Save the output image
cv2.imwrite("crowd_detected.jpg", image)

# Show result
cv2.imshow("Crowd Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
