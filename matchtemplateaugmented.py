import cv2
import numpy as np
import imutils
import sys

def augment(image):
    def scale_image(image, scale_range):
        """Scales an image by a given range of percentage.

        Args:
            image: The image to be scaled.
            scale_range: A range of percentage to scale the image by.

        Returns:
            A list of images scaled by the given range of percentage.
        """

        scaled_images = []
        for scale in scale_range:
            scaled_image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
            scaled_images.append(scaled_image)
        return scaled_images

    def rotate_images(image):
        # Define the rotation angles
        rotation_angles = np.arange(0, 360, 4)

        # List to store rotated images
        rotated_images = []

        # Perform rotations
        for angle in rotation_angles:
            # Rotate the image
            rotated_image = imutils.rotate_bound(image, angle)
            # Append the rotated image to the list
            rotated_images.append(rotated_image)

        return rotated_images

    images = []
    scalings = scale_image(image,(0.8,0.9,1,1.1,1.2))
    for im in scalings:
        rotaions = rotate_images(im)
        images.extend(rotaions)
    return images

def match_template_augmented(image,template,threshold):

    def remove_gray(img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        image_modified = img.copy()
        mask = (hsv[:, :, 1] < 5)
        image_modified[mask] = [255, 255, 255]
        return image_modified

    def binarize(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = (gray == 255)
        gray[~mask] = 0
        return gray

    def filter_close_points(points, min_distance):
        filtered_points = []
        points = np.array(points)

        for i in range(len(points)):
            current_point = np.array(points[i])

            # Check the distance to all other points
            if filtered_points:
                distances = np.linalg.norm(filtered_points - current_point, axis=1)

                # Check if the minimum distance is satisfied
                if np.all(distances > 10):
                    filtered_points.append(points[i])
            else:
                filtered_points.append(points[i])

        return filtered_points

    temps = augment(template)
    processed_image = binarize(remove_gray(image))
    result = image.copy()
    points = []
    for i,template in enumerate(temps):
        print(i,len(points))
        template = binarize(remove_gray(template))
        w, h= template.shape
        res = cv2.matchTemplate(processed_image, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        points.extend([i for i in zip(*loc[::-1])])

    filtered = filter_close_points(points,(w+h)/4)

    for pt in filtered:
        cv2.rectangle(result, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    return result, filtered


image_path = sys.argv[1]
template_path = sys.argv[2]
threshold = float(sys.argv[3])

image = cv2.imread(image_path)
template = cv2.imread(template_path)
matchimage, matches = match_template_augmented(image,template,threshold)

cv2.imwrite("match.png",matchimage)
print(f"{len(matches)} instances of match found. Visualisation stored in match.png")
