import cv2
import numpy as np
import sys


def match_template(image,template,threshold):

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

    processed_image = binarize(remove_gray(image))
    result = image.copy()
    template = binarize(remove_gray(template))
    w, h= template.shape

    res = cv2.matchTemplate(processed_image, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)

    points = [i for i in zip(*loc[::-1])]
    filtered = filter_close_points(points,(w+h)/4)

    for pt in filtered:
        cv2.rectangle(result, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    return result, filtered

image_path = sys.argv[1]
template_path = sys.argv[2]
threshold = float(sys.argv[3])

image = cv2.imread(image_path)
template = cv2.imread(template_path)
matchimage, matches = match_template(image,template,threshold)

cv2.imwrite("match.png",matchimage)
print(f"{len(matches)} instances of match found. Visualisation stored in match.png")
