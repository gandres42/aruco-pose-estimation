import cv2

SONAR_BINS = 512
SONAR_AZIMUTH = 60
SONAR_VERTICAL_APERTURE = 12
SONAR_MAX_DIST = 5
SCALING_ITERATIONS = 50

class SonarProcessor:
    def __init__():
        pass

    def sonar_unwarp(self, sonar_img: np.ndarray):
        output_height = SONAR_MAX_DIST * 100

        if len(sonar_img.shape) == 3: sonar_img = cv2.cvtColor(sonar_img, cv2.COLOR_BGR2GRAY)
        sonar_img = cv2.flip(sonar_img, 0)
        height, width = sonar_img.shape
        
        # Create output image for unwarped sonar
        output_size = (2 * output_height)
        unwarped = np.zeros((output_size, output_size), dtype=np.uint8)
        
        # Convert SONAR_AZIMUTH from degrees to radians
        azimuth_rad = math.radians(SONAR_AZIMUTH)
        
        # Create coordinate grids for the output image
        center = output_size // 2
        y_coords, x_coords = np.mgrid[0:output_size, 0:output_size]
        
        # Convert cartesian coordinates to polar
        x_rel = x_coords - center
        y_rel = y_coords - center
        
        # Calculate radius and angle
        radius = np.sqrt(x_rel**2 + y_rel**2)
        angle = np.arctan2(x_rel, y_rel)  # Note: x,y swapped for sonar orientation
        
        # Map to sonar image coordinates
        # Radius maps to height (range bins)
        # Angle maps to width (azimuth bins)
        max_radius = center
        
        # Only process points within valid radius and angle range
        valid_mask = (radius <= max_radius) & (np.abs(angle) <= azimuth_rad / 2)
        
        # Convert to sonar image indices
        r_indices = (radius[valid_mask] * (height - 1) / max_radius).astype(int)
        angle_normalized = (angle[valid_mask] + azimuth_rad / 2) / azimuth_rad
        theta_indices = (angle_normalized * (width - 1)).astype(int)
        
        # Clamp indices to valid range
        r_indices = np.clip(r_indices, 0, height - 1)
        theta_indices = np.clip(theta_indices, 0, width - 1)
        
        # Sample from the original sonar image
        unwarped[y_coords[valid_mask], x_coords[valid_mask]] = sonar_img[r_indices, theta_indices]
        
        # Flip the unwarped result to correct orientation
        unwarped = cv2.flip(unwarped, 0)
        half_width = int(output_height * math.sin(math.radians(SONAR_AZIMUTH / 2)))
        start_idx = (unwarped.shape[1] // 2) - half_width - 1
        end_idx = (unwarped.shape[1] // 2) + half_width + 1
        unwarped = unwarped[0:output_height, start_idx:end_idx]
        unwarped = cv2.flip(unwarped, 0)

        return unwarped

    def akaze_matching(self, img_a, img_b):
        akaze = cv2.AKAZE_create( # type: ignore
            descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB_UPRIGHT,
            threshold=0.0001,  # Lower threshold = more features (default: 0.001)
            diffusivity=cv2.KAZE_DIFF_PM_G1
        )
        kp_a, des_a = akaze.detectAndCompute(img_a, None)
        kp_b, des_b = akaze.detectAndCompute(img_b, None)        

        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        if des_a is not None and des_b is not None:
            # Use KNN matching to find 2 best matches for each descriptor
            knn_matches = bf.knnMatch(des_a, des_b, k=2)
            
            # Apply Lowe's ratio test to filter out ambiguous matches
            matches = []
            for match_pair in knn_matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:  # Ratio threshold
                        matches.append(m)
        else:
            matches = []

        if len(matches) >= 5:
            src_pts = np.float32([kp_a[m.queryIdx].pt for m in matches]).reshape(-1, 2) # type: ignore
            dst_pts = np.float32([kp_b[m.trainIdx].pt for m in matches]).reshape(-1, 2) # type: ignore

            # Use RANSAC to estimate a Euclidean transform and reject outliers
            _, inliers = ransac(
                (src_pts, dst_pts),
                EuclideanTransform,
                min_samples=2,
                residual_threshold=1,
                max_trials=10
            )
            # Keep only inlier matches
            if inliers is None:
                matches = []
            else:
                matches = [m for i, m in enumerate(matches) if inliers[i]]

        return matches, kp_a, kp_b

    def coordinate_scaling(self, matches, kp_a, kp_b, img_shape):
        # region: extract the coordinates of matched keypoints in img_a and img_b
        matched_coords = []
        for match in matches:
            pt_a = kp_a[match.queryIdx].pt
            pt_b = kp_b[match.trainIdx].pt
            matched_coords.append((pt_a, pt_b))
        # endregion

        y_max, x_max = img_shape
        scaled_coords = []

        for coord_set in matched_coords:
            scaled_coord_set = []
            for coord in coord_set:                
                # update x to center on image middle
                x, y = coord
                x = x - (x_max / 2)

                # calculate r, theta
                r = math.sqrt(math.pow(x, 2) + math.pow(y, 2))
                theta = math.atan2(y, x)

                # scale r based on max distance
                r = (r / y_max) * SONAR_MAX_DIST

                # convert back to cartesian
                x_new = r * math.cos(theta)
                y_new = r * math.sin(theta)

                # append to scaled coord set
                scaled_coord_set.append((x_new, y_new))

            # append new coord set
            scaled_coords.append(tuple(scaled_coord_set))
        
        return scaled_coords

if __name__ == '__main__':
    