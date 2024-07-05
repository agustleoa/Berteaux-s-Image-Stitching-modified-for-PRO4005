"""
Image stitching section, last run with python 3.11.9, 2 July.
Compiled and modified by Ágúst Leó Axelsson.

Original code developed by Corentin Berteaux for a course at CentraleSupelec. Github respository: 
https://github.com/CorentinBrtx/image-stitching.

Corentin Berteaux based the code was on Automatic Panoramic Image Stitching using Invariant Features
by MATTHEW BROWN AND DAVID G. LOWE. DOI: 
https://doi.org/10.1007/s11263-006-0002-3
"""

import argparse
import logging
import time
from pathlib import Path
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os
import shutil

# Prompt for user to select directory containing images that need stitching
def select_directory():
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window

    save_path = filedialog.askdirectory()
    root.destroy()  # Close the tkinter window after selection

    return save_path

save_path = select_directory()

if save_path:
    print("Selected directory:", save_path)
    # Use the selected directory in the rest of your code
else:
    print("No directory selected.")

# image.py
class Image:
    def __init__(self, path: str, size: int | None = None) -> None:
        """
        Image constructor.

        Args:
            path: path to the image
            size: maximum dimension to resize the image to
        """
        self.path = path
        self.image: np.ndarray = cv2.imread(path)
        if size is not None:
            h, w = self.image.shape[:2]
            if max(w, h) > size:
                if w > h:
                    self.image = cv2.resize(self.image, (size, int(h * size / w)))
                else:
                    self.image = cv2.resize(self.image, (int(w * size / h), size))

        self.keypoints = None
        self.features = None
        self.H: np.ndarray = np.eye(3)
        self.component_id: int = 0
        self.gain: np.ndarray = np.ones(3, dtype=np.float32)

    def compute_features(self, no_features_folder="no_features"):
        """Compute the features and the keypoints of the image using SIFT."""
        descriptor = cv2.SIFT_create()
        keypoints, features = descriptor.detectAndCompute(self.image, None)

        # intialising a path to save images without features, if needed.
        no_features_path = os.path.join(save_path, "no_features")

        if features is None or len(features) == 0:
            # No features computed or empty features
            try:
                # Ensure the no_features_folder exists
                if not os.path.exists(no_features_path):
                    os.makedirs(no_features_path)
                
                # Use shutil.move for better cross-platform support
                shutil.move(self.path, no_features_path)
                print(f"No features found for {self.path}. Moved to {no_features_path}")
            except Exception as e:
                print(f"Error moving {self.path} to {no_features_folder}: {e}")

        else:
            # Features computed successfully
            self.keypoints = keypoints
            self.features = features

    
# pair_match.py
class PairMatch:
    def __init__(self, image_a: Image, image_b: Image, matches: list | None = None) -> None:
        """
        Create a new PairMatch object.

        Args:
            image_a: First image of the pair
            image_b: Second image of the pair
            matches: List of matches between image_a and image_b
        """
        self.image_a = image_a
        self.image_b = image_b
        self.matches = matches
        self.H = None
        self.status = None
        self.overlap = None
        self.area_overlap = None
        self._Iab = None
        self._Iba = None
        self.matchpoints_a = None
        self.matchpoints_b = None

    def compute_homography_original(
        self, ransac_reproj_thresh: float = 5, ransac_max_iter: int = 500
    ) -> None:
        """
        Compute the homography between the two images of the pair.

        Args:
            ransac_reproj_thresh: reprojection threshold used in the RANSAC algorithm
            ransac_max_iter: number of maximum iterations for the RANSAC algorithm
        """
        self.matchpoints_a = np.float32(
            [self.image_a.keypoints[match.queryIdx].pt for match in self.matches]
        )
        self.matchpoints_b = np.float32(
            [self.image_b.keypoints[match.trainIdx].pt for match in self.matches]
        )

        self.H, self.status = cv2.findHomography(
            self.matchpoints_b,
            self.matchpoints_a,
            cv2.RANSAC,
            ransac_reproj_thresh,
            maxIters=ransac_max_iter,
        )

    def compute_homography(
        self, ransac_reproj_thresh: float = 5, ransac_max_iter: int = 500
    ) -> None:
        if len(self.matches) < 4:
            self.H = None
            self.status = None
            return

        self.matchpoints_a = np.float32(
            [self.image_a.keypoints[match.queryIdx].pt for match in self.matches]
        )
        self.matchpoints_b = np.float32(
            [self.image_b.keypoints[match.trainIdx].pt for match in self.matches]
        )

        self.H, self.status = cv2.findHomography(
            self.matchpoints_b,
            self.matchpoints_a,
            cv2.RANSAC,
            ransac_reproj_thresh,
            maxIters=ransac_max_iter,
        )

        if self.H is None:
            self.status = None

    def set_overlap_original(self) -> None:
        """Compute and set the overlap region between the two images."""
        if self.H is None:
            self.compute_homography()

        mask_a = np.ones_like(self.image_a.image[:, :, 0], dtype=np.uint8)
        mask_b = cv2.warpPerspective(
            np.ones_like(self.image_b.image[:, :, 0], dtype=np.uint8), self.H, mask_a.shape[::-1]
        )

        self.overlap = mask_a * mask_b
        self.area_overlap = self.overlap.sum()

    def set_overlap(self) -> None:
        if self.H is None:
            self.compute_homography()
        
        if self.H is None:
            # If homography computation failed, set overlap to zero
            self.overlap = np.zeros_like(self.image_a.image[:, :, 0], dtype=np.uint8)
            self.area_overlap = 0
            return

        mask_a = np.ones_like(self.image_a.image[:, :, 0], dtype=np.uint8)
        mask_b = cv2.warpPerspective(
            np.ones_like(self.image_b.image[:, :, 0], dtype=np.uint8), 
            self.H, 
            mask_a.shape[::-1]
        )

        self.overlap = mask_a * mask_b
        self.area_overlap = self.overlap.sum()

    def is_valid_original(self, alpha: float = 8, beta: float = 0.3) -> bool:
        """
        Check if the pair match is valid (i.e. if there are enough inliers with regard to the overlap region).

        Args:
            alpha: alpha parameter used in the comparison
            beta: beta parameter used in the comparison

        Returns:
            valid: True if the pair match is valid, False otherwise
        """
        if self.overlap is None:
            self.set_overlap()

        if self.status is None:
            self.compute_homography()

        matches_in_overlap = self.matchpoints_a[
            self.overlap[
                self.matchpoints_a[:, 1].astype(np.int64),
                self.matchpoints_a[:, 0].astype(np.int64),
            ]
            == 1
        ]

        return self.status.sum() > alpha + beta * matches_in_overlap.shape[0]

    def is_valid(self, alpha: float = 8, beta: float = 0.3) -> bool:
        if self.overlap is None:
            self.set_overlap()

        if self.H is None or self.status is None:
            return False

        if self.area_overlap == 0:
            return False

        matches_in_overlap = self.matchpoints_a[
            self.overlap[
                self.matchpoints_a[:, 1].astype(np.int64),
                self.matchpoints_a[:, 0].astype(np.int64),
            ]
            == 1
        ]

        return self.status.sum() > alpha + beta * matches_in_overlap.shape[0]

    def contains(self, image: Image) -> bool:
        """
        Check if the given image is contained in the pair match.

        Args:
            image: Image to check

        Returns:
            True if the given image is contained in the pair match, False otherwise
        """
        return self.image_a == image or self.image_b == image

    @property
    def Iab(self):
        if self._Iab is None:
            self.set_intensities()
        return self._Iab

    @Iab.setter
    def Iab(self, Iab):
        self._Iab = Iab

    @property
    def Iba(self):
        if self._Iba is None:
            self.set_intensities()
        return self._Iba

    @Iba.setter
    def Iba(self, Iba):
        self._Iba = Iba

    def set_intensities(self) -> None:
        """
        Compute the intensities of the two images in the overlap region.
        Used for the gain compensation calculation.
        """
        if self.overlap is None:
            self.set_overlap()

        inverse_overlap = cv2.warpPerspective(
            self.overlap, np.linalg.inv(self.H), self.image_b.image.shape[1::-1]
        )

        if self.overlap.sum() == 0:
            print(self.image_a.path, self.image_b.path)

        self._Iab = (
            np.sum(
                self.image_a.image * np.repeat(self.overlap[:, :, np.newaxis], 3, axis=2),
                axis=(0, 1),
            )
            / self.overlap.sum()
        )
        self._Iba = (
            np.sum(
                self.image_b.image * np.repeat(inverse_overlap[:, :, np.newaxis], 3, axis=2),
                axis=(0, 1),
            )
            / inverse_overlap.sum()
        )

# multi_images_matches.py
class MultiImageMatches:
    def __init__(self, images: list[Image], ratio: float = 0.75) -> None:
        """
        Create a new MultiImageMatches object.

        Args:
            images: images to compare
            ratio: ratio used for the Lowe's ratio test
        """
        self.images = images
        self.matches = {image.path: {} for image in images}
        self.ratio = ratio

    def get_matches(self, image_a: Image, image_b: Image) -> list:
        """
        Get matches for the given images.

        Args:
            image_a: First image
            image_b: Second image

        Returns:
            matches: List of matches between the two images
        """
        if image_b.path not in self.matches[image_a.path]:
            matches = self.compute_matches(image_a, image_b)
            self.matches[image_a.path][image_b.path] = matches

        return self.matches[image_a.path][image_b.path]

    def get_pair_matches(self, max_images: int = 6) -> list[PairMatch]:
        """
        Get the pair matches for the given images.

        Args:
            max_images: Number of matches maximum for each image

        Returns:
            pair_matches: List of pair matches
        """
        pair_matches = []
        for i, image_a in enumerate(self.images):
            possible_matches = sorted(
                self.images[:i] + self.images[i + 1 :],
                key=lambda image, ref=image_a: len(self.get_matches(ref, image)),
                reverse=True,
            )[:max_images]
            for image_b in possible_matches:
                if self.images.index(image_b) > i:
                    pair_match = PairMatch(image_a, image_b, self.get_matches(image_a, image_b))
                    if pair_match.is_valid():
                        pair_matches.append(pair_match)
        return pair_matches

    def compute_matches(self, image_a: Image, image_b: Image) -> list:
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        matches = []

        raw_matches = matcher.knnMatch(image_a.features, image_b.features, 2)
        matches = []

        for match in raw_matches:
            if len(match) == 2:
                m, n = match
                # ensure the distance is within a certain ratio of each
                # other (i.e. Lowe's ratio test)
                if m.distance < n.distance * self.ratio:
                    matches.append(m)

        return matches

# build_homographies.py
def build_homographies(
    connected_components: list[list[Image]], pair_matches: list[PairMatch]
) -> None:
    """
    Build homographies for each image of each connected component, using the pair matches.
    The homographies are saved in the images themselves.

    Args:
        connected_components: The connected components of the panorama
        pair_matches: The valid pair matches
    """
    for connected_component in connected_components:
        component_matches = [
            pair_match for pair_match in pair_matches if pair_match.image_a in connected_component
        ]

        images_added = set()
        current_homography = np.eye(3)

        pair_match = component_matches[0]
        pair_match.compute_homography()

        nb_pairs = len(pair_matches)

        if sum(
            [
                10 * (nb_pairs - i)
                for i, match in enumerate(pair_matches)
                if match.contains(pair_match.image_a)
            ]
        ) > sum(
            [
                10 * (nb_pairs - i)
                for i, match in enumerate(pair_matches)
                if match.contains(pair_match.image_b)
            ]
        ):
            pair_match.image_a.H = np.eye(3)
            pair_match.image_b.H = pair_match.H
        else:
            pair_match.image_b.H = np.eye(3)
            pair_match.image_a.H = np.linalg.inv(pair_match.H)

        images_added.add(pair_match.image_a)
        images_added.add(pair_match.image_b)

        while len(images_added) < len(connected_component):
            for pair_match in component_matches:

                if pair_match.image_a in images_added and pair_match.image_b not in images_added:
                    pair_match.compute_homography()
                    homography = pair_match.H @ current_homography
                    pair_match.image_b.H = pair_match.image_a.H @ homography
                    images_added.add(pair_match.image_b)
                    break

                if pair_match.image_a not in images_added and pair_match.image_b in images_added:
                    pair_match.compute_homography()
                    homography = np.linalg.inv(pair_match.H) @ current_homography
                    pair_match.image_a.H = pair_match.image_b.H @ homography
                    images_added.add(pair_match.image_a)
                    break

# connected_components.py
def find_connected_components(pair_matches: list[PairMatch]) -> list[list[PairMatch]]:
    """
    Find the connected components of the given pair matches.

    Args:
        pair_matches: The list of pair matches.

    Returns:
        connected_components: List of connected components.
    """
    connected_components = []
    pair_matches_to_check = pair_matches.copy()
    component_id = 0
    while len(pair_matches_to_check) > 0:
        pair_match = pair_matches_to_check.pop(0)
        connected_component = {pair_match.image_a, pair_match.image_b}
        size = len(connected_component)
        stable = False
        while not stable:
            i = 0
            while i < len(pair_matches_to_check):
                pair_match = pair_matches_to_check[i]
                if (
                    pair_match.image_a in connected_component
                    or pair_match.image_b in connected_component
                ):
                    connected_component.add(pair_match.image_a)
                    connected_component.add(pair_match.image_b)
                    pair_matches_to_check.pop(i)
                else:
                    i += 1
            stable = size == len(connected_component)
            size = len(connected_component)
        connected_components.append(list(connected_component))
        for image in connected_component:
            image.component_id = component_id
        component_id += 1

    return connected_components

# utils.py
def apply_homography(H: np.ndarray, point: np.ndarray) -> np.ndarray:
    """
    Apply a homography to a point.

    Args:
        H: Homography matrix
        point: Point to apply the homography to, with shape (2,1)

    Returns:
        new_point: Point after applying the homography, with shape (2,1)
    """
    point = np.asarray([[point[0][0], point[1][0], 1]]).T
    new_point = H @ point
    return new_point[0:2] / new_point[2]
def apply_homography_list(H: np.ndarray, points: list[np.ndarray]) -> list[np.ndarray]:
    """
    Apply a homography to a list of points.

    Args:
        H: Homography matrix
        points: List of points to apply the homography to, each with shape (2,1)

    Returns:
        new_points: List of points after applying the homography, each with shape (2,1)
    """
    return [apply_homography(H, point) for point in points]
def get_new_corners(image: np.ndarray, H: np.ndarray) -> list[np.ndarray]:
    """
    Get the new corners of an image after applying a homography.

    Args:
        image: Original image
        H: Homography matrix

    Returns:
        corners: Corners of the image after applying the homography
    """
    top_left = np.asarray([[0, 0]]).T
    top_right = np.asarray([[image.shape[1], 0]]).T
    bottom_left = np.asarray([[0, image.shape[0]]]).T
    bottom_right = np.asarray([[image.shape[1], image.shape[0]]]).T

    return apply_homography_list(H, [top_left, top_right, bottom_left, bottom_right])
def get_offset(corners: list[np.ndarray]) -> np.ndarray:
    """
    Get offset matrix so that all corners are in positive coordinates.

    Args:
        corners: List of corners of the image

    Returns:
        offset: Offset matrix
    """
    top_left, top_right, bottom_left = corners[:3]
    return np.array(
        [
            [1, 0, max(0, -float(min(top_left[0], bottom_left[0])))],
            [0, 1, max(0, -float(min(top_left[1], top_right[1])))],
            [0, 0, 1],
        ],
        np.float32,
    )
def get_new_size(corners_images: list[list[np.ndarray]]) -> tuple[int, int]:
    """
    Get the size of the image that would contain all the given corners.

    Args:
        corners_images: List of corners of the images
            (i.e. corners_images[i] is the list of corners of image i)

    Returns:
        (width, height): Size of the image
    """
    top_right_x = np.max([corners_image[1][0] for corners_image in corners_images])
    bottom_right_x = np.max([corners_images[3][0] for corners_images in corners_images])

    bottom_left_y = np.max([corners_images[2][1] for corners_images in corners_images])
    bottom_right_y = np.max([corners_images[3][1] for corners_images in corners_images])

    width = int(np.ceil(max(bottom_right_x, top_right_x)))
    height = int(np.ceil(max(bottom_right_y, bottom_left_y)))

    width = min(width, 5000)
    height = min(height, 4000)

    return width, height
def get_new_parameters(
    panorama: np.ndarray, image: np.ndarray, H: np.ndarray
) -> tuple[tuple[int, int], np.ndarray]:
    """
    Get the new size of the image and the offset matrix.

    Args:
        panorama: Current panorama
        image: Image to add to the panorama
        H: Homography matrix for the image

    Returns:
        size, offset: Size of the new image and offset matrix.
    """
    corners = get_new_corners(image, H)
    added_offset = get_offset(corners)

    corners_image = get_new_corners(image, added_offset @ H)
    if panorama is None:
        size = get_new_size([corners_image])
    else:
        corners_panorama = get_new_corners(panorama, added_offset)
        size = get_new_size([corners_image, corners_panorama])

    return size, added_offset
def single_weights_array(size: int) -> np.ndarray:
    """
    Create a 1D weights array.

    Args:
        size: Size of the array

    Returns:
        weights: 1D weights array
    """
    if size % 2 == 1:
        return np.concatenate(
            [np.linspace(0, 1, (size + 1) // 2), np.linspace(1, 0, (size + 1) // 2)[1:]]
        )
    else:
        return np.concatenate([np.linspace(0, 1, size // 2), np.linspace(1, 0, size // 2)])
def single_weights_matrix(shape: tuple[int]) -> np.ndarray:
    """
    Create a 2D weights matrix.

    Args:
        shape: Shape of the matrix

    Returns:
        weights: 2D weights matrix
    """
    return (
        single_weights_array(shape[0])[:, np.newaxis]
        @ single_weights_array(shape[1])[:, np.newaxis].T
    )

# multiband_blending.py
def add_weights(
    weights_matrix: np.ndarray, image: Image, offset: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Add the weights corresponding to the given image to the given existing weights matrix.

    Args:
        weights_matrix: Existing weights matrix
        image: New image to add to the weights matrix
        offset: Offset already applied to the weights matrix

    Returns:
        weights_matrix, offset: The updated weights matrix and the updated offset
    """
    H = offset @ image.H
    size, added_offset = get_new_parameters(weights_matrix, image.image, H)

    weights = single_weights_matrix(image.image.shape)
    weights = cv2.warpPerspective(weights, added_offset @ H, size)[:, :, np.newaxis]

    if weights_matrix is None:
        weights_matrix = weights
    else:
        weights_matrix = cv2.warpPerspective(weights_matrix, added_offset, size)

        if len(weights_matrix.shape) == 2:
            weights_matrix = weights_matrix[:, :, np.newaxis]

        weights_matrix = np.concatenate([weights_matrix, weights], axis=2)

    return weights_matrix, added_offset @ offset
def get_max_weights_matrix(images: list[Image]) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the maximum weights matrix for the given images.

    Args:
        images: List of images to compute the maximum weights matrix for

    Returns:
        max_weights_matrix: Maximum weights matrix
        offset: Offset matrix
    """
    weights_matrix = None
    offset = np.eye(3)

    for image in images:
        weights_matrix, offset = add_weights(weights_matrix, image, offset)

    weights_maxes = np.max(weights_matrix, axis=2)[:, :, np.newaxis]
    max_weights_matrix = np.where(
        np.logical_and(weights_matrix == weights_maxes, weights_matrix > 0), 1.0, 0.0
    )

    max_weights_matrix = np.transpose(max_weights_matrix, (2, 0, 1))

    return max_weights_matrix, offset
def get_cropped_weights(
    images: list[Image], weights: np.ndarray, offset: np.ndarray
) -> list[np.ndarray]:
    """
    Convert a global weights matrix to a list of weights matrices for each image,
    where each weight matrix is the size of the corresponding image.

    Args:
        images: List of images to convert the weights matrix for
        weights: Global weights matrix
        offset: Offset matrix

    Returns:
        cropped_weights: List of weights matrices for each image
    """
    cropped_weights = []
    for i, image in enumerate(images):
        cropped_weights.append(
            cv2.warpPerspective(
                weights[i], np.linalg.inv(offset @ image.H), image.image.shape[:2][::-1]
            )
        )

    return cropped_weights
def build_band_panorama(
    images: list[Image],
    weights: list[np.ndarray],
    bands: list[np.ndarray],
    offset: np.ndarray,
    size: tuple[int, int],
) -> np.ndarray:
    """
    Build a panorama from the given bands and weights matrices.
    The images are needed for their homographies.

    Args:
        images: Images to build the panorama from
        weights: Weights matrices for each image
        bands: Bands for each image
        offset: Offset matrix
        size: Size of the panorama

    Returns:
        panorama: Panorama for the given bands and weights
    """
    pano_weights = np.zeros(size)
    pano_bands = np.zeros((*size, 3))

    for i, image in enumerate(images):
        weights_at_scale = cv2.warpPerspective(weights[i], offset @ image.H, size[::-1])
        pano_weights += weights_at_scale
        pano_bands += weights_at_scale[:, :, np.newaxis] * cv2.warpPerspective(
            bands[i], offset @ image.H, size[::-1]
        )

    return np.divide(
        pano_bands, pano_weights[:, :, np.newaxis], where=pano_weights[:, :, np.newaxis] != 0
    )
def multi_band_blending(images: list[Image], num_bands: int, sigma: float) -> np.ndarray:
    """
    Build a panorama from the given images using multi-band blending.

    Args:
        images: Images to build the panorama from
        num_bands: Number of bands to use for multi-band blending
        sigma: Standard deviation for the multi-band blending

    Returns:
        panorama: Panorama after multi-band blending
    """
    max_weights_matrix, offset = get_max_weights_matrix(images)
    size = max_weights_matrix.shape[1:]

    max_weights = get_cropped_weights(images, max_weights_matrix, offset)

    weights = [[cv2.GaussianBlur(max_weights[i], (0, 0), 2 * sigma) for i in range(len(images))]]
    sigma_images = [cv2.GaussianBlur(image.image, (0, 0), sigma) for image in images]
    bands = [
        [
            np.where(
                images[i].image.astype(np.int64) - sigma_images[i].astype(np.int64) > 0,
                images[i].image - sigma_images[i],
                0,
            )
            for i in range(len(images))
        ]
    ]

    for k in range(1, num_bands - 1):
        sigma_k = np.sqrt(2 * k + 1) * sigma
        weights.append(
            [cv2.GaussianBlur(weights[-1][i], (0, 0), sigma_k) for i in range(len(images))]
        )

        old_sigma_images = sigma_images

        sigma_images = [
            cv2.GaussianBlur(old_sigma_image, (0, 0), sigma_k)
            for old_sigma_image in old_sigma_images
        ]
        bands.append(
            [
                np.where(
                    old_sigma_images[i].astype(np.int64) - sigma_images[i].astype(np.int64) > 0,
                    old_sigma_images[i] - sigma_images[i],
                    0,
                )
                for i in range(len(images))
            ]
        )

    weights.append([cv2.GaussianBlur(weights[-1][i], (0, 0), sigma_k) for i in range(len(images))])
    bands.append([sigma_images[i] for i in range(len(images))])

    panorama = np.zeros((*max_weights_matrix.shape[1:], 3))

    for k in range(0, num_bands):
        panorama += build_band_panorama(images, weights[k], bands[k], offset, size)
        panorama[panorama < 0] = 0
        panorama[panorama > 255] = 255

    return panorama

# gain_compensation.py
def set_gain_compensations(
    images: list[Image], pair_matches: list[PairMatch], sigma_n: float = 10.0, sigma_g: float = 0.1
) -> None:
    """
    Compute the gain compensation for each image, and save it into the images objects.

    Args:
        images: Images of the panorama
        pair_matches: Pair matches between the images
        sigma_n: Standard deviation of the normalized intensity error
        sigma_g: Standard deviation of the gain
    """
    coefficients = []
    results = []

    for k, image in enumerate(images):
        coefs = [np.zeros(3) for _ in range(len(images))]
        result = np.zeros(3)

        for pair_match in pair_matches:
            if pair_match.image_a == image:
                coefs[k] += pair_match.area_overlap * (
                    (2 * pair_match.Iab ** 2 / sigma_n ** 2) + (1 / sigma_g ** 2)
                )

                i = images.index(pair_match.image_b)
                coefs[i] -= (
                    (2 / sigma_n ** 2) * pair_match.area_overlap * pair_match.Iab * pair_match.Iba
                )

                result += pair_match.area_overlap / sigma_g ** 2

            elif pair_match.image_b == image:
                coefs[k] += pair_match.area_overlap * (
                    (2 * pair_match.Iba ** 2 / sigma_n ** 2) + (1 / sigma_g ** 2)
                )

                i = images.index(pair_match.image_a)
                coefs[i] -= (
                    (2 / sigma_n ** 2) * pair_match.area_overlap * pair_match.Iab * pair_match.Iba
                )

                result += pair_match.area_overlap / sigma_g ** 2

        coefficients.append(coefs)
        results.append(result)

    coefficients = np.array(coefficients)
    results = np.array(results)

    gains = np.zeros_like(results)

    for channel in range(coefficients.shape[2]):
        coefs = coefficients[:, :, channel]
        res = results[:, channel]

        gains[:, channel] = np.linalg.solve(coefs, res)

    max_pixel_value = np.max([image.image for image in images])

    if gains.max() * max_pixel_value > 255:
        gains = gains / (gains.max() * max_pixel_value) * 255

    for i, image in enumerate(images):
        image.gain = gains[i]

# simple_blending.py
def add_image(
    panorama: np.ndarray, image: Image, offset: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Add a new image to the panorama using the provided offset and weights.

    Args:
        panorama: Existing panorama
        image: Image to add to the panorama
        offset: Offset already applied to the panorama
        weights: Weights matrix of the panorama

    Returns:
        panorama: Panorama with the new image
        offset: New offset matrix
        weights: New weights matrix
    """
    H = offset @ image.H
    size, added_offset = get_new_parameters(panorama, image.image, H)

    new_image = cv2.warpPerspective(image.image, added_offset @ H, size)

    if panorama is None:
        panorama = np.zeros_like(new_image)
        weights = np.zeros_like(new_image)
    else:
        panorama = cv2.warpPerspective(panorama, added_offset, size)
        weights = cv2.warpPerspective(weights, added_offset, size)

    image_weights = single_weights_matrix(image.image.shape)
    image_weights = np.repeat(
        cv2.warpPerspective(image_weights, added_offset @ H, size)[:, :, np.newaxis], 3, axis=2
    )

    normalized_weights = np.zeros_like(weights)
    normalized_weights = np.divide(
        weights, (weights + image_weights), where=weights + image_weights != 0
    )

    panorama = np.where(
        np.logical_and(
            np.repeat(np.sum(panorama, axis=2)[:, :, np.newaxis], 3, axis=2) == 0,
            np.repeat(np.sum(new_image, axis=2)[:, :, np.newaxis], 3, axis=2) == 0,
        ),
        0,
        new_image * (1 - normalized_weights) + panorama * normalized_weights,
    ).astype(np.uint8)

    new_weights = (weights + image_weights) / (weights + image_weights).max()

    return panorama, added_offset @ offset, new_weights
def simple_blending(images: list[Image]) -> np.ndarray:
    """
    Build a panorama from the given images using simple blending.

    Args:
        images: Images to build the panorama from

    Returns:
        panorama: Panorama of the given images
    """
    panorama = None
    weights = None
    offset = np.eye(3)
    for image in images:
        panorama, offset, weights = add_image(panorama, image, offset, weights)

    return panorama

# main.py (modified)
parser = argparse.ArgumentParser(
    description="Create panoramas from a set of images. \
                 All the images must be in the same directory. \
                 Multiple panoramas can be created at once"
)
parser.add_argument(
    "-mbb",
    "--multi-band-blending",
    action="store_true",
    help="use multi-band blending instead of simple blending",
)
parser.add_argument(
    "--size", type=int, help="maximum dimension to resize the images to"
)
parser.add_argument(
    "--num-bands", type=int, default=5, help="number of bands for multi-band blending"
)
parser.add_argument(
    "--mbb-sigma", type=float, default=1, help="sigma for multi-band blending"
)
parser.add_argument(
    "--gain-sigma-n", type=float, default=10, help="sigma_n for gain compensation"
)
parser.add_argument(
    "--gain-sigma-g", type=float, default=0.1, help="sigma_g for gain compensation"
)
parser.add_argument(
    "-v", "--verbose", action="store_true", help="increase output verbosity"
)
args = vars(parser.parse_args())
if args["verbose"]:
    logging.basicConfig(level=logging.INFO)
logging.info("Gathering images...")
valid_images_extensions = {".jpg", ".png", ".bmp", ".jpeg", ".tiff"}
image_paths = [
    str(filepath)
    # for filepath in args["data_dir"].iterdir()
    for filepath in Path(save_path).iterdir()
    if filepath.suffix.lower() in valid_images_extensions
]
images = [Image(path, args.get("size")) for path in image_paths]
logging.info("Found %d images", len(images))
logging.info("Computing features with SIFT...")
for image in images:
    image.compute_features()
# running the same thing again to reinitialise images in case some have no features
image_paths = [
    str(filepath)
    # for filepath in args["data_dir"].iterdir()
    for filepath in Path(save_path).iterdir()
    if filepath.suffix.lower() in valid_images_extensions
]
images = [Image(path, args.get("size")) for path in image_paths]
logging.info("Found %d images", len(images))
logging.info("Computing features with SIFT...")
for image in images:
    image.compute_features()
# rest of stuff
logging.info("Matching images with features...")
matcher = MultiImageMatches(images)
pair_matches: list[PairMatch] = matcher.get_pair_matches()
pair_matches.sort(key=lambda pair_match: len(pair_match.matches), reverse=True)
logging.info("Finding connected components...")
connected_components = find_connected_components(pair_matches)
logging.info("Found %d connected components", len(connected_components))
logging.info("Building homographies...")
build_homographies(connected_components, pair_matches)
time.sleep(0.1)
logging.info("Computing gain compensations...")
for connected_component in connected_components:
    component_matches = [
        pair_match
        for pair_match in pair_matches
        if pair_match.image_a in connected_component
    ]

    set_gain_compensations(
        connected_component,
        component_matches,
        sigma_n=args["gain_sigma_n"],
        sigma_g=args["gain_sigma_g"],
    )
time.sleep(0.1)
for image in images:
    image.image = (image.image * image.gain[np.newaxis, np.newaxis, :]).astype(np.uint8)
results = []
if args["multi_band_blending"]:
    logging.info("Applying multi-band blending...")
    results = [
        multi_band_blending(
            connected_component,
            num_bands=args["num_bands"],
            sigma=args["mbb_sigma"],
        )
        for connected_component in connected_components
    ]
else:
    logging.info("Applying simple blending...")
    results = [
        simple_blending(connected_component)
        for connected_component in connected_components
    ]
results_dir = Path(save_path) / "results"
logging.info("Saving results to %s", results_dir)
results_dir.mkdir(exist_ok=True, parents=True)
for i, result in enumerate(results):
    cv2.imwrite(str(results_dir / f"stitched_{i}.jpg"), result)

